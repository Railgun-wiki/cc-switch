//! Copilot Fusion Session Manager
//!
//! Provides process-level session caching, multi-turn decision reuse, single-flight resolution,
//! bounded LRU/TTL lifecycle management, and account isolation for Copilot Fusion.

use super::error::ProxyError;
use super::providers::copilot_fusion::{
    CopilotAutoMultiTurnConfig, CopilotAutoResponse, CopilotFusionClient, ParsedSessionToken,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};

/// Maximum cached fusion sessions (LRU capacity)
pub const MAX_CACHED_SESSIONS: usize = 256;

/// Idle TTL before an unused session is evicted (30 minutes)
pub const SESSION_IDLE_TTL: Duration = Duration::from_secs(30 * 60);

/// Cache key uniquely identifying a Copilot Fusion conversational session
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FusionSessionKey {
    pub app_namespace: String,
    pub provider_id: String,
    pub account_id: String,
    pub api_base: String,
    pub credential_generation: u64,
    pub client_session_id: String,
    pub branch_id: Option<String>,
}

/// Cloned snapshot of an active resolution held by executing requests
#[derive(Clone)]
pub struct FusionSessionSnapshot {
    pub selected_model: String,
    pub session_token: String,
    pub created_at: Instant,
    pub expires_at: Option<Instant>,
    pub multi_turn: Option<CopilotAutoMultiTurnConfig>,
    pub multi_turn_mode: Option<String>,
    pub credential_generation: u64,
}

impl std::fmt::Debug for FusionSessionSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusionSessionSnapshot")
            .field("selected_model", &self.selected_model)
            .field("session_token", &"[REDACTED]")
            .field("created_at", &self.created_at)
            .field("expires_at", &self.expires_at)
            .field("multi_turn", &self.multi_turn)
            .field("multi_turn_mode", &self.multi_turn_mode)
            .field("credential_generation", &self.credential_generation)
            .finish()
    }
}

impl FusionSessionSnapshot {
    /// Returns true if this resolution is still valid (not expired)
    pub fn is_valid(&self) -> bool {
        if let Some(exp) = self.expires_at {
            if Instant::now() >= exp {
                return false;
            }
        }
        true
    }
}

/// Internal session record stored in the manager cache
#[derive(Debug, Clone)]
struct FusionSessionRecord {
    snapshot: FusionSessionSnapshot,
    last_accessed: Instant,
    #[allow(dead_code)]
    turns_count: u32,
}

/// Process-level manager for Copilot Fusion sessions
#[derive(Default)]
pub struct FusionSessionManager {
    sessions: RwLock<HashMap<FusionSessionKey, FusionSessionRecord>>,
    single_flight: Mutex<HashMap<FusionSessionKey, Arc<Mutex<()>>>>,
}

#[allow(dead_code)]
impl FusionSessionManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fast lookup: returns an active snapshot if present and valid
    pub async fn get(&self, key: &FusionSessionKey) -> Option<FusionSessionSnapshot> {
        let mut sessions = self.sessions.write().await;
        if let Some(record) = sessions.get_mut(key) {
            let now = Instant::now();
            if now.duration_since(record.last_accessed) > SESSION_IDLE_TTL {
                return None;
            }
            if !record.snapshot.is_valid() {
                return None;
            }
            if let Some(ref mt) = record.snapshot.multi_turn {
                if !mt.enabled && record.turns_count > 0 {
                    return None;
                }
            }
            record.last_accessed = now;
            return Some(record.snapshot.clone());
        }
        None
    }

    /// Resolves or retrieves a cached resolution with single-flight concurrency
    pub async fn get_or_resolve(
        &self,
        key: &FusionSessionKey,
        copilot_token: &str,
        prompt: &str,
        has_image: bool,
    ) -> Result<FusionSessionSnapshot, ProxyError> {
        // Fast path: cached & valid
        if let Some(snapshot) = self.get(key).await {
            return Ok(snapshot);
        }

        // Single-flight coordination per unique FusionSessionKey
        let gate = {
            let mut locks = self.single_flight.lock().await;
            locks
                .entry(key.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };

        let result = async {
            let _guard = gate.lock().await;

            // Double-check cache after acquiring lock
            if let Some(snapshot) = self.get(key).await {
                return Ok(snapshot);
            }

            // Network call to POST /auto
            log::info!(
                "[Copilot/Fusion] Resolving /auto for session='{}' app='{}'",
                key.client_session_id,
                key.app_namespace
            );

            let response = CopilotFusionClient::resolve(
                &crate::proxy::http_client::get(),
                &key.api_base,
                copilot_token,
                prompt,
                has_image,
            )
            .await?;

            let snapshot = self.create_snapshot_from_response(key, response);

            // Store into cache and apply LRU / TTL eviction
            self.insert(key.clone(), snapshot.clone()).await;

            Ok(snapshot)
        }
        .await;

        // Clean up lock from single_flight map if no other callers are waiting and it wasn't replaced
        {
            let mut locks = self.single_flight.lock().await;
            if Arc::strong_count(&gate) <= 2
                && locks.get(key).is_some_and(|l| Arc::ptr_eq(l, &gate))
            {
                locks.remove(key);
            }
        }

        result
    }

    fn create_snapshot_from_response(
        &self,
        key: &FusionSessionKey,
        response: CopilotAutoResponse,
    ) -> FusionSessionSnapshot {
        let now = Instant::now();
        let mut expires_at = None;
        let mut parsed_exp = false;

        if let Some(parsed) = ParsedSessionToken::parse(&response.session_token) {
            if let Some(exp_unix) = parsed.exp {
                parsed_exp = true;
                let system_now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                if exp_unix > system_now {
                    let secs = exp_unix - system_now;
                    expires_at = Some(now + Duration::from_secs(secs));
                } else {
                    // Token exp is already in the past
                    expires_at = Some(now.checked_sub(Duration::from_secs(1)).unwrap_or(now));
                }
            }
        }

        // Fallback default expiration ONLY if JWT lacked exp claim: 24 hours
        if !parsed_exp {
            expires_at = Some(now + Duration::from_secs(24 * 3600));
        }

        FusionSessionSnapshot {
            selected_model: response.selected_model.model_id().to_string(),
            session_token: response.session_token,
            created_at: now,
            expires_at,
            multi_turn: response.multi_turn,
            multi_turn_mode: response.multi_turn_mode,
            credential_generation: key.credential_generation,
        }
    }

    pub async fn insert(&self, key: FusionSessionKey, snapshot: FusionSessionSnapshot) {
        let mut sessions = self.sessions.write().await;
        let now = Instant::now();

        // Evict expired / idle entries
        sessions.retain(|_, rec| {
            now.duration_since(rec.last_accessed) <= SESSION_IDLE_TTL && rec.snapshot.is_valid()
        });

        // Evict oldest by last_accessed if capacity exceeded
        while sessions.len() >= MAX_CACHED_SESSIONS {
            if let Some(oldest_key) = sessions
                .iter()
                .min_by_key(|(_, rec)| rec.last_accessed)
                .map(|(k, _)| k.clone())
            {
                sessions.remove(&oldest_key);
            } else {
                break;
            }
        }

        sessions.insert(
            key,
            FusionSessionRecord {
                snapshot,
                last_accessed: now,
                turns_count: 0,
            },
        );
    }

    /// Record turn progress and update access timestamp
    pub async fn record_turn(&self, key: &FusionSessionKey) {
        let mut sessions = self.sessions.write().await;
        if let Some(record) = sessions.get_mut(key) {
            record.turns_count = record.turns_count.saturating_add(1);
            record.last_accessed = Instant::now();
        }
    }

    /// Invalidate a specific session (e.g. after 401 or explicit reset)
    pub async fn invalidate(&self, key: &FusionSessionKey) {
        let mut sessions = self.sessions.write().await;
        sessions.remove(key);
        let mut locks = self.single_flight.lock().await;
        locks.remove(key);
    }

    /// Invalidate all sessions associated with an account (e.g. logout / account switch)
    pub async fn invalidate_account(&self, account_id: &str) {
        let mut sessions = self.sessions.write().await;
        sessions.retain(|k, _| k.account_id != account_id);
        let mut locks = self.single_flight.lock().await;
        locks.retain(|k, _| k.account_id != account_id);
    }

    /// Clear all cached sessions
    pub async fn clear(&self) {
        let mut sessions = self.sessions.write().await;
        sessions.clear();
        let mut locks = self.single_flight.lock().await;
        locks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_key(session_id: &str, generation: u64) -> FusionSessionKey {
        FusionSessionKey {
            app_namespace: "codex".to_string(),
            provider_id: "prov-1".to_string(),
            account_id: "acc-1".to_string(),
            api_base: "https://api.individual.githubcopilot.com".to_string(),
            credential_generation: generation,
            client_session_id: session_id.to_string(),
            branch_id: None,
        }
    }

    fn dummy_snapshot(model: &str, generation: u64) -> FusionSessionSnapshot {
        FusionSessionSnapshot {
            selected_model: model.to_string(),
            session_token: "mock-session-token".to_string(),
            created_at: Instant::now(),
            expires_at: Some(Instant::now() + Duration::from_secs(3600)),
            multi_turn: None,
            multi_turn_mode: None,
            credential_generation: generation,
        }
    }

    #[tokio::test]
    async fn test_session_manager_insert_and_get() {
        let manager = FusionSessionManager::new();
        let key = dummy_key("session-1", 1);
        let snapshot = dummy_snapshot("gpt-5.6-luna", 1);

        manager.insert(key.clone(), snapshot.clone()).await;

        let retrieved = manager.get(&key).await.unwrap();
        assert_eq!(retrieved.selected_model, "gpt-5.6-luna");
        assert_eq!(retrieved.session_token, "mock-session-token");
    }

    #[tokio::test]
    async fn test_session_manager_account_isolation() {
        let manager = FusionSessionManager::new();
        let mut key1 = dummy_key("session-1", 1);
        key1.account_id = "user_a".to_string();

        let mut key2 = dummy_key("session-1", 1);
        key2.account_id = "user_b".to_string();

        manager
            .insert(key1.clone(), dummy_snapshot("model-a", 1))
            .await;
        manager
            .insert(key2.clone(), dummy_snapshot("model-b", 1))
            .await;

        assert_eq!(manager.get(&key1).await.unwrap().selected_model, "model-a");
        assert_eq!(manager.get(&key2).await.unwrap().selected_model, "model-b");

        // Invalidate user_a should not affect user_b
        manager.invalidate_account("user_a").await;
        assert!(manager.get(&key1).await.is_none());
        assert!(manager.get(&key2).await.is_some());
    }

    #[tokio::test]
    async fn test_session_manager_generation_mismatch() {
        let manager = FusionSessionManager::new();
        let key_v1 = dummy_key("session-1", 1);
        let key_v2 = dummy_key("session-1", 2);

        manager
            .insert(key_v1.clone(), dummy_snapshot("model-v1", 1))
            .await;

        assert!(manager.get(&key_v1).await.is_some());
        assert!(manager.get(&key_v2).await.is_none());
    }

    #[tokio::test]
    async fn test_session_manager_expired_token() {
        let manager = FusionSessionManager::new();
        let key = dummy_key("session-exp", 1);
        let mut snapshot = dummy_snapshot("model-1", 1);
        snapshot.expires_at = Some(Instant::now() - Duration::from_secs(1)); // Expired

        manager.insert(key.clone(), snapshot).await;
        assert!(manager.get(&key).await.is_none());
    }

    #[tokio::test]
    async fn test_session_manager_lru_capacity_eviction() {
        let manager = FusionSessionManager::new();

        // Fill up to MAX_CACHED_SESSIONS (256)
        for i in 0..MAX_CACHED_SESSIONS {
            let key = dummy_key(&format!("session-{i}"), 1);
            manager
                .insert(key, dummy_snapshot(&format!("model-{i}"), 1))
                .await;
        }

        assert_eq!(manager.sessions.read().await.len(), MAX_CACHED_SESSIONS);

        // Access session-0 to make it recently used
        let key0 = dummy_key("session-0", 1);
        let _ = manager.get(&key0).await;

        // Insert session-256 (257th entry)
        let key_new = dummy_key("session-256", 1);
        manager
            .insert(key_new.clone(), dummy_snapshot("model-256", 1))
            .await;

        // Total should stay capped at 256
        assert_eq!(manager.sessions.read().await.len(), MAX_CACHED_SESSIONS);
        // session-0 was accessed, so session-1 (oldest accessed) should have been evicted
        let key1 = dummy_key("session-1", 1);
        assert!(manager.get(&key1).await.is_none());
        assert!(manager.get(&key0).await.is_some());
        assert!(manager.get(&key_new).await.is_some());
    }

    #[tokio::test]
    async fn test_past_expiration_token_rejected_immediately() {
        let manager = FusionSessionManager::new();
        let key = dummy_key("session-past-exp", 1);

        // JWT with exp in 2020 (UNIX: 1577836800)
        let token = "eyJhbGciOiJFUzI1NiJ9.eyJleHAiOjE1Nzc4MzY4MDAsImF1dG9fc2Vzc2lvbl9pZCI6IjEyMyIsImF1dG9fdGllciI6ImJhbGFuY2UiLCJleGVjdXRpb25fbW9kZSI6ImZ1c2lvbiJ9.sig";
        let auto_resp = CopilotAutoResponse {
            session_token: token.to_string(),
            selected_model: crate::proxy::providers::copilot_fusion::SelectedModelPayload::String(
                "gpt-5.6-luna".to_string(),
            ),
            selection_reason: None,
            hydra_scores: None,
            multi_turn: None,
            multi_turn_mode: None,
            extra: Default::default(),
        };

        let snapshot = manager.create_snapshot_from_response(&key, auto_resp);
        assert!(!snapshot.is_valid());
    }
}
