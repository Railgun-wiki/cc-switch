//! Antigravity 2.0 会话日志使用追踪
//!
//! 从脑库目录的 transcript.jsonl 文件中提取 token 使用数据。
//!
//! ## 数据流
//! ```text
//! ~/.gemini/antigravity*/brain/<session_id>/.system_generated/logs/transcript.jsonl → 增量解析 → 去重 → 费用计算 → proxy_request_logs 表
//! ```

use crate::database::{lock_conn, Database};
use crate::error::AppError;
use crate::proxy::usage::calculator::{CostCalculator, ModelPricing};
use crate::proxy::usage::parser::TokenUsage;
use crate::services::session_usage::{
    get_sync_state, metadata_modified_nanos, update_sync_state, SessionSyncResult,
};
use crate::services::usage_stats::{find_model_pricing, should_skip_session_insert, DedupKey};
use rust_decimal::Decimal;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// 从 Antigravity transcript step 中提取 of token 数据
#[derive(Debug)]
struct AntigravityTokens {
    input: u32,
    output: u32,
    cached: u32,
    thoughts: u32,
    cache_creation: u32,
}

fn get_model_from_env(dir: &Path) -> Option<String> {
    let env_path = dir.join(".env");
    if !env_path.is_file() {
        return None;
    }
    let file = fs::File::open(env_path).ok()?;
    let reader = BufReader::new(file);
    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => continue,
        };
        let trimmed = line.trim();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        if let Some((key, val)) = trimmed.split_once('=') {
            let key = key.trim();
            if key == "GEMINI_MODEL"
                || key == "CLAUDE_MODEL"
                || key == "MODEL"
                || key == "API_MODEL"
            {
                let val = val.trim().trim_matches(|c| c == '"' || c == '\'');
                if !val.is_empty() {
                    return Some(val.to_string());
                }
            }
        }
    }
    None
}

fn get_model_from_json(path: &Path) -> Option<String> {
    if !path.is_file() {
        return None;
    }
    let file = fs::File::open(path).ok()?;
    let val: serde_json::Value = serde_json::from_reader(file).ok()?;
    val.get("model")
        .or_else(|| val.pointer("/env/GEMINI_MODEL"))
        .or_else(|| val.pointer("/env/CLAUDE_MODEL"))
        .or_else(|| val.pointer("/env/MODEL"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn get_model_from_env_or_config(session_dir: &Path) -> Option<String> {
    if let Some(m) = get_model_from_env(session_dir) {
        return Some(m);
    }
    if let Some(m) = get_model_from_json(&session_dir.join("settings.json")) {
        return Some(m);
    }

    if let Some(root) = session_dir.parent().and_then(|p| p.parent()) {
        if let Some(m) = get_model_from_env(root) {
            return Some(m);
        }
        if let Some(m) = get_model_from_json(&root.join("settings.json")) {
            return Some(m);
        }
    }

    let global_dir = crate::antigravity_config::get_antigravity_dir();
    if let Some(m) = get_model_from_json(&global_dir.join("settings.json")) {
        return Some(m);
    }

    None
}

/// 同步 Antigravity 使用数据（从 transcript.jsonl）
pub fn sync_antigravity_usage(db: &Database) -> Result<SessionSyncResult, AppError> {
    let files = collect_antigravity_session_files();

    let mut result = SessionSyncResult {
        imported: 0,
        skipped: 0,
        files_scanned: files.len() as u32,
        errors: vec![],
    };

    if files.is_empty() {
        return Ok(result);
    }

    for (file_path, session_id) in &files {
        match sync_single_antigravity_file(db, file_path, session_id) {
            Ok((imported, skipped)) => {
                result.imported += imported;
                result.skipped += skipped;
            }
            Err(e) => {
                let msg = format!("Antigravity 会话文件解析失败 {}: {e}", file_path.display());
                log::warn!("[ANTIGRAVITY-SYNC] {msg}");
                result.errors.push(msg);
            }
        }
    }

    if result.imported > 0 {
        log::info!(
            "[ANTIGRAVITY-SYNC] 同步完成: 导入 {} 条, 跳过 {} 条, 扫描 {} 个文件",
            result.imported,
            result.skipped,
            result.files_scanned
        );
    }

    Ok(result)
}

/// 收集所有 Antigravity 会话的 transcript.jsonl 文件
fn collect_antigravity_session_files() -> Vec<(PathBuf, String)> {
    let mut files = Vec::new();
    for root in crate::session_manager::providers::antigravity::session_roots() {
        let brain = root.join("brain");
        let Ok(entries) = fs::read_dir(brain) else {
            continue;
        };

        for entry in entries.flatten() {
            let session_dir = entry.path();
            if !session_dir.is_dir() {
                continue;
            }
            let Some(session_id) = session_dir.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let transcript = session_dir
                .join(".system_generated")
                .join("logs")
                .join("transcript.jsonl");
            if transcript.is_file() {
                files.push((transcript, session_id.to_string()));
            }
        }
    }
    files
}

/// 同步单个 transcript.jsonl，返回 (imported, skipped)
fn sync_single_antigravity_file(
    db: &Database,
    file_path: &Path,
    session_id: &str,
) -> Result<(u32, u32), AppError> {
    let file_path_str = file_path.to_string_lossy().to_string();

    // 获取文件元数据
    let metadata = fs::metadata(file_path)
        .map_err(|e| AppError::Config(format!("无法读取文件元数据: {e}")))?;
    let file_modified = metadata_modified_nanos(&metadata);

    // 检查同步状态
    let (last_modified, last_offset) = get_sync_state(db, &file_path_str)?;

    // 文件未变化则跳过
    if file_modified <= last_modified {
        return Ok((0, 0));
    }

    let file =
        fs::File::open(file_path).map_err(|e| AppError::Config(format!("无法打开文件: {e}")))?;
    let reader = BufReader::new(file);

    let session_dir = file_path
        .parent() // logs
        .and_then(|p| p.parent()) // .system_generated
        .and_then(|p| p.parent()); // <session_id>

    let mut current_model = session_dir.and_then(get_model_from_env_or_config);

    let mut line_offset: i64 = 0;
    let mut imported: u32 = 0;
    let mut skipped: u32 = 0;

    for line_result in reader.lines() {
        line_offset += 1;

        let line = match line_result {
            Ok(l) => l,
            Err(_) => continue, // 容忍不完整的最后一行
        };

        if line.trim().is_empty() {
            continue;
        }

        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if let Some(m) = value.get("model").and_then(|v| v.as_str()) {
            current_model = Some(m.to_string());
        }

        // 跳过已处理的行
        if line_offset <= last_offset {
            continue;
        }

        let source = value.get("source").and_then(|v| v.as_str()).unwrap_or("");
        let event_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");

        // 只处理 MODEL 产生的 PLANNER_RESPONSE / GENERIC
        if source != "MODEL" || (event_type != "PLANNER_RESPONSE" && event_type != "GENERIC") {
            continue;
        }

        let tokens_val = match value.get("tokens").or_else(|| value.get("usage")) {
            Some(t) if t.is_object() => t,
            _ => continue,
        };

        let tokens = parse_antigravity_tokens(tokens_val);
        if tokens.input == 0
            && tokens.output == 0
            && tokens.thoughts == 0
            && tokens.cached == 0
            && tokens.cache_creation == 0
        {
            continue; // 跳过空 Token 记录
        }

        let model = current_model.as_deref().unwrap_or("gpt-oss-120b");
        let timestamp = value.get("created_at").and_then(|v| v.as_str());

        let step_index_str = match value.get("step_index").and_then(|v| v.as_i64()) {
            Some(idx) => idx.to_string(),
            None => {
                let content_str = value.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let hash_input = format!(
                    "ts:{:?}|in:{}|out:{}|cached:{}|thoughts:{}|cc:{}|content:{}",
                    timestamp,
                    tokens.input,
                    tokens.output,
                    tokens.cached,
                    tokens.thoughts,
                    tokens.cache_creation,
                    content_str
                );
                use sha2::{Digest, Sha256};
                let hash_val = Sha256::digest(hash_input.as_bytes());
                format!("hash_{:x}", hash_val)
            }
        };
        let request_id = format!("agy_session:{session_id}:{step_index_str}");

        match insert_antigravity_session_entry(
            db,
            &request_id,
            &tokens,
            model,
            session_id,
            timestamp,
        ) {
            Ok(true) => imported += 1,
            Ok(false) => skipped += 1,
            Err(e) => {
                log::warn!("[ANTIGRAVITY-SYNC] 插入失败 ({}): {e}", request_id);
                skipped += 1;
            }
        }
    }

    // 更新同步状态
    update_sync_state(db, &file_path_str, file_modified, line_offset)?;

    Ok((imported, skipped))
}

/// 解析 token 数据（兼容多种字段命名，包含 thoughts）
fn parse_antigravity_tokens(tokens: &serde_json::Value) -> AntigravityTokens {
    let input = tokens
        .get("input_tokens")
        .or_else(|| tokens.get("inputTokens"))
        .or_else(|| tokens.get("input"))
        .or_else(|| tokens.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let output = tokens
        .get("output_tokens")
        .or_else(|| tokens.get("outputTokens"))
        .or_else(|| tokens.get("output"))
        .or_else(|| tokens.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let cached = tokens
        .get("cache_read_tokens")
        .or_else(|| tokens.get("cacheReadTokens"))
        .or_else(|| tokens.get("cache_read_input_tokens"))
        .or_else(|| tokens.get("cached"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let thoughts = tokens
        .get("thinking_output_tokens")
        .or_else(|| tokens.get("thoughts_token_count"))
        .or_else(|| tokens.get("thoughtsTokenCount"))
        .or_else(|| tokens.get("thoughts"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let cache_creation = tokens
        .get("cache_creation_input_tokens")
        .or_else(|| tokens.get("cache_creation_tokens"))
        .or_else(|| tokens.get("cache_creation"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    AntigravityTokens {
        input,
        output,
        cached,
        thoughts,
        cache_creation,
    }
}

/// 插入单条 Antigravity 记录到 proxy_request_logs
fn insert_antigravity_session_entry(
    db: &Database,
    request_id: &str,
    tokens: &AntigravityTokens,
    model: &str,
    session_id: &str,
    timestamp: Option<&str>,
) -> Result<bool, AppError> {
    let conn = lock_conn!(db.conn);

    let created_at = timestamp
        .and_then(|ts| {
            chrono::DateTime::parse_from_rfc3339(ts)
                .ok()
                .map(|dt| dt.timestamp())
        })
        .unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0)
        });

    // 取 output 与 thoughts 中的最大值，避免重复计算 thoughts 并确保安全
    let output_tokens = std::cmp::max(tokens.output, tokens.thoughts);

    let dedup_key = DedupKey {
        app_type: "antigravity",
        model,
        input_tokens: tokens.input,
        output_tokens,
        cache_read_tokens: tokens.cached,
        cache_creation_tokens: tokens.cache_creation,
        created_at,
    };
    if should_skip_session_insert(&conn, request_id, &dedup_key)? {
        return Ok(false);
    }

    // 计算费用
    let usage = TokenUsage {
        input_tokens: tokens.input,
        output_tokens,
        cache_read_tokens: tokens.cached,
        cache_creation_tokens: tokens.cache_creation,
        model: Some(model.to_string()),
        message_id: None,
    };

    let pricing = find_model_pricing(&conn, model);
    let multiplier = Decimal::from(1);
    let (input_cost, output_cost, cache_read_cost, cache_creation_cost, total_cost) = match pricing
    {
        Some(p) => {
            let cost = CostCalculator::calculate_for_app("antigravity", &usage, &p, multiplier);
            (
                cost.input_cost.to_string(),
                cost.output_cost.to_string(),
                cost.cache_read_cost.to_string(),
                cost.cache_creation_cost.to_string(),
                cost.total_cost.to_string(),
            )
        }
        None => (
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
        ),
    };

    conn.execute(
        "INSERT INTO proxy_request_logs (
            request_id, provider_id, app_type, model, request_model,
            input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens,
            input_cost_usd, output_cost_usd, cache_read_cost_usd, cache_creation_cost_usd, total_cost_usd,
            latency_ms, first_token_ms, status_code, error_message, session_id,
            provider_type, is_streaming, cost_multiplier, created_at, data_source
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24)
        ON CONFLICT(request_id) DO UPDATE SET
            model = excluded.model,
            input_tokens = excluded.input_tokens,
            output_tokens = excluded.output_tokens,
            cache_read_tokens = excluded.cache_read_tokens,
            cache_creation_tokens = excluded.cache_creation_tokens,
            input_cost_usd = excluded.input_cost_usd,
            output_cost_usd = excluded.output_cost_usd,
            cache_read_cost_usd = excluded.cache_read_cost_usd,
            cache_creation_cost_usd = excluded.cache_creation_cost_usd,
            total_cost_usd = excluded.total_cost_usd
        WHERE input_tokens != excluded.input_tokens
           OR output_tokens != excluded.output_tokens
           OR cache_read_tokens != excluded.cache_read_tokens
           OR cache_creation_tokens != excluded.cache_creation_tokens
           OR model != excluded.model",
        rusqlite::params![
            request_id,
            "_antigravity_session",   // provider_id
            "antigravity",            // app_type
            model,
            model,               // request_model = model
            tokens.input,
            output_tokens,
            tokens.cached,
            tokens.cache_creation, // cache_creation_tokens
            input_cost,
            output_cost,
            cache_read_cost,
            cache_creation_cost,
            total_cost,
            0i64,                // latency_ms
            Option::<i64>::None, // first_token_ms
            200i64,              // status_code
            Option::<String>::None, // error_message
            Some(session_id.to_string()),
            Some("antigravity_session"), // provider_type
            1i64,                // is_streaming
            "1.0",               // cost_multiplier
            created_at,
            "antigravity_session",    // data_source
        ],
    )
    .map_err(|e| AppError::Database(format!("插入 Antigravity 会话日志失败: {e}")))?;

    let changed = conn.changes() > 0;
    if changed {
        crate::usage_events::notify_log_recorded();
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_antigravity_tokens() {
        let json = serde_json::json!({
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 20,
            "thoughts": 10,
            "cache_creation_input_tokens": 15
        });
        let tokens = parse_antigravity_tokens(&json);
        assert_eq!(tokens.input, 100);
        assert_eq!(tokens.output, 50);
        assert_eq!(tokens.cached, 20);
        assert_eq!(tokens.thoughts, 10);
        assert_eq!(tokens.cache_creation, 15);

        let json_short = serde_json::json!({
            "input": 200,
            "output": 80,
            "cached": 30
        });
        let tokens_short = parse_antigravity_tokens(&json_short);
        assert_eq!(tokens_short.input, 200);
        assert_eq!(tokens_short.output, 80);
        assert_eq!(tokens_short.cached, 30);
        assert_eq!(tokens_short.thoughts, 0);
        assert_eq!(tokens_short.cache_creation, 0);

        let json_fallback = serde_json::json!({
            "prompt_tokens": 120,
            "completion_tokens": 60,
            "cache_read_tokens": 25,
            "thinking_output_tokens": 15,
            "cache_creation_tokens": 5
        });
        let tokens_fallback = parse_antigravity_tokens(&json_fallback);
        assert_eq!(tokens_fallback.input, 120);
        assert_eq!(tokens_fallback.output, 60);
        assert_eq!(tokens_fallback.cached, 25);
        assert_eq!(tokens_fallback.thoughts, 15);
        assert_eq!(tokens_fallback.cache_creation, 5);
    }
}
