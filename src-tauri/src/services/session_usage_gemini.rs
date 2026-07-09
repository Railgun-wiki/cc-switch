//! Gemini CLI 会话日志使用追踪
//!
//! 从 Gemini CLI JSON 会话和 Antigravity DB 中提取 token 使用数据。
//!
//! ## 数据流
//! ```text
//! ~/.gemini/tmp/*/chats/session-*.json
//! ~/.gemini/{antigravity,antigravity-cli,antigravity-ide}/conversations/*.db
//!   → 解析 → 费用计算 → proxy_request_logs 表
//! ```
//!
//! ## 与 Claude/Codex 解析器的差异
//! - JSON 格式（非 JSONL）：每个文件是单个 JSON 对象，包含 messages 数组
//! - 无需 delta 计算：tokens 字段是 per-message 独立值
//! - 无需状态恢复：不依赖前一条消息的累计值
//! - 天然去重：每条消息有唯一 id 字段

use crate::database::{lock_conn, Database};
use crate::error::AppError;
use crate::gemini_config::get_gemini_dir;
use crate::proxy::usage::calculator::{CostCalculator, ModelPricing};
use crate::proxy::usage::parser::TokenUsage;
use crate::services::session_usage::{
    get_sync_state, metadata_modified_nanos, update_sync_state, SessionSyncResult,
};
use crate::services::usage_stats::{
    find_model_pricing, is_placeholder_pricing_model, should_skip_session_insert, DedupKey,
};
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// 从 Gemini message 中提取的 token 数据
#[derive(Debug)]
struct GeminiTokens {
    input: u32,
    output: u32,
    cached: u32,
    thoughts: u32,
}

/// 同步 Gemini 使用数据（从 JSON 会话日志）
pub fn sync_gemini_usage(db: &Database) -> Result<SessionSyncResult, AppError> {
    let gemini_dir = get_gemini_dir();

    let files = collect_gemini_session_files(&gemini_dir);
    let antigravity_files = collect_antigravity_db_files(&gemini_dir);

    let mut result = SessionSyncResult {
        imported: 0,
        skipped: 0,
        files_scanned: (files.len() + antigravity_files.len()) as u32,
        errors: vec![],
    };

    for file_path in &files {
        match sync_single_gemini_file(db, file_path) {
            Ok((imported, skipped)) => {
                result.imported += imported;
                result.skipped += skipped;
            }
            Err(e) => {
                let msg = format!("Gemini 会话文件解析失败 {}: {e}", file_path.display());
                log::warn!("[GEMINI-SYNC] {msg}");
                result.errors.push(msg);
            }
        }
    }

    for file_path in &antigravity_files {
        match sync_single_antigravity_db(db, file_path) {
            Ok((imported, skipped)) => {
                result.imported += imported;
                result.skipped += skipped;
            }
            Err(e) => {
                let msg = format!(
                    "Antigravity 会话数据库解析失败 {}: {e}",
                    file_path.display()
                );
                log::warn!("[GEMINI-SYNC] {msg}");
                result.errors.push(msg);
            }
        }
    }

    if result.imported > 0 {
        log::info!(
            "[GEMINI-SYNC] 同步完成: 导入 {} 条, 跳过 {} 条, 扫描 {} 个文件",
            result.imported,
            result.skipped,
            result.files_scanned
        );
    }

    Ok(result)
}

/// 收集所有 Gemini 会话 JSON 文件
fn collect_gemini_session_files(gemini_dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    let tmp_dir = gemini_dir.join("tmp");
    if !tmp_dir.is_dir() {
        return files;
    }

    // 遍历 tmp/<project_hash>/chats/session-*.json
    let project_dirs = match fs::read_dir(&tmp_dir) {
        Ok(entries) => entries,
        Err(_) => return files,
    };

    for entry in project_dirs.flatten() {
        let chats_dir = entry.path().join("chats");
        if !chats_dir.is_dir() {
            continue;
        }

        let chat_files = match fs::read_dir(&chats_dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        for file_entry in chat_files.flatten() {
            let path = file_entry.path();
            let is_session = path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("session-") && n.ends_with(".json"))
                .unwrap_or(false);
            if is_session {
                files.push(path);
            }
        }
    }

    files
}

/// 同步单个 Gemini 会话 JSON 文件，返回 (imported, skipped)
fn sync_single_gemini_file(db: &Database, file_path: &Path) -> Result<(u32, u32), AppError> {
    let file_path_str = file_path.to_string_lossy().to_string();

    // 获取文件元数据
    let metadata = fs::metadata(file_path)
        .map_err(|e| AppError::Config(format!("无法读取文件元数据: {e}")))?;
    let file_modified = metadata_modified_nanos(&metadata);

    // 检查同步状态
    let (last_modified, _last_offset) = get_sync_state(db, &file_path_str)?;

    // 文件未变化则跳过
    if file_modified <= last_modified {
        return Ok((0, 0));
    }

    // 读取并解析整个 JSON 文件
    let content = fs::read_to_string(file_path)
        .map_err(|e| AppError::Config(format!("无法读取文件: {e}")))?;
    let value: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| AppError::Config(format!("JSON 解析失败: {e}")))?;

    // 提取顶层 sessionId
    let session_id = value
        .get("sessionId")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // 遍历 messages 数组
    let messages = match value.get("messages").and_then(|v| v.as_array()) {
        Some(msgs) => msgs,
        None => return Ok((0, 0)),
    };

    let mut imported: u32 = 0;
    let mut skipped: u32 = 0;
    let mut gemini_msg_count: i64 = 0;

    for msg in messages {
        // 只处理 type == "gemini" 的消息
        if msg.get("type").and_then(|t| t.as_str()) != Some("gemini") {
            continue;
        }

        // 提取 tokens 对象
        let tokens_obj = match msg.get("tokens") {
            Some(t) if t.is_object() => t,
            _ => continue,
        };

        let tokens = parse_gemini_tokens(tokens_obj);
        if tokens.input == 0 && tokens.output == 0 && tokens.thoughts == 0 && tokens.cached == 0 {
            continue; // 跳过全零的空 token 消息
        }

        gemini_msg_count += 1;

        // 提取消息 ID 和模型
        let message_id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
        let model = msg
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let timestamp = msg.get("timestamp").and_then(|v| v.as_str());

        // 生成唯一 request_id
        let session_id_str = session_id.as_deref().unwrap_or("unknown");
        let request_id = format!("gemini_session:{session_id_str}:{message_id}");

        match insert_gemini_session_entry(
            db,
            &request_id,
            &tokens,
            model,
            session_id.as_deref(),
            timestamp,
        ) {
            Ok(true) => imported += 1,
            Ok(false) => skipped += 1,
            Err(e) => {
                log::warn!("[GEMINI-SYNC] 插入失败 ({}): {e}", request_id);
                skipped += 1;
            }
        }
    }

    // 更新同步状态
    update_sync_state(db, &file_path_str, file_modified, gemini_msg_count)?;

    Ok((imported, skipped))
}

/// 从 tokens JSON 对象中提取 token 数据
fn parse_gemini_tokens(tokens: &serde_json::Value) -> GeminiTokens {
    GeminiTokens {
        input: tokens.get("input").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        output: tokens.get("output").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        cached: tokens.get("cached").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        thoughts: tokens.get("thoughts").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
    }
}

/// 插入单条 Gemini 会话记录到 proxy_request_logs
fn insert_gemini_session_entry(
    db: &Database,
    request_id: &str,
    tokens: &GeminiTokens,
    model: &str,
    session_id: Option<&str>,
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

    // 合并 thoughts 到 output（思考 token 按输出计费）
    let output_tokens = tokens.output + tokens.thoughts;

    let dedup_key = DedupKey {
        app_type: "gemini",
        model,
        input_tokens: tokens.input,
        output_tokens,
        cache_read_tokens: tokens.cached,
        cache_creation_tokens: 0,
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
        cache_creation_tokens: 0,
        model: Some(model.to_string()),
        message_id: None,
    };

    let pricing = find_gemini_pricing(&conn, model);
    let multiplier = Decimal::from(1);
    let (input_cost, output_cost, cache_read_cost, cache_creation_cost, total_cost) = match pricing
    {
        Some(p) => {
            let cost = CostCalculator::calculate_for_app("gemini", &usage, &p, multiplier);
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

    // 使用 UPSERT：新记录插入，已存在记录更新 token 和费用（Gemini 全量重读可能携带更新值）
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
            input_cost_usd = excluded.input_cost_usd,
            output_cost_usd = excluded.output_cost_usd,
            cache_read_cost_usd = excluded.cache_read_cost_usd,
            cache_creation_cost_usd = excluded.cache_creation_cost_usd,
            total_cost_usd = excluded.total_cost_usd
        WHERE input_tokens != excluded.input_tokens
           OR output_tokens != excluded.output_tokens
           OR cache_read_tokens != excluded.cache_read_tokens
           OR model != excluded.model",
        rusqlite::params![
            request_id,
            "_gemini_session",   // provider_id
            "gemini",            // app_type
            model,
            model,               // request_model = model
            tokens.input,
            output_tokens,
            tokens.cached,
            0i64,                // cache_creation_tokens
            input_cost,
            output_cost,
            cache_read_cost,
            cache_creation_cost,
            total_cost,
            0i64,                // latency_ms
            Option::<i64>::None, // first_token_ms
            200i64,              // status_code
            Option::<String>::None, // error_message
            session_id.map(|s| s.to_string()),
            Some("gemini_session"), // provider_type
            1i64,                // is_streaming
            "1.0",               // cost_multiplier
            created_at,
            "gemini_session",    // data_source
        ],
    )
    .map_err(|e| AppError::Database(format!("插入 Gemini 会话日志失败: {e}")))?;

    // changes() > 0 表示新插入或已更新，== 0 表示值完全相同（无实际变更）
    let changed = conn.changes() > 0;
    if changed {
        crate::usage_events::notify_log_recorded();
    }
    Ok(changed)
}

/// 查找 Gemini 模型定价
fn find_gemini_pricing(conn: &rusqlite::Connection, model_id: &str) -> Option<ModelPricing> {
    find_model_pricing(conn, model_id)
}

const ANTIGRAVITY_ROOTS: [&str; 3] = ["antigravity", "antigravity-cli", "antigravity-ide"];

#[derive(Debug)]
enum ProtoValue {
    Varint(u64),
    String(String),
    Nested(Vec<u8>),
}

struct ProtoParser<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ProtoParser<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn decode_varint(&mut self) -> Option<u64> {
        let mut result = 0u64;
        let mut shift = 0u32;
        while self.offset < self.data.len() {
            let byte = self.data[self.offset];
            self.offset += 1;
            result |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Some(result);
            }
            shift += 7;
            if shift >= 64 {
                return None;
            }
        }
        None
    }

    fn next_field(&mut self) -> Option<(u32, ProtoValue)> {
        if self.offset >= self.data.len() {
            return None;
        }

        let tag = self.decode_varint()?;
        let field_num = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u32;

        match wire_type {
            0 => self
                .decode_varint()
                .map(|value| (field_num, ProtoValue::Varint(value))),
            1 => {
                if self.offset + 8 > self.data.len() {
                    return None;
                }
                self.offset += 8;
                self.next_field()
            }
            2 => {
                let length = self.decode_varint()? as usize;
                if self.offset + length > self.data.len() {
                    return None;
                }
                let blob = &self.data[self.offset..self.offset + length];
                self.offset += length;
                if let Ok(text) = std::str::from_utf8(blob) {
                    if text
                        .chars()
                        .all(|c| c.is_ascii_graphic() || c.is_ascii_whitespace() || c == '\0')
                    {
                        return Some((field_num, ProtoValue::String(text.to_string())));
                    }
                }
                Some((field_num, ProtoValue::Nested(blob.to_vec())))
            }
            5 => {
                if self.offset + 4 > self.data.len() {
                    return None;
                }
                self.offset += 4;
                self.next_field()
            }
            _ => None,
        }
    }

    fn get_varint(&mut self, target_field: u32) -> Option<u64> {
        while let Some((field, value)) = self.next_field() {
            if field == target_field {
                if let ProtoValue::Varint(value) = value {
                    return Some(value);
                }
            }
        }
        None
    }

    fn get_nested(&mut self, target_field: u32) -> Option<Vec<u8>> {
        while let Some((field, value)) = self.next_field() {
            if field == target_field {
                if let ProtoValue::Nested(value) = value {
                    return Some(value);
                }
            }
        }
        None
    }

    fn get_string(&mut self, target_field: u32) -> Option<String> {
        while let Some((field, value)) = self.next_field() {
            if field == target_field {
                if let ProtoValue::String(value) = value {
                    return Some(value);
                }
            }
        }
        None
    }
}

#[derive(Debug, Default)]
struct AntigravityTokenData {
    input_tokens: u32,
    output_tokens: u32,
    cached_tokens: u32,
    thoughts_tokens: u32,
    model: String,
}

impl AntigravityTokenData {
    fn has_tokens(&self) -> bool {
        self.input_tokens != 0
            || self.output_tokens != 0
            || self.cached_tokens != 0
            || self.thoughts_tokens != 0
    }
}

#[derive(Debug, Default)]
struct TrajectoryMetadata {
    session_id: Option<String>,
    created_at_seconds: Option<i64>,
}

#[derive(Debug)]
struct GenMetadataEntry {
    idx: i64,
    token_data: Option<AntigravityTokenData>,
}

fn collect_antigravity_db_files(gemini_dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for root in ANTIGRAVITY_ROOTS {
        let conversations_dir = gemini_dir.join(root).join("conversations");
        let entries = match fs::read_dir(&conversations_dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("db") {
                files.push(path);
            }
        }
    }
    files
}

fn sync_single_antigravity_db(db: &Database, db_path: &Path) -> Result<(u32, u32), AppError> {
    let file_path_str = db_path.to_string_lossy().to_string();
    let metadata =
        fs::metadata(db_path).map_err(|e| AppError::Config(format!("无法读取文件元数据: {e}")))?;
    let file_modified = metadata_modified_nanos(&metadata);
    let file_modified_secs = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);

    let (last_modified, last_gen_idx) = get_sync_state(db, &file_path_str)?;
    if file_modified <= last_modified {
        return Ok((0, 0));
    }

    let agy_conn = rusqlite::Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| AppError::Config(format!("无法只读打开 Antigravity DB: {e}")))?;

    let trajectory_meta = read_trajectory_metadata(&agy_conn);
    let gen_entries = read_gen_metadata_entries(&agy_conn);
    if gen_entries.is_empty() {
        update_sync_state(db, &file_path_str, file_modified, 0)?;
        return Ok((0, 0));
    }

    let step_timestamps = read_step_timestamps(&agy_conn);
    let session_id = trajectory_meta
        .as_ref()
        .and_then(|meta| meta.session_id.clone())
        .or_else(|| {
            db_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(|stem| stem.to_string())
        });
    let fallback_created_at = trajectory_meta
        .as_ref()
        .and_then(|meta| meta.created_at_seconds)
        .unwrap_or(file_modified_secs);

    let mut imported = 0u32;
    let mut skipped = 0u32;
    let max_idx = gen_entries.last().map(|entry| entry.idx + 1).unwrap_or(0);

    for entry in &gen_entries {
        if entry.idx < last_gen_idx {
            continue;
        }
        let Some(token_data) = &entry.token_data else {
            continue;
        };

        let session_id_str = session_id.as_deref().unwrap_or("unknown");
        let request_id = format!("gemini_antigravity_session:{session_id_str}:{}", entry.idx);
        let created_at = step_timestamps
            .get(&entry.idx)
            .copied()
            .unwrap_or(fallback_created_at);

        match insert_antigravity_session_entry(
            db,
            &request_id,
            token_data,
            session_id.as_deref(),
            created_at,
        ) {
            Ok(true) => imported += 1,
            Ok(false) => skipped += 1,
            Err(e) => {
                log::warn!("[GEMINI-SYNC] 插入 Antigravity 会话失败 ({request_id}): {e}");
                skipped += 1;
            }
        }
    }

    update_sync_state(db, &file_path_str, file_modified, max_idx)?;
    Ok((imported, skipped))
}

fn read_trajectory_metadata(conn: &rusqlite::Connection) -> Option<TrajectoryMetadata> {
    let mut stmt = conn
        .prepare("SELECT data FROM trajectory_metadata_blob WHERE id = 'main'")
        .ok()?;
    let data: Vec<u8> = stmt.query_row([], |row| row.get(0)).ok()?;
    let mut parser = ProtoParser::new(&data);
    let mut meta = TrajectoryMetadata::default();

    while let Some((field, value)) = parser.next_field() {
        match (field, value) {
            (2, ProtoValue::Nested(nested)) => {
                let mut timestamp = ProtoParser::new(&nested);
                meta.created_at_seconds = timestamp.get_varint(1).map(|value| value as i64);
            }
            (3, ProtoValue::String(session_id)) => {
                meta.session_id = Some(session_id);
            }
            _ => {}
        }
    }

    Some(meta)
}

fn read_gen_metadata_entries(conn: &rusqlite::Connection) -> Vec<GenMetadataEntry> {
    let mut entries = Vec::new();
    let mut stmt = match conn.prepare("SELECT idx, data FROM gen_metadata ORDER BY idx") {
        Ok(stmt) => stmt,
        Err(_) => return entries,
    };
    let rows = stmt
        .query_map([], |row| {
            let idx: i64 = row.get(0)?;
            let data: Vec<u8> = row.get(1)?;
            Ok((idx, data))
        })
        .ok();

    if let Some(rows) = rows {
        for row in rows.flatten() {
            let (idx, data) = row;
            entries.push(GenMetadataEntry {
                idx,
                token_data: parse_gen_metadata_blob(&data),
            });
        }
    }

    entries
}

fn read_step_timestamps(conn: &rusqlite::Connection) -> HashMap<i64, i64> {
    let mut map = HashMap::new();
    let mut stmt = match conn.prepare("SELECT metadata FROM steps WHERE metadata IS NOT NULL") {
        Ok(stmt) => stmt,
        Err(_) => return map,
    };
    let rows = stmt.query_map([], |row| row.get::<_, Vec<u8>>(0)).ok();
    if let Some(rows) = rows {
        for row in rows.flatten() {
            if let Some((gen_idx, timestamp)) = parse_step_timestamp(&row) {
                map.entry(gen_idx)
                    .and_modify(|existing| {
                        if timestamp < *existing {
                            *existing = timestamp;
                        }
                    })
                    .or_insert(timestamp);
            }
        }
    }
    map
}

fn parse_step_timestamp(data: &[u8]) -> Option<(i64, i64)> {
    let mut parser = ProtoParser::new(data);
    let mut timestamp: Option<i64> = None;
    let mut gen_idx: Option<i64> = None;

    while let Some((field, value)) = parser.next_field() {
        match (field, value) {
            (1, ProtoValue::Nested(nested)) => {
                let mut ts = ProtoParser::new(&nested);
                timestamp = ts.get_varint(1).map(|value| value as i64);
            }
            (20, ProtoValue::Nested(nested)) => {
                let mut f20 = ProtoParser::new(&nested);
                gen_idx = f20.get_varint(3).map(|value| value as i64);
            }
            _ => {}
        }
    }

    match (gen_idx, timestamp) {
        (Some(gen_idx), Some(timestamp)) => Some((gen_idx, timestamp)),
        _ => None,
    }
}

fn parse_gen_metadata_blob(data: &[u8]) -> Option<AntigravityTokenData> {
    let mut parser = ProtoParser::new(data);
    let f1_blob = parser.get_nested(1)?;
    let mut f1 = ProtoParser::new(&f1_blob);
    let mut step_tokens = AntigravityTokenData::default();
    let mut cumulative_tokens = AntigravityTokenData::default();
    let mut model = String::new();

    while let Some((field, value)) = f1.next_field() {
        match (field, value) {
            (4, ProtoValue::Nested(nested)) => {
                extract_token_fields(&nested, &mut step_tokens);
            }
            (17, ProtoValue::Nested(nested)) => {
                let mut f17 = ProtoParser::new(&nested);
                if let Some(f2_blob) = f17.get_nested(2) {
                    extract_token_fields(&f2_blob, &mut cumulative_tokens);
                }
            }
            (19, ProtoValue::String(value)) => {
                model = value;
            }
            (20, ProtoValue::Nested(nested)) if model.is_empty() => {
                let mut tag_parser = ProtoParser::new(&nested);
                let tag_key = tag_parser.get_string(1);
                let tag_value = tag_parser.get_string(2);
                if tag_key.as_deref() == Some("model_enum") {
                    if let Some(value) = tag_value {
                        model = value;
                    }
                }
            }
            _ => {}
        }
    }

    let mut token_data = if step_tokens.has_tokens() {
        step_tokens
    } else {
        cumulative_tokens
    };
    if !token_data.has_tokens() {
        return None;
    }
    if model.trim().is_empty() {
        model = "unknown".to_string();
    }
    token_data.model = model;
    Some(token_data)
}

fn extract_token_fields(data: &[u8], tokens: &mut AntigravityTokenData) {
    let mut parser = ProtoParser::new(data);
    while let Some((field, value)) = parser.next_field() {
        let ProtoValue::Varint(value) = value else {
            continue;
        };
        match field {
            2 => tokens.input_tokens = value as u32,
            3 => tokens.output_tokens = value as u32,
            9 => tokens.cached_tokens = value as u32,
            10 => tokens.thoughts_tokens = value as u32,
            _ => {}
        }
    }
}

fn normalize_antigravity_pricing_model(raw_model: &str) -> String {
    let normalized = raw_model.trim().to_ascii_lowercase();
    if normalized == "gemini-pro-default" {
        return "gemini-3-pro-preview".to_string();
    }
    if is_placeholder_pricing_model(&normalized) || normalized.starts_with("model_placeholder_") {
        return raw_model.to_string();
    }

    let without_thinking = normalized
        .strip_suffix("-thinking")
        .unwrap_or(&normalized)
        .to_string();
    if let Some(base) = without_thinking.strip_suffix("-a") {
        return format!("{base}-preview");
    }
    if let Some(base) = without_thinking.strip_suffix("-b") {
        return format!("{base}-preview");
    }
    without_thinking
}

fn insert_antigravity_session_entry(
    db: &Database,
    request_id: &str,
    token_data: &AntigravityTokenData,
    session_id: Option<&str>,
    created_at: i64,
) -> Result<bool, AppError> {
    let conn = lock_conn!(db.conn);
    let output_tokens = token_data.output_tokens.max(token_data.thoughts_tokens);
    let raw_model = token_data.model.trim();
    let model = if raw_model.is_empty() {
        "unknown"
    } else {
        raw_model
    };
    let pricing_model = normalize_antigravity_pricing_model(model);

    let dedup_key = DedupKey {
        app_type: "gemini",
        model: &pricing_model,
        input_tokens: token_data.input_tokens,
        output_tokens,
        cache_read_tokens: token_data.cached_tokens,
        cache_creation_tokens: 0,
        created_at,
    };
    if should_skip_session_insert(&conn, request_id, &dedup_key)? {
        return Ok(false);
    }

    let usage = TokenUsage {
        input_tokens: token_data.input_tokens,
        output_tokens,
        cache_read_tokens: token_data.cached_tokens,
        cache_creation_tokens: 0,
        model: Some(pricing_model.clone()),
        message_id: None,
    };

    let pricing = find_gemini_pricing(&conn, &pricing_model);
    if pricing.is_none() && !is_placeholder_pricing_model(&pricing_model) {
        log::warn!("[GEMINI-SYNC] Antigravity 模型未命中定价: {model} -> {pricing_model}");
    }

    let multiplier = Decimal::from(1);
    let (input_cost, output_cost, cache_read_cost, cache_creation_cost, total_cost) = match pricing
    {
        Some(pricing) => {
            let cost = CostCalculator::calculate_for_app("gemini", &usage, &pricing, multiplier);
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
            provider_type, is_streaming, cost_multiplier, created_at, data_source, pricing_model
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25)
        ON CONFLICT(request_id) DO UPDATE SET
            model = excluded.model,
            request_model = excluded.request_model,
            pricing_model = excluded.pricing_model,
            input_tokens = excluded.input_tokens,
            output_tokens = excluded.output_tokens,
            cache_read_tokens = excluded.cache_read_tokens,
            input_cost_usd = excluded.input_cost_usd,
            output_cost_usd = excluded.output_cost_usd,
            cache_read_cost_usd = excluded.cache_read_cost_usd,
            cache_creation_cost_usd = excluded.cache_creation_cost_usd,
            total_cost_usd = excluded.total_cost_usd,
            created_at = excluded.created_at
        WHERE input_tokens != excluded.input_tokens
           OR output_tokens != excluded.output_tokens
           OR cache_read_tokens != excluded.cache_read_tokens
           OR model != excluded.model
           OR COALESCE(pricing_model, '') != excluded.pricing_model
           OR created_at != excluded.created_at",
        rusqlite::params![
            request_id,
            "_gemini_antigravity_session",
            "gemini",
            model,
            model,
            token_data.input_tokens,
            output_tokens,
            token_data.cached_tokens,
            0i64,
            input_cost,
            output_cost,
            cache_read_cost,
            cache_creation_cost,
            total_cost,
            0i64,
            Option::<i64>::None,
            200i64,
            Option::<String>::None,
            session_id.map(|value| value.to_string()),
            Some("antigravity_session"),
            1i64,
            "1.0",
            created_at,
            "antigravity_session",
            pricing_model,
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
    fn test_collect_gemini_session_files_nonexistent() {
        let files = collect_gemini_session_files(Path::new("/nonexistent/path"));
        assert!(files.is_empty());
    }

    #[test]
    fn test_insert_gemini_session_skips_matching_proxy_log() -> Result<(), AppError> {
        let db = Database::memory()?;
        {
            let conn = lock_conn!(db.conn);
            conn.execute(
                "INSERT INTO proxy_request_logs (
                    request_id, provider_id, app_type, model, request_model,
                    input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens,
                    total_cost_usd, latency_ms, status_code, created_at, data_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rusqlite::params![
                    "gemini-proxy",
                    "google",
                    "gemini",
                    "gemini-2.5-pro",
                    "gemini-2.5-pro",
                    10,
                    7,
                    1,
                    0,
                    "0.01",
                    100,
                    200,
                    1000,
                    "proxy"
                ],
            )?;
        }

        let tokens = GeminiTokens {
            input: 10,
            output: 2,
            cached: 1,
            thoughts: 5,
        };
        let inserted = insert_gemini_session_entry(
            &db,
            "gemini-session-dup",
            &tokens,
            "gemini-2.5-pro",
            Some("session-1"),
            Some("1970-01-01T00:16:45Z"),
        )?;
        assert!(!inserted);

        let conn = lock_conn!(db.conn);
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM proxy_request_logs", [], |row| {
            row.get(0)
        })?;
        assert_eq!(count, 1);

        Ok(())
    }

    #[test]
    fn test_parse_gemini_tokens() {
        let json: serde_json::Value = serde_json::json!({
            "input": 8522,
            "output": 29,
            "cached": 3138,
            "thoughts": 405,
            "tool": 0,
            "total": 8956
        });
        let tokens = parse_gemini_tokens(&json);
        assert_eq!(tokens.input, 8522);
        assert_eq!(tokens.output, 29);
        assert_eq!(tokens.cached, 3138);
        assert_eq!(tokens.thoughts, 405);
        // output + thoughts = 29 + 405 = 434（用于计费）
        assert_eq!(tokens.output + tokens.thoughts, 434);
    }

    #[test]
    fn test_parse_gemini_tokens_missing_fields() {
        // 缺少某些字段时应返回 0
        let json: serde_json::Value = serde_json::json!({
            "input": 100,
            "output": 50
        });
        let tokens = parse_gemini_tokens(&json);
        assert_eq!(tokens.input, 100);
        assert_eq!(tokens.output, 50);
        assert_eq!(tokens.cached, 0);
        assert_eq!(tokens.thoughts, 0);
    }

    #[test]
    fn test_parse_gemini_tokens_all_zero() {
        let json: serde_json::Value = serde_json::json!({
            "input": 0,
            "output": 0,
            "cached": 0,
            "thoughts": 0,
            "tool": 0,
            "total": 0
        });
        let tokens = parse_gemini_tokens(&json);
        assert_eq!(tokens.input, 0);
        assert_eq!(tokens.output, 0);
        // 全零（包括 cached=0）会被 sync 逻辑跳过
        assert!(
            tokens.input == 0 && tokens.output == 0 && tokens.thoughts == 0 && tokens.cached == 0
        );
    }

    #[test]
    fn test_parse_gemini_tokens_cache_only_not_skipped() {
        // 纯缓存命中消息（input/output/thoughts=0 但 cached>0）不应被跳过
        let json: serde_json::Value = serde_json::json!({
            "input": 0,
            "output": 0,
            "cached": 5000,
            "thoughts": 0
        });
        let tokens = parse_gemini_tokens(&json);
        assert_eq!(tokens.cached, 5000);
        // 跳过条件：所有四个字段都为 0 才跳过
        let should_skip =
            tokens.input == 0 && tokens.output == 0 && tokens.thoughts == 0 && tokens.cached == 0;
        assert!(!should_skip, "纯缓存命中记录不应被跳过");
    }

    #[test]
    fn test_normalize_antigravity_pricing_model_aliases() {
        assert_eq!(
            normalize_antigravity_pricing_model("gemini-3-flash-a-thinking"),
            "gemini-3-flash-preview"
        );
        assert_eq!(
            normalize_antigravity_pricing_model("gemini-3-pro-b"),
            "gemini-3-pro-preview"
        );
        assert_eq!(
            normalize_antigravity_pricing_model("gemini-pro-default"),
            "gemini-3-pro-preview"
        );
        assert_eq!(
            normalize_antigravity_pricing_model("MODEL_PLACEHOLDER_M132"),
            "MODEL_PLACEHOLDER_M132"
        );
    }
}
