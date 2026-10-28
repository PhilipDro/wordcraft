//! The two wire formats: building a request from the conversation and reading a reply.
//!
//! Replies are untrusted input: every field is optional, nothing is indexed.

use serde_json::{Value, json};

use crate::providers::{Endpoint, Protocol};

/// Longest reply we ask for. 4096 fits every listed provider (DeepSeek's chat models cap output
/// at 8192 and reject more).
const MAX_TOKENS: u32 = 4096;
/// OpenAI's own API counts reasoning tokens in the limit, so it gets more room.
const MAX_TOKENS_OPENAI: u32 = 16_384;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
}

/// One HTTP request, sent by the host (the assistant crate does no I/O). A `Get` has no body.
#[derive(Clone, PartialEq)]
pub struct HttpRequest {
    pub method: Method,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolOutput {
    pub id: String,
    pub text: String,
    pub is_error: bool,
}

/// The conversation as the model sees it.
#[derive(Clone, Debug, PartialEq)]
pub enum Msg {
    User(String),
    Assistant { text: String, calls: Vec<ToolCall> },
    Tools(Vec<ToolOutput>),
}

/// What the model answered.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reply {
    pub text: String,
    pub calls: Vec<ToolCall>,
}

/// `tools` are MCP tool definitions (`name`, `description`, `inputSchema`).
pub fn build(ep: &Endpoint, system: &str, msgs: &[Msg], tools: &[Value]) -> HttpRequest {
    let field = |t: &Value, k: &str| t.get(k).cloned().unwrap_or(Value::Null);
    let (url, auth, body) = match ep.protocol {
        Protocol::Anthropic => {
            let tools: Vec<Value> = tools
                .iter()
                .map(|t| json!({"name": field(t, "name"), "description": field(t, "description"), "input_schema": field(t, "inputSchema")}))
                .collect();
            let body = json!({
                "model": ep.model,
                "max_tokens": MAX_TOKENS,
                "system": system,
                "messages": anthropic_msgs(msgs),
                "tools": tools,
            });
            (format!("{}/messages", ep.base_url), ("x-api-key".to_string(), ep.api_key.clone()), body)
        }
        Protocol::OpenAi => {
            let tools: Vec<Value> = tools
                .iter()
                .map(|t| json!({"type": "function", "function": {"name": field(t, "name"), "description": field(t, "description"), "parameters": field(t, "inputSchema")}}))
                .collect();
            let mut messages = vec![json!({"role": "system", "content": system})];
            for m in msgs {
                openai_msgs(m, &mut messages);
            }
            let mut body = json!({"model": ep.model, "messages": messages, "tools": tools});
            // OpenAI's newer (reasoning) models reject `max_tokens` and want
            // `max_completion_tokens`; Gemini's compat layer, DeepSeek, Nvidia NIM, Fireworks,
            // xAI and local servers (Ollama, vLLM, llama.cpp) all take `max_tokens`, and not all
            // of them know the newer name. So: the new name for OpenAI's (and Azure OpenAI's)
            // hosts, the old one everywhere else.
            if ep.base_url.contains("openai.com") || ep.base_url.contains("openai.azure.com") {
                body["max_completion_tokens"] = json!(MAX_TOKENS_OPENAI);
            } else {
                body["max_tokens"] = json!(MAX_TOKENS);
            }
            (format!("{}/chat/completions", ep.base_url), ("authorization".to_string(), format!("Bearer {}", ep.api_key)), body)
        }
    };
    let mut headers = vec![("content-type".to_string(), "application/json".to_string()), auth];
    if ep.protocol == Protocol::Anthropic {
        headers.push(("anthropic-version".into(), "2023-06-01".into()));
    }
    HttpRequest { method: Method::Post, url, headers, body: body.to_string() }
}

/// The request listing a provider's models.
pub fn models_request(ep: &Endpoint) -> HttpRequest {
    let (url, auth) = match ep.protocol {
        Protocol::Anthropic => (format!("{}/models?limit=1000", ep.base_url), ("x-api-key".to_string(), ep.api_key.clone())),
        Protocol::OpenAi => (format!("{}/models", ep.base_url), ("authorization".to_string(), format!("Bearer {}", ep.api_key))),
    };
    let mut headers = vec![auth];
    if ep.protocol == Protocol::Anthropic {
        headers.push(("anthropic-version".into(), "2023-06-01".into()));
    }
    HttpRequest { method: Method::Get, url, headers, body: String::new() }
}

/// Model ids from a `/models` reply, exactly as the provider spells them: sorted, unique, capped.
pub fn parse_models(_protocol: Protocol, body: &str) -> Result<Vec<String>, String> {
    let v: Value = serde_json::from_str(body).map_err(|e| format!("unreadable reply ({e})"))?;
    if v.get("error").is_some_and(|e| !e.is_null()) {
        return Err(error_message(body));
    }
    // Both protocols answer {"data":[{"id":...}]}.
    let mut ids: Vec<String> = v
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| m.get("id").and_then(Value::as_str))
        .filter(|id| !id.trim().is_empty())
        .map(|id| id.chars().take(200).collect())
        .collect();
    ids.sort();
    ids.dedup();
    ids.truncate(1000);
    Ok(ids)
}

/// Anthropic wants user and assistant turns to alternate. A request cut short (an error, Stop)
/// leaves a user turn (the question, or tool results) unanswered, so consecutive turns of one
/// role are merged; the agent only ever appends a question after them, so tool results stay
/// first as the API requires.
fn anthropic_msgs(msgs: &[Msg]) -> Vec<Value> {
    let mut out: Vec<(&str, Vec<Value>)> = Vec::new();
    for m in msgs {
        let (role, blocks) = anthropic_msg(m);
        match out.last_mut() {
            _ if blocks.is_empty() => {}
            Some((r, b)) if *r == role => b.extend(blocks),
            _ => out.push((role, blocks)),
        }
    }
    out.into_iter().map(|(role, content)| json!({"role": role, "content": content})).collect()
}

fn anthropic_msg(m: &Msg) -> (&'static str, Vec<Value>) {
    match m {
        // An empty text block is rejected by the API.
        Msg::User(t) if t.is_empty() => ("user", vec![]),
        Msg::User(t) => ("user", vec![json!({"type": "text", "text": t})]),
        Msg::Assistant { text, calls } => {
            let mut content: Vec<Value> = if text.is_empty() { vec![] } else { vec![json!({"type": "text", "text": text})] };
            content.extend(calls.iter().map(|c| json!({"type": "tool_use", "id": c.id, "name": c.name, "input": object(&c.args)})));
            ("assistant", content)
        }
        Msg::Tools(outs) => (
            "user",
            outs.iter().map(|o| json!({"type": "tool_result", "tool_use_id": o.id, "content": nonempty(&o.text), "is_error": o.is_error})).collect(),
        ),
    }
}

/// Tool output for the wire: some servers reject an empty result.
fn nonempty(t: &str) -> &str {
    if t.is_empty() { "(no output)" } else { t }
}

fn openai_msgs(m: &Msg, out: &mut Vec<Value>) {
    match m {
        Msg::User(t) => out.push(json!({"role": "user", "content": t})),
        Msg::Assistant { text, calls } => {
            // A tool-call turn without text sends `content: null`, as the OpenAI spec and SDKs
            // do; some compatible servers reject an empty string there.
            let content = if text.is_empty() && !calls.is_empty() { Value::Null } else { json!(text) };
            let mut v = json!({"role": "assistant", "content": content});
            if !calls.is_empty() {
                v["tool_calls"] = calls
                    .iter()
                    .map(|c| json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": object(&c.args).to_string()}}))
                    .collect();
            }
            out.push(v);
        }
        Msg::Tools(outs) => out.extend(outs.iter().map(|o| json!({"role": "tool", "tool_call_id": o.id, "content": nonempty(&o.text)}))),
    }
}

/// Tool arguments are always an object on the wire.
fn object(v: &Value) -> Value {
    if v.is_object() { v.clone() } else { json!({}) }
}

/// Read a successful response body.
pub fn parse(protocol: Protocol, body: &str) -> Result<Reply, String> {
    let v: Value = serde_json::from_str(body).map_err(|e| format!("unreadable reply ({e})"))?;
    if v.get("error").is_some_and(|e| !e.is_null()) {
        return Err(error_message(body));
    }
    let str_of = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    let mut reply = Reply::default();
    match protocol {
        Protocol::Anthropic => {
            for block in v.get("content").and_then(Value::as_array).into_iter().flatten() {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => push_text(&mut reply.text, &str_of(block, "text")),
                    Some("tool_use") => reply.calls.push(ToolCall {
                        id: str_of(block, "id"),
                        name: str_of(block, "name"),
                        args: object(block.get("input").unwrap_or(&Value::Null)),
                    }),
                    _ => {}
                }
            }
        }
        Protocol::OpenAi => {
            let msg = v.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("message")).ok_or("reply has no message")?;
            push_text(&mut reply.text, &str_of(msg, "content"));
            for c in msg.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                let f = c.get("function").unwrap_or(&Value::Null);
                // Arguments arrive as a JSON string; a model that writes broken JSON gets `{}` and
                // the tool's own "missing parameter" error back.
                let args = f.get("arguments").and_then(Value::as_str).and_then(|a| serde_json::from_str(a).ok()).unwrap_or(json!({}));
                reply.calls.push(ToolCall { id: str_of(c, "id"), name: str_of(f, "name"), args: object(&args) });
            }
        }
    }
    // Some OpenAI-compatible servers leave ids out; the next request needs them to match.
    for (i, c) in reply.calls.iter_mut().enumerate() {
        if c.id.is_empty() {
            c.id = format!("call_{i}");
        }
    }
    Ok(reply)
}

fn push_text(out: &mut String, t: &str) {
    if t.is_empty() {
        return;
    }
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str(t);
}

/// A readable message from an error body (`{"error": {"message": …}}` in both APIs), else the
/// body itself, shortened.
pub fn error_message(body: &str) -> String {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let e = v.get("error");
    let msg = e.and_then(|e| e.get("message")).and_then(Value::as_str).or_else(|| e.and_then(Value::as_str));
    match msg {
        Some(m) => m.to_string(),
        None => crate::clip(body.trim(), 300),
    }
}
