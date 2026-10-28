use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, channel};

use serde_json::{Value, json};
use wordcraft_doc::Document;
use wordcraft_engine::Session;

use super::*;

/// A fake provider: answers requests from a queue and keeps what was sent.
#[derive(Clone, Default)]
struct Fake {
    answers: Rc<RefCell<VecDeque<Result<String, String>>>>,
    sent: Rc<RefCell<Vec<HttpRequest>>>,
}

impl Fake {
    fn new(answers: Vec<Result<Value, String>>) -> Self {
        let f = Fake::default();
        f.answers.borrow_mut().extend(answers.into_iter().map(|a| a.map(|v| v.to_string())));
        f
    }
    fn http(&self) -> Box<Http> {
        let f = self.clone();
        Box::new(move |req: HttpRequest| -> Receiver<Result<String, String>> {
            f.sent.borrow_mut().push(req);
            let (tx, rx) = channel();
            if let Some(a) = f.answers.borrow_mut().pop_front() {
                let _ = tx.send(a);
            }
            rx
        })
    }
    fn body(&self, i: usize) -> Value {
        serde_json::from_str(&self.sent.borrow()[i].body).unwrap()
    }
}

fn settings(provider: &str) -> Settings {
    let mut s = Settings { provider: provider.into(), ..Default::default() };
    s.current_mut().api_key = "test-key".into();
    if provider == providers::CUSTOM {
        s.current_mut().base_url = "http://localhost:11434/v1/".into();
        s.current_mut().model = "local-model".into();
    }
    s
}

fn session(text: &str) -> Session {
    let mut s = Session::new(Document::new());
    s.run("text.insert", &json!({"text": text})).unwrap();
    s.reset_history();
    s.dirty = false;
    s
}

/// Run until the agent is idle.
fn drive(agent: &mut Agent, s: &mut Session, http: &Http) {
    for _ in 0..100 {
        if !agent.busy() {
            return;
        }
        agent.poll(s, http);
    }
    panic!("agent never finished");
}

fn text(s: &mut Session) -> String {
    s.run("document.text", &json!({})).unwrap()["text"].as_str().unwrap().trim().to_string()
}

fn anthropic_tools(calls: &[(&str, Value)]) -> Value {
    let content: Vec<Value> = calls
        .iter()
        .enumerate()
        .map(|(i, (name, input))| json!({"type": "tool_use", "id": format!("tu_{i}"), "name": name, "input": input}))
        .collect();
    json!({"content": content, "stop_reason": "tool_use"})
}

fn anthropic_text(t: &str) -> Value {
    json!({"content": [{"type": "text", "text": t}], "stop_reason": "end_turn"})
}

fn openai_tools(calls: &[(&str, Value)]) -> Value {
    let tc: Vec<Value> = calls
        .iter()
        .enumerate()
        .map(|(i, (name, args))| json!({"id": format!("call_{i}"), "type": "function", "function": {"name": name, "arguments": args.to_string()}}))
        .collect();
    json!({"choices": [{"message": {"role": "assistant", "content": null, "tool_calls": tc}}]})
}

fn openai_text(t: &str) -> Value {
    json!({"choices": [{"message": {"role": "assistant", "content": t}}]})
}

#[test]
fn claude_edits_are_tracked_and_one_undo_step() {
    let mut s = session("The quick fox");
    let fake = Fake::new(vec![
        Ok(anthropic_tools(&[("get_text", json!({})), ("select_text", json!({"text": "quick"})), ("type_text", json!({"text": "slow"}))])),
        Ok(anthropic_tools(&[("execute", json!({"command": "format.bold"}))])),
        Ok(anthropic_tools(&[("get_text", json!({}))])),
        Ok(anthropic_text("Replaced “quick” with “slow” and made it bold.")),
    ]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("make it slow", &settings("claude"), &mut s, &*http).unwrap();
    assert!(s.doc.settings.track_changes, "tracking is on while the assistant works");
    drive(&mut agent, &mut s, &*http);

    // The request: Anthropic shape, key in its header, only the allowed tools.
    let req = fake.sent.borrow()[0].clone();
    assert_eq!(req.url, "https://api.anthropic.com/v1/messages");
    assert!(req.headers.contains(&("x-api-key".into(), "test-key".into())));
    let names: Vec<String> = fake.body(0)["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap().to_string()).collect();
    assert_eq!(names.len(), TOOLS.len() + 2);
    assert!(names.iter().any(|n| n == "plan") && names.iter().any(|n| n == "ask_user"));
    assert!(!names.iter().any(|n| n == "save_document" || n == "open_document"));
    // The second request answers every tool call by id.
    let second = fake.body(1);
    let results = second["messages"].as_array().unwrap().last().unwrap()["content"].as_array().unwrap().clone();
    assert_eq!(results.iter().map(|r| r["tool_use_id"].as_str().unwrap()).collect::<Vec<_>>(), ["tu_0", "tu_1", "tu_2"]);

    assert!(matches!(agent.log.last(), Some(Entry::Assistant(_))));
    assert!(!s.doc.settings.track_changes, "tracking is put back");
    assert_eq!(s.author, "WordCraft User", "the author is put back");
    assert!(text(&mut s).contains("slow"));

    // Everything the assistant did is one Undo step…
    assert_eq!(s.undo_depth(), 1, "{:?}", s.undo_labels());
    s.undo();
    assert_eq!(text(&mut s), "The quick fox");
    // …and, redone, tracked changes the user can reject.
    s.redo();
    s.run("review.rejectAll", &json!({})).unwrap();
    assert_eq!(text(&mut s), "The quick fox");
}

#[test]
fn openai_compatible_providers_work_and_quote_the_selection() {
    for provider in ["openai", "gemini", "nvidia", "fireworks", "deepseek", "grok", providers::CUSTOM] {
        let mut s = session("Hello");
        s.run("select.all", &json!({})).unwrap();
        let fake = Fake::new(vec![
            Ok(openai_tools(&[("get_text", json!({})), ("type_text", json!({"text": "Hi there"}))])),
            Ok(openai_tools(&[("get_text", json!({}))])),
            Ok(openai_text("Done.")),
        ]);
        let http = fake.http();
        let mut agent = Agent::default();
        agent.ask("friendlier", &settings(provider), &mut s, &*http).unwrap();
        drive(&mut agent, &mut s, &*http);

        let req = fake.sent.borrow()[0].clone();
        assert!(req.url.ends_with("/chat/completions") && !req.url.contains("//chat"), "{provider}: {}", req.url);
        assert!(req.headers.contains(&("authorization".into(), "Bearer test-key".into())), "{provider}");
        let first = fake.body(0);
        assert_eq!(first["messages"][0]["role"], "system");
        assert!(first["messages"][1]["content"].as_str().unwrap().contains("<selection>\nHello"), "{provider}");
        // The tool result goes back as a `tool` message with the call's id.
        let second = fake.body(1);
        let last = second["messages"].as_array().unwrap().last().unwrap().clone();
        assert_eq!((last["role"].as_str(), last["tool_call_id"].as_str()), (Some("tool"), Some("call_1")), "{provider}");
        assert_eq!(text(&mut s).replace("Hello", ""), "Hi there", "{provider}: tracked deletion keeps the old text");
        assert_eq!(agent.log.last(), Some(&Entry::Assistant("Done.".into())));
    }
}

#[test]
fn blocked_commands_are_refused_and_hidden() {
    let mut s = session("Text");
    let fake = Fake::new(vec![
        Ok(anthropic_tools(&[
            ("execute", json!({"command": "file.save", "params": {"path": "/tmp/x.docx"}})),
            ("execute", json!({"command": "review.trackChanges", "params": {"value": false}})),
            ("execute", json!({"command": "edit.undo"})),
            ("open_document", json!({"path": "/etc/passwd"})),
            ("list_commands", json!({"query": "file."})),
        ])),
        Ok(anthropic_text("ok")),
    ]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("go", &settings("claude"), &mut s, &*http).unwrap();
    drive(&mut agent, &mut s, &*http);
    let second = fake.body(1);
    let results = second["messages"].as_array().unwrap().last().unwrap()["content"].as_array().unwrap().clone();
    for r in &results[..4] {
        assert_eq!(r["is_error"], true, "{r}");
    }
    let listed = results[4]["content"].as_str().unwrap();
    assert!(!listed.contains("\"file.save\""), "blocked commands aren't listed");
    assert!(s.path.is_none() && !s.dirty);
}

#[test]
fn errors_end_the_request_and_restore_the_session() {
    let mut s = session("Text");
    let fake = Fake::new(vec![Err("HTTP 401: invalid x-api-key".into())]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("hi", &settings("claude"), &mut s, &*http).unwrap();
    drive(&mut agent, &mut s, &*http);
    assert_eq!(agent.log.last(), Some(&Entry::Error("HTTP 401: invalid x-api-key".into())));
    assert!(!s.doc.settings.track_changes);
    assert_eq!(s.author, "WordCraft User");

    // No key: refused before anything is sent.
    let none = Settings { provider: "openai".into(), ..Default::default() };
    assert!(agent.ask("hi", &none, &mut s, &*http).is_err());
    assert!(matches!(agent.log.last(), Some(Entry::Error(e)) if e.contains("API key")));
    assert_eq!(fake.sent.borrow().len(), 1);
}

#[test]
fn a_looping_model_is_stopped() {
    let mut s = session("Text");
    let fake = Fake::new((0..MAX_ROUNDS + 5).map(|_| Ok(anthropic_tools(&[("get_text", json!({}))]))).collect());
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("loop", &settings("claude"), &mut s, &*http).unwrap();
    drive(&mut agent, &mut s, &*http);
    assert_eq!(fake.sent.borrow().len() as u32, MAX_ROUNDS);
    assert!(matches!(agent.log.last(), Some(Entry::Error(e)) if e.contains("Stopped")));
}

#[test]
fn hostile_replies_never_panic() {
    let bodies = [
        "",
        "not json",
        "[]",
        "{}",
        r#"{"content": "x"}"#,
        r#"{"content": [{"type": "tool_use"}]}"#,
        r#"{"content": [{"type": "tool_use", "name": "execute", "input": [1, 2]}]}"#,
        r#"{"content": [{"type": "tool_use", "name": "execute", "input": {"command": 7, "params": "x"}}]}"#,
        r#"{"choices": []}"#,
        r#"{"choices": [{"message": {"tool_calls": [{"function": {"name": "batch", "arguments": "{broken"}}]}}]}"#,
        r#"{"choices": [{"message": {"tool_calls": [{"function": {"name": "batch", "arguments": "{\"commands\": [{\"command\": \"text.delete\", \"params\": {\"count\": -99999999999}}]}"}}]}}]}"#,
        r#"{"error": {"message": "overloaded"}}"#,
        r#"{"error": "plain"}"#,
    ];
    for protocol in [Protocol::Anthropic, Protocol::OpenAi] {
        for b in bodies {
            let mut s = session("Some text here");
            let fake = Fake::new(vec![Ok(Value::Null), Ok(anthropic_text("end"))]);
            fake.answers.borrow_mut()[0] = Ok(b.to_string());
            let http = fake.http();
            let mut agent = Agent::default();
            let provider = if protocol == Protocol::Anthropic { "claude" } else { "openai" };
            agent.ask("x", &settings(provider), &mut s, &*http).unwrap();
            drive(&mut agent, &mut s, &*http);
        }
    }
}

#[test]
fn settings_resolve_presets_and_overrides() {
    let gemini = settings("gemini").endpoint().unwrap();
    assert_eq!(gemini.protocol, Protocol::OpenAi);
    assert_eq!(gemini.base_url, "https://generativelanguage.googleapis.com/v1beta/openai");
    let custom = settings(providers::CUSTOM).endpoint().unwrap();
    assert_eq!((custom.base_url.as_str(), custom.model.as_str()), ("http://localhost:11434/v1", "local-model"));

    let mut no_url = Settings { provider: providers::CUSTOM.into(), ..Default::default() };
    no_url.current_mut().api_key = "k".into();
    assert!(no_url.endpoint().is_err());
    // An unknown provider id falls back to the first preset.
    assert_eq!(Settings { provider: "nope".into(), ..Default::default() }.provider_id(), "claude");
    assert_eq!(clip("héllo wörld", 4), "héll…");
}

#[test]
fn user_author_is_known_only_while_running() {
    let mut s = session("Text");
    let fake = Fake::new(vec![Ok(anthropic_text("hi"))]);
    let http = fake.http();
    let mut agent = Agent::default();
    assert_eq!(agent.user_author(), None);
    agent.ask("hi", &settings("claude"), &mut s, &*http).unwrap();
    assert_eq!(agent.user_author(), Some("WordCraft User"));
    assert!(s.author.starts_with("AI ("), "{}", s.author);
    drive(&mut agent, &mut s, &*http);
    assert_eq!(agent.user_author(), None);
    assert_eq!(s.author, "WordCraft User");
}

#[test]
fn the_transcript_stays_valid_after_an_error_or_stop() {
    let mut s = session("Text");
    // A tool round, then a failure: the next request must still alternate roles, with the tool
    // results right after the tool_use turn.
    let fake = Fake::new(vec![Ok(anthropic_tools(&[("get_text", json!({}))])), Err("HTTP 529: overloaded".into()), Ok(anthropic_text("ok"))]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("first", &settings("claude"), &mut s, &*http).unwrap();
    drive(&mut agent, &mut s, &*http);
    agent.ask("second", &settings("claude"), &mut s, &*http).unwrap();
    // Stop before the answer is read, then ask again: two unanswered questions in a row.
    agent.stop(&mut s);
    agent.ask("third", &settings("claude"), &mut s, &*http).unwrap();
    let body = fake.body(3);
    let msgs = body["messages"].as_array().unwrap();
    let roles: Vec<&str> = msgs.iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["user", "assistant", "user"]);
    let last: Vec<&str> = msgs[2]["content"].as_array().unwrap().iter().map(|b| b["type"].as_str().unwrap()).collect();
    assert_eq!(last, ["tool_result", "text", "text"]);
    drive(&mut agent, &mut s, &*http);
}

#[test]
fn openai_wire_details() {
    let ep = |base: &str| Endpoint { protocol: Protocol::OpenAi, base_url: base.into(), model: "m".into(), api_key: "k".into() };
    let msgs = [
        wire::Msg::User("hi".into()),
        wire::Msg::Assistant { text: String::new(), calls: vec![wire::ToolCall { id: "c1".into(), name: "get_text".into(), args: json!({}) }] },
        wire::Msg::Tools(vec![wire::ToolOutput { id: "c1".into(), text: String::new(), is_error: false }]),
    ];
    let body = |base: &str| -> Value { serde_json::from_str(&wire::build(&ep(base), "sys", &msgs, &[]).body).unwrap() };
    let openai = body("https://api.openai.com/v1");
    assert!(openai.get("max_tokens").is_none() && openai["max_completion_tokens"].is_u64());
    let other = body("https://api.deepseek.com/v1");
    assert!(other.get("max_completion_tokens").is_none() && other["max_tokens"].is_u64());
    assert!(other["messages"][2]["content"].is_null(), "a tool-call turn without text sends null");
    assert_eq!(other["messages"][3]["content"], "(no output)");
}

#[test]
fn the_users_typing_after_a_request_is_its_own_undo_step() {
    let mut s = session("Text");
    s.run("caret.docEnd", &json!({})).unwrap();
    let fake = Fake::new(vec![
        Ok(anthropic_tools(&[("get_text", json!({})), ("type_text", json!({"text": " AI"}))])),
        Ok(anthropic_tools(&[("get_text", json!({}))])),
        Ok(anthropic_text("done")),
    ]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("add", &settings("claude"), &mut s, &*http).unwrap();
    drive(&mut agent, &mut s, &*http);
    s.run("text.insert", &json!({"text": " me"})).unwrap();
    assert_eq!(s.undo_depth(), 2, "{:?}", s.undo_labels());
}

mod harness;

#[test]
fn models_request_shape() {
    let a = settings("claude").connection().unwrap();
    let r = wire::models_request(&a);
    assert_eq!(r.method, Method::Get);
    assert_eq!(r.url, "https://api.anthropic.com/v1/models?limit=1000");
    assert!(r.body.is_empty());
    assert!(r.headers.contains(&("x-api-key".into(), "test-key".into())));
    assert!(r.headers.contains(&("anthropic-version".into(), "2023-06-01".into())));
    let o = wire::models_request(&settings("openai").connection().unwrap());
    assert_eq!((o.method, o.url.as_str()), (Method::Get, "https://api.openai.com/v1/models"));
    assert!(o.headers.contains(&("authorization".into(), "Bearer test-key".into())));
    assert!(o.body.is_empty());
}

#[test]
fn connection_needs_no_model() {
    let mut s = settings(providers::CUSTOM);
    s.current_mut().model.clear();
    assert!(s.connection().is_ok());
    assert!(s.endpoint().is_err());
    s.current_mut().api_key.clear();
    assert!(s.connection().is_err());
}

#[test]
fn parse_models_replies() {
    let anth = r#"{"data":[{"id":"claude-b","display_name":"B"},{"id":"claude-a"}],"has_more":false}"#;
    assert_eq!(wire::parse_models(Protocol::Anthropic, anth).unwrap(), ["claude-a", "claude-b"]);
    let gem = r#"{"object":"list","data":[{"id":"models/gemini-2.5-flash"},{"id":"models/gemini-2.5-flash"},{"id":""},{"id":5},{}]}"#;
    assert_eq!(wire::parse_models(Protocol::OpenAi, gem).unwrap(), ["models/gemini-2.5-flash"]);
    let err = wire::parse_models(Protocol::OpenAi, r#"{"error":{"message":"bad key"}}"#).unwrap_err();
    assert_eq!(err, "bad key");
    let many = format!(r#"{{"data":[{}]}}"#, (0..1500).map(|i| format!(r#"{{"id":"m{i}"}}"#)).collect::<Vec<_>>().join(","));
    assert_eq!(wire::parse_models(Protocol::OpenAi, &many).unwrap().len(), 1000);
    let long = format!(r#"{{"data":[{{"id":"{}"}}]}}"#, "x".repeat(500));
    assert_eq!(wire::parse_models(Protocol::OpenAi, &long).unwrap()[0].chars().count(), 200);
}

#[test]
fn parse_models_hostile() {
    for b in ["", "null", "[]", "{}", r#"{"data":null}"#, r#"{"data":"x"}"#, r#"{"data":[null,[],{"id":null}]}"#, r#"{"error":1}"#] {
        let _ = wire::parse_models(Protocol::Anthropic, b);
        let _ = wire::parse_models(Protocol::OpenAi, b);
    }
}
