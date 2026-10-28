//! The harness rules: read before edit, plan, permission prompts, questions, verification,
//! elision, selection scope.

use serde_json::{Value, json};
use wordcraft_doc::{Document, StoryRef};
use wordcraft_engine::Session;

use super::*;

const P1: &str = "Our studio opened in the spring and has grown every season since then, thanks to our neighbours.";
const P2: &str = "This paragraph describes the old warehouse we used before the move, with its leaking roof and noisy pipes.";
const P3: &str = "Next year we plan to open a second room for workshops, readings and small concerts in the evenings.";

fn session_paras(paras: &[&str]) -> Session {
    let mut s = Session::new(Document::new());
    for (i, p) in paras.iter().enumerate() {
        if i > 0 {
            s.run("text.newParagraph", &json!({})).unwrap();
        }
        s.run("text.insert", &json!({"text": p, "raw": true})).unwrap();
    }
    s.reset_history();
    s.dirty = false;
    s
}

/// The body as it reads with every change accepted.
fn accepted(s: &Session) -> String {
    let paths = s.doc.para_paths(StoryRef::Body);
    paths.iter().filter_map(|p| s.doc.para(StoryRef::Body, p)).map(|p| p.final_text()).collect::<Vec<_>>().join("\n")
}

/// The tool results the `i`-th request sends back last (Anthropic).
fn last_results(fake: &Fake, i: usize) -> Vec<Value> {
    let body = fake.body(i);
    let msgs = body["messages"].as_array().unwrap();
    msgs.last().unwrap()["content"].as_array().unwrap().iter().filter(|b| b["type"] == "tool_result").cloned().collect()
}

fn delete_p2() -> Value {
    anthropic_tools(&[("get_text", json!({})), ("select_text", json!({"text": P2})), ("execute", json!({"command": "text.delete"}))])
}

#[test]
fn edits_are_refused_until_the_document_is_read() {
    let mut s = session("Hello");
    s.run("caret.docEnd", &json!({})).unwrap();
    let fake = Fake::new(vec![
        Ok(anthropic_tools(&[("type_text", json!({"text": " world"})), ("execute", json!({"command": "format.bold"}))])),
        Ok(anthropic_tools(&[("get_text", json!({})), ("type_text", json!({"text": " world"}))])),
        Ok(anthropic_tools(&[("get_text", json!({}))])),
        Ok(anthropic_text("Added “world”.")),
    ]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("add world", &settings("claude"), &mut s, &*http).unwrap();
    drive(&mut agent, &mut s, &*http);
    let refused = last_results(&fake, 1);
    assert!(refused.iter().all(|r| r["is_error"] == true && r["content"].as_str().unwrap().contains("read the document first")), "{refused:?}");
    let ok = last_results(&fake, 2);
    assert!(ok.iter().all(|r| r["is_error"] == false), "{ok:?}");
    assert_eq!(accepted(&s), "Hello world");

    // Each request reads again.
    let fake = Fake::new(vec![Ok(anthropic_tools(&[("type_text", json!({"text": "!"}))])), Ok(anthropic_text("no"))]);
    let http = fake.http();
    agent.ask("again", &settings("claude"), &mut s, &*http).unwrap();
    drive(&mut agent, &mut s, &*http);
    assert_eq!(last_results(&fake, 1)[0]["is_error"], true);
    assert_eq!(accepted(&s), "Hello world");
}

#[test]
fn the_plan_is_stored_and_replaced() {
    let mut s = session("Text");
    let fake = Fake::new(vec![
        Ok(anthropic_tools(&[("plan", json!({"steps": [{"text": "Read", "done": true}, {"text": "Fix the title", "done": false}]}))])),
        Ok(anthropic_tools(&[("plan", json!({"steps": [{"text": "Fix the title", "done": true}]}))])),
        Ok(anthropic_text("ok")),
    ]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("go", &settings("claude"), &mut s, &*http).unwrap();
    agent.poll(&mut s, &*http);
    assert_eq!(agent.plan(), [PlanStep { text: "Read".into(), done: true }, PlanStep { text: "Fix the title".into(), done: false }]);
    assert!(last_results(&fake, 1)[0]["content"].as_str().unwrap().contains("1 of 2"));
    agent.poll(&mut s, &*http);
    assert_eq!(agent.plan(), [PlanStep { text: "Fix the title".into(), done: true }]);
    drive(&mut agent, &mut s, &*http);
    assert_eq!(agent.plan().len(), 1, "the last plan stays visible after the request");
    // A new request starts without a plan.
    let fake = Fake::new(vec![Ok(anthropic_text("hi"))]);
    let http = fake.http();
    agent.ask("hello", &settings("claude"), &mut s, &*http).unwrap();
    assert!(agent.plan().is_empty());
    drive(&mut agent, &mut s, &*http);
}

#[test]
fn deleting_a_paragraph_asks_and_allow_once_runs_it() {
    let mut s = session_paras(&[P1, P2, P3]);
    let fake = Fake::new(vec![Ok(delete_p2()), Ok(anthropic_tools(&[("get_text", json!({}))])), Ok(anthropic_text("Deleted it."))]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("delete the warehouse paragraph", &settings("claude"), &mut s, &*http).unwrap();
    agent.poll(&mut s, &*http);
    let Some(Pending::Permission { kind, description }) = agent.pending().cloned() else { panic!("{:?}", agent.pending()) };
    assert_eq!(kind, PermissionKind::Delete);
    assert!(description.contains("warehouse"), "{description}");
    assert!(agent.busy());
    assert_eq!(accepted(&s), format!("{P1}\n{P2}\n{P3}"), "nothing changes while it waits");
    assert_eq!(fake.sent.borrow().len(), 1);
    // Polling while paused does nothing.
    assert!(!agent.poll(&mut s, &*http));

    agent.decide(Decision::AllowOnce, &mut s, &*http);
    assert!(agent.pending().is_none());
    drive(&mut agent, &mut s, &*http);
    assert_eq!(accepted(&s), format!("{P1}\n\n{P3}"));
    assert!(agent.log.iter().any(|e| matches!(e, Entry::Permission { granted: true, .. })));
    assert_eq!(s.undo_depth(), 1, "{:?}", s.undo_labels());
    s.run("review.rejectAll", &json!({})).unwrap();
    assert_eq!(accepted(&s), format!("{P1}\n{P2}\n{P3}"));
}

#[test]
fn deny_answers_with_the_refusal_and_changes_nothing() {
    let mut s = session_paras(&[P1, P2, P3]);
    let fake = Fake::new(vec![Ok(delete_p2()), Ok(anthropic_text("I left it."))]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("delete it", &settings("claude"), &mut s, &*http).unwrap();
    agent.poll(&mut s, &*http);
    assert!(matches!(agent.pending(), Some(Pending::Permission { .. })));
    // A question's answer doesn't settle a permission prompt.
    agent.answer("yes", &mut s, &*http);
    assert!(agent.pending().is_some());
    agent.decide(Decision::Deny, &mut s, &*http);
    drive(&mut agent, &mut s, &*http);
    let r = last_results(&fake, 1);
    assert_eq!(r.len(), 3);
    assert_eq!(r[2]["is_error"], true);
    assert!(r[2]["content"].as_str().unwrap().contains("declined"));
    assert_eq!(accepted(&s), format!("{P1}\n{P2}\n{P3}"));
    assert_eq!(s.undo_depth(), 0);
    assert!(agent.log.iter().any(|e| matches!(e, Entry::Permission { granted: false, .. })));
}

#[test]
fn allow_for_request_covers_the_rest_of_the_request_only() {
    let mut s = session_paras(&[P1, P2, P3]);
    let fake = Fake::new(vec![
        Ok(delete_p2()),
        Ok(anthropic_tools(&[("select_text", json!({"text": P3})), ("execute", json!({"command": "text.delete"}))])),
        Ok(anthropic_tools(&[("get_text", json!({}))])),
        Ok(anthropic_text("Deleted both.")),
        Ok(anthropic_tools(&[("get_text", json!({})), ("select_text", json!({"text": P1})), ("execute", json!({"command": "text.delete"}))])),
    ]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("delete the last two paragraphs", &settings("claude"), &mut s, &*http).unwrap();
    agent.poll(&mut s, &*http);
    agent.decide(Decision::AllowForRequest, &mut s, &*http);
    for _ in 0..10 {
        assert!(agent.pending().is_none(), "no second prompt in the same request");
        agent.poll(&mut s, &*http);
    }
    assert!(!agent.busy());
    assert_eq!(accepted(&s), format!("{P1}\n\n"));
    // The next request asks again.
    agent.ask("and the first", &settings("claude"), &mut s, &*http).unwrap();
    agent.poll(&mut s, &*http);
    assert!(matches!(agent.pending(), Some(Pending::Permission { kind: PermissionKind::Delete, .. })));
    agent.stop(&mut s);
}

#[test]
fn rewriting_a_sentence_does_not_ask() {
    let mut s = session_paras(&[P1, P2]);
    let fake = Fake::new(vec![
        Ok(anthropic_tools(&[
            ("get_text", json!({})),
            ("select_text", json!({"text": P2})),
            ("type_text", json!({"text": "Before the move we worked in an old warehouse whose roof leaked and whose pipes were noisy."})),
            ("select_text", json!({"text": "in the spring"})),
            ("type_text", json!({"text": "in April"})),
        ])),
        Ok(anthropic_tools(&[("get_text", json!({}))])),
        Ok(anthropic_text("Rewrote it.")),
    ]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("rewrite the second paragraph", &settings("claude"), &mut s, &*http).unwrap();
    for _ in 0..10 {
        assert!(agent.pending().is_none());
        agent.poll(&mut s, &*http);
    }
    assert!(!agent.busy());
    let text = accepted(&s);
    assert!(text.contains("whose roof leaked") && text.contains("in April") && !text.contains("leaking"), "{text}");
}

#[test]
fn removal_thresholds() {
    use guard::{Measure, removal};
    let m = |original: usize, own: usize, paras: usize| Measure { original, own, paras, objects: 0 };
    // A rewrite keeps about as much as it removes.
    assert!(removal(&m(1000, 0, 5), &m(800, 190, 5), 1000, "").is_none());
    // Net loss.
    assert!(removal(&m(1000, 0, 5), &m(870, 0, 5), 1000, "").is_some());
    // A long passage replaced by a much shorter one.
    assert!(removal(&m(1000, 0, 5), &m(930, 15, 5), 1000, "").is_some());
    // A short shortening doesn't ask.
    assert!(removal(&m(1000, 0, 5), &m(960, 20, 5), 1000, "").is_none());
    // A paragraph went; merging two paragraphs (nothing removed) doesn't ask.
    assert!(removal(&m(1000, 0, 5), &m(985, 0, 4), 1000, "").is_some());
    assert!(removal(&m(1000, 0, 5), &m(1000, 0, 4), 1000, "").is_none());
    // Small deletions add up over a request.
    assert!(removal(&m(720, 0, 5), &m(690, 0, 5), 1000, "").is_some());
    // A picture went.
    assert!(removal(&Measure { objects: 2, ..m(1000, 0, 5) }, &m(1000, 0, 5), 1000, "").is_some());
}

#[test]
fn style_changes_ask_and_ordinary_formatting_does_not() {
    assert!(style_change("format.color", &json!({"color": "FF0000"})));
    assert!(style_change("format.font", &json!({"name": "Arial"})));
    assert!(style_change("format.set", &json!({"props": {"size": 14}})));
    assert!(!style_change("format.set", &json!({"props": {"bold": true}})));
    assert!(!style_change("format.set", &json!({"props": "junk"})));
    assert!(!style_change("format.bold", &json!({})));
    assert!(!style_change("para.style", &json!({"style": "Heading1"})));
    // Every listed id is a real command.
    let s = session("x");
    for id in guard::STYLE_COMMANDS {
        assert!(s.registry.get(id).is_some(), "{id}");
    }

    let mut s = session("Make this red");
    let fake = Fake::new(vec![
        Ok(anthropic_tools(&[
            ("get_text", json!({})),
            ("select_text", json!({"text": "red"})),
            ("execute", json!({"command": "format.color", "params": {"color": "FF0000"}})),
        ])),
        Ok(anthropic_tools(&[("get_text", json!({}))])),
        Ok(anthropic_text("Coloured.")),
    ]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("make red red", &settings("claude"), &mut s, &*http).unwrap();
    agent.poll(&mut s, &*http);
    let Some(Pending::Permission { kind, description }) = agent.pending().cloned() else { panic!("{:?}", agent.pending()) };
    assert_eq!(kind, PermissionKind::Style);
    assert!(description.contains("Font Color") && description.contains("red"), "{description}");
    agent.decide(Decision::AllowOnce, &mut s, &*http);
    drive(&mut agent, &mut s, &*http);
    s.run("select.text", &json!({"text": "red"})).unwrap();
    assert_eq!(s.run("format.state", &json!({})).unwrap()["color"], "FF0000");
}

#[test]
fn ask_user_pauses_until_answered() {
    let mut s = session("Text");
    let fake = Fake::new(vec![
        Ok(anthropic_tools(&[("ask_user", json!({"question": "Formal or casual?", "options": ["Formal", "Casual"]}))])),
        Ok(anthropic_text("ok")),
    ]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("rewrite it", &settings("claude"), &mut s, &*http).unwrap();
    agent.poll(&mut s, &*http);
    assert_eq!(agent.pending(), Some(&Pending::Question { question: "Formal or casual?".into(), options: vec!["Formal".into(), "Casual".into()] }));
    assert!(agent.busy());
    // A permission decision doesn't answer a question.
    agent.decide(Decision::AllowOnce, &mut s, &*http);
    assert!(agent.pending().is_some());
    assert_eq!(fake.sent.borrow().len(), 1);
    agent.answer("Casual, please", &mut s, &*http);
    assert!(agent.pending().is_none());
    let r = last_results(&fake, 1);
    assert_eq!((r[0]["content"].as_str(), r[0]["is_error"].as_bool()), (Some("Casual, please"), Some(false)));
    drive(&mut agent, &mut s, &*http);
    assert!(agent.log.iter().any(|e| matches!(e, Entry::Question { .. })));
    assert!(agent.log.contains(&Entry::User("Casual, please".into())));
}

#[test]
fn the_verification_reminder_is_sent_once() {
    let mut s = session("Hello");
    s.run("caret.docEnd", &json!({})).unwrap();
    let fake = Fake::new(vec![
        Ok(anthropic_tools(&[("get_text", json!({})), ("type_text", json!({"text": "!"}))])),
        Ok(anthropic_text("Done.")),
        Ok(anthropic_text("Really done.")),
    ]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("add !", &settings("claude"), &mut s, &*http).unwrap();
    drive(&mut agent, &mut s, &*http);
    assert_eq!(fake.sent.borrow().len(), 3);
    let body = fake.body(2);
    let msgs = body["messages"].as_array().unwrap();
    let roles: Vec<&str> = msgs.iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["user", "assistant", "user", "assistant", "user"]);
    let last = msgs.last().unwrap()["content"][0]["text"].as_str().unwrap();
    assert!(last.starts_with("<system-reminder>") && last.ends_with("</system-reminder>"));
    assert_eq!(agent.log.last(), Some(&Entry::Assistant("Really done.".into())));

    // Reading back after the edit: no reminder.
    let fake = Fake::new(vec![
        Ok(anthropic_tools(&[("get_text", json!({})), ("type_text", json!({"text": "?"})), ("get_text", json!({}))])),
        Ok(anthropic_text("Done.")),
    ]);
    let http = fake.http();
    agent.ask("add ?", &settings("claude"), &mut s, &*http).unwrap();
    drive(&mut agent, &mut s, &*http);
    assert_eq!(fake.sent.borrow().len(), 2);
}

#[test]
fn old_tool_output_is_elided() {
    let mut s = session(&"Long text here. ".repeat(40));
    let mut answers: Vec<Result<Value, String>> = (0..6).map(|_| Ok(anthropic_tools(&[("get_text", json!({}))]))).collect();
    answers.push(Ok(anthropic_text("ok")));
    let fake = Fake::new(answers);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("read", &settings("claude"), &mut s, &*http).unwrap();
    drive(&mut agent, &mut s, &*http);
    let body = fake.body(6);
    let results: Vec<String> = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["content"].as_array())
        .flatten()
        .filter(|b| b["type"] == "tool_result")
        .map(|b| b["content"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(results.len(), 6);
    assert!(results[..3].iter().all(|r| r == ELIDED), "{results:?}");
    assert!(results[3..].iter().all(|r| r.contains("Long text")));

    // The OpenAI shape keeps every call answered too.
    let mut answers: Vec<Result<Value, String>> = (0..5).map(|_| Ok(openai_tools(&[("get_text", json!({}))]))).collect();
    answers.push(Ok(openai_text("ok")));
    let fake = Fake::new(answers);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("read", &settings("openai"), &mut s, &*http).unwrap();
    drive(&mut agent, &mut s, &*http);
    let body = fake.body(5);
    let tools: Vec<&Value> = body["messages"].as_array().unwrap().iter().filter(|m| m["role"] == "tool").collect();
    assert_eq!(tools.len(), 5);
    assert!(tools.iter().any(|m| m["content"] == ELIDED) && tools.iter().all(|m| m["tool_call_id"].is_string()));
}

#[test]
fn edits_outside_the_selection_are_refused() {
    let mut s = session_paras(&[P1, P2, P3]);
    s.run("select.text", &json!({"text": P2})).unwrap();
    let fake = Fake::new(vec![
        Ok(anthropic_tools(&[
            ("get_text", json!({})),
            ("select_text", json!({"text": "spring"})),
            ("type_text", json!({"text": "autumn"})),
            ("select_text", json!({"text": "leaking"})),
            ("type_text", json!({"text": "dripping"})),
            // A new paragraph inside the selection moves its end, not its neighbours.
            ("execute", json!({"command": "text.newParagraph"})),
            ("type_text", json!({"text": "Added."})),
            ("select_text", json!({"text": "noisy"})),
            ("type_text", json!({"text": "loud"})),
            ("select_text", json!({"text": "concerts"})),
            ("type_text", json!({"text": "plays"})),
        ])),
        Ok(anthropic_tools(&[("get_text", json!({}))])),
        Ok(anthropic_text("ok")),
    ]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("fix the wording", &settings("claude"), &mut s, &*http).unwrap();
    drive(&mut agent, &mut s, &*http);
    let r = last_results(&fake, 1);
    let errs: Vec<bool> = r.iter().map(|x| x["is_error"] == true).collect();
    assert_eq!(errs, [false, false, true, false, false, false, false, false, false, false, true], "{r:?}");
    assert!(r[2]["content"].as_str().unwrap().contains("outside"));
    let text = accepted(&s);
    assert!(text.starts_with(P1) && text.ends_with(P3), "{text}");
    assert!(text.contains("dripping") && text.contains("Added.") && text.contains("loud"), "{text}");
}

#[test]
fn stop_while_paused_leaves_a_valid_transcript() {
    let mut s = session_paras(&[P1, P2, P3]);
    let mut tools = delete_p2();
    tools["content"].as_array_mut().unwrap().push(json!({"type": "tool_use", "id": "tu_3", "name": "get_text", "input": {}}));
    let fake = Fake::new(vec![Ok(tools), Ok(anthropic_text("ok"))]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.ask("delete it", &settings("claude"), &mut s, &*http).unwrap();
    agent.poll(&mut s, &*http);
    assert!(agent.pending().is_some());
    assert!(s.author.starts_with("AI ("));
    agent.stop(&mut s);
    assert!(!agent.busy() && agent.pending().is_none());
    assert_eq!(s.author, "WordCraft User");
    assert!(!s.doc.settings.track_changes);
    assert_eq!(accepted(&s), format!("{P1}\n{P2}\n{P3}"));
    agent.ask("never mind", &settings("claude"), &mut s, &*http).unwrap();
    let body = fake.body(1);
    let msgs = body["messages"].as_array().unwrap();
    let roles: Vec<&str> = msgs.iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["user", "assistant", "user"]);
    let kinds: Vec<&str> = msgs[2]["content"].as_array().unwrap().iter().map(|b| b["type"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["tool_result", "tool_result", "tool_result", "tool_result", "text"]);
    assert_eq!(msgs[2]["content"][2]["is_error"], true);
    assert_eq!(msgs[2]["content"][3]["is_error"], true);
    drive(&mut agent, &mut s, &*http);
}

#[test]
fn hostile_harness_args_never_panic() {
    let many: Vec<Value> = (0..500).map(|i| json!({"text": format!("s{i}"), "done": false})).collect();
    let junk = [
        ("plan", json!({})),
        ("plan", json!({"steps": "x"})),
        ("plan", json!({"steps": [1, null, {"text": 5}]})),
        ("plan", json!({"steps": [{"text": "", "done": "yes"}]})),
        ("plan", json!({"steps": many})),
        ("plan", json!({"steps": [{"text": "é".repeat(10_000), "done": true}]})),
        ("ask_user", json!({})),
        ("ask_user", json!({"question": 5})),
        ("ask_user", json!({"question": "   "})),
        ("ask_user", json!({"question": "q", "options": "x"})),
        ("ask_user", json!({"question": "q".repeat(50_000), "options": [1, "", "ü".repeat(9_000), null, "a", "b", "c", "d", "e", "f", "g"]})),
    ];
    for (name, args) in junk {
        let mut s = session("Text");
        let fake = Fake::new(vec![Ok(anthropic_tools(&[(name, args.clone())])), Ok(anthropic_text("end"))]);
        let http = fake.http();
        let mut agent = Agent::default();
        agent.ask("x", &settings("claude"), &mut s, &*http).unwrap();
        for _ in 0..10 {
            if let Some(Pending::Question { question, options }) = agent.pending() {
                assert!(question.chars().count() <= 1_001 && options.len() <= 6 && options.iter().all(|o| !o.is_empty()));
                agent.answer("", &mut s, &*http);
            }
            agent.poll(&mut s, &*http);
        }
        assert!(!agent.busy(), "{name} {args}");
        assert!(agent.plan().len() <= 30 && agent.plan().iter().all(|p| p.text.chars().count() <= 301));
    }
    // Answers and decisions with nothing waiting are ignored.
    let mut s = session("Text");
    let fake = Fake::new(vec![]);
    let http = fake.http();
    let mut agent = Agent::default();
    agent.answer("x", &mut s, &*http);
    agent.decide(Decision::Deny, &mut s, &*http);
    assert!(agent.log.is_empty() && fake.sent.borrow().is_empty());
}
