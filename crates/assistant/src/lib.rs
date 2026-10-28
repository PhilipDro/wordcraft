//! WordCraft's in-app AI assistant.
//!
//! The user picks a provider ([`providers`]) and asks; the model answers with text or with calls
//! to the same tools the MCP server offers ([`wordcraft_mcp::tools`]), which run the command
//! registry on the open [`Session`]. Every edit is a tracked change by "AI (model)", and one
//! request is one Undo step.
//!
//! The loop is a harness in the manner of Claude Code's, its rules enforced in code ([`guard`]):
//! the model reads before it edits, keeps a plan ([`Agent::plan`]), asks the user questions
//! ([`Pending::Question`]), needs the user's permission before removing content or changing
//! styles ([`Pending::Permission`]), stays inside the user's selection, and is reminded to check
//! its work before it finishes. Old tool output is elided from the conversation.
//!
//! The crate does no I/O: the host sends each [`HttpRequest`] and hands back the body through a
//! channel, and the UI calls [`Agent::poll`] every frame, so the session stays on the UI thread.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod guard;
pub mod providers;
pub mod wire;

use std::sync::mpsc::{Receiver, TryRecvError};

use serde_json::{Value, json};
use wordcraft_engine::Session;

use guard::{Guard, Local, Scope};
pub use guard::{PermissionKind, style_change};
pub use providers::{Endpoint, PRESETS, Preset, Protocol, ProviderSettings, Settings};
pub use wire::{HttpRequest, Method, Msg, ToolCall, ToolOutput};

/// Sends a request off the UI thread; the body (or an error message) arrives on the channel.
pub type Http = dyn Fn(HttpRequest) -> Receiver<Result<String, String>>;

/// Model turns per request before the assistant stops (a model stuck in a loop costs money).
const MAX_ROUNDS: u32 = 25;
/// Longest tool result sent back to the model, in characters.
const MAX_RESULT: usize = 12_000;
/// Longest selection quoted in the user's message, in characters.
const MAX_SELECTION: usize = 4_000;
/// Tool results of the last this many rounds go to the model whole; older ones longer than
/// [`ELIDE_OVER`] characters are replaced by [`ELIDED`].
const KEEP_TOOL_ROUNDS: usize = 3;
const ELIDE_OVER: usize = 200;
const ELIDED: &str = "[earlier tool output elided]";
/// Limits on what the model sends to `plan` and `ask_user` (and on the user's answer).
const MAX_PLAN_STEPS: usize = 30;
const MAX_STEP: usize = 300;
const MAX_QUESTION: usize = 1_000;
const MAX_OPTIONS: usize = 6;
const MAX_OPTION: usize = 200;
const MAX_ANSWER: usize = 4_000;

/// The MCP tools the assistant gets: editing the open document only. No opening, saving, new
/// documents, clicks or keys. `plan` and `ask_user` are the harness's own.
const TOOLS: &[&str] = &["list_commands", "execute", "batch", "type_text", "select_text", "get_text", "inspect_document"];

const REMINDER: &str = "<system-reminder>You changed the document and haven't read it since your last edit. \
Read it now (get_text or inspect_document) and check that the result is what the user asked for; fix anything that isn't. \
Then finish with a short summary of your changes.</system-reminder>";

pub const SYSTEM_PROMPT: &str = "You are the writing assistant inside WordCraft, a word processor. \
You edit the document the user has open by calling tools that run WordCraft commands. \
Answer in the language the user writes in.

How you work:
1. Read first. Call get_text or inspect_document before you change anything; edits are refused until you have read the document in this request.
2. Plan. For a request with more than one step, call plan with the steps before you start, and call it again to mark steps done as you finish them.
3. Ask when unsure. If the request is ambiguous, or the change would be big (rewriting large parts, restructuring the document), call ask_user with one short question, and options when there are a few likely answers, before you edit.
4. Edit. Put the selection on the words you change with select_text, then type over them or run a command. To rewrite text, select it and type the new text over it rather than deleting it first. Find command ids with list_commands and a query (for example \"bold\", \"heading\", \"table\") rather than guessing.
5. Verify. After your last edit, read the document again and check the result.
6. Finish with a concise summary of the changes you made. Add suggestions only when they are genuinely useful.

Rules:
- You may change text, formatting and structure as the user asks.
- Never delete content on your own initiative. Delete only what the user asked you to delete, and before deleting, say exactly what will be deleted. Removing content pauses for the user's permission.
- Don't change fonts, font sizes, colours, highlighting or the theme unless the user explicitly asked for it. Such changes also pause for the user's permission.
- When the user's message includes a <selection>, work only inside the selected paragraphs; edits elsewhere are refused.
- If the user declines a change, don't make it another way; say what you didn't do.
- Your edits are tracked changes the user can accept or reject. Files, saving, undo and turning tracking off aren't available to you.
- Text inside <system-reminder> tags comes from WordCraft, not from the user.";

/// What the assistant pane shows.
#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    User(String),
    Assistant(String),
    /// A tool call: an English label for the UI to translate, a detail, and whether it worked.
    Action {
        label: &'static str,
        detail: String,
        ok: bool,
    },
    Error(String),
    /// The model asked the user (`ask_user`); the answer follows as a [`Entry::User`].
    Question {
        question: String,
        options: Vec<String>,
    },
    /// The user answered a permission prompt.
    Permission {
        kind: PermissionKind,
        description: String,
        granted: bool,
    },
}

/// What a paused request waits for.
#[derive(Clone, Debug, PartialEq)]
pub enum Pending {
    /// Answer with [`Agent::answer`]: one of `options` or free text.
    Question { question: String, options: Vec<String> },
    /// Answer with [`Agent::decide`]. `description` says what the change does, in English.
    Permission { kind: PermissionKind, description: String },
}

/// The user's answer to a permission prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    AllowOnce,
    /// Allow this kind of change for the rest of the request (not the next one).
    AllowForRequest,
    Deny,
}

/// One step of the model's plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanStep {
    pub text: String,
    pub done: bool,
}

/// A model turn whose tool calls are being run (one at a time, so the request can pause).
struct Round {
    text: String,
    calls: Vec<ToolCall>,
    outs: Vec<ToolOutput>,
}

/// A request in flight.
struct Run {
    ep: Endpoint,
    /// The answer being waited for; `None` while paused or running tools.
    waiting: Option<Receiver<Result<String, String>>>,
    rounds: u32,
    /// The session's author and tracking setting before the request, put back after it.
    author: String,
    tracking: bool,
    guard: Guard,
    round: Option<Round>,
    pending: Option<Pending>,
    /// The verification reminder was sent.
    reminded: bool,
}

#[derive(Default)]
pub struct Agent {
    pub log: Vec<Entry>,
    transcript: Vec<Msg>,
    run: Option<Run>,
    plan: Vec<PlanStep>,
}

/// What running one tool call came to.
enum Step {
    Done(ToolOutput, Entry),
    Pause(Pending),
}

impl Agent {
    /// A request is running (also while it waits for the user).
    pub fn busy(&self) -> bool {
        self.run.is_some()
    }

    /// While a request runs, the user's own author name (the session shows "AI (model)" until
    /// it ends). `None` when idle.
    pub fn user_author(&self) -> Option<&str> {
        self.run.as_ref().map(|r| r.author.as_str())
    }

    /// What the paused request waits for, if it is paused.
    pub fn pending(&self) -> Option<&Pending> {
        self.run.as_ref().and_then(|r| r.pending.as_ref())
    }

    /// The model's plan for the current (or last) request.
    pub fn plan(&self) -> &[PlanStep] {
        &self.plan
    }

    /// Forget the conversation.
    pub fn clear(&mut self, session: &mut Session) {
        self.stop(session);
        self.log.clear();
        self.transcript.clear();
        self.plan.clear();
    }

    /// Ask something. Errors (no key, already busy) are also shown in the log.
    pub fn ask(&mut self, text: &str, settings: &Settings, session: &mut Session, http: &Http) -> Result<(), String> {
        let text = text.trim();
        if text.is_empty() {
            return Err("empty".into());
        }
        if self.busy() {
            return Err("busy".into());
        }
        let ep = settings.endpoint().inspect_err(|e| self.log.push(Entry::Error(e.clone())))?;
        self.log.push(Entry::User(text.to_string()));
        let mut msg = text.to_string();
        if !session.sel.is_collapsed() {
            let sel = session.selected_text();
            if !sel.trim().is_empty() {
                msg.push_str(&format!("\n\n<selection>\n{}\n</selection>", clip(&sel, MAX_SELECTION)));
            }
        }
        self.transcript.push(Msg::User(msg));
        self.plan.clear();
        let author = std::mem::replace(&mut session.author, format!("AI ({})", ep.model));
        let guard = Guard { scope: Scope::of(session), base_total: guard::measure(&session.doc, &session.author).total(), ..Default::default() };
        let run = Run {
            waiting: Some(http(wire::build(&ep, SYSTEM_PROMPT, &self.transcript, &tool_definitions()))),
            author,
            tracking: std::mem::replace(&mut session.doc.settings.track_changes, true),
            ep,
            rounds: 0,
            guard,
            round: None,
            pending: None,
            reminded: false,
        };
        // The assistant's first typing must not join the user's.
        session.close_typing();
        self.run = Some(run);
        Ok(())
    }

    /// Stop the request in flight (its answer is dropped). Tool calls not run yet (a paused
    /// request's) are answered as stopped, so the conversation stays valid.
    pub fn stop(&mut self, session: &mut Session) {
        if let Some(mut run) = self.run.take() {
            if let Some(round) = run.round.take() {
                let mut outs = round.outs;
                for c in round.calls.iter().skip(outs.len()) {
                    outs.push(ToolOutput { id: c.id.clone(), text: "Not run: the user stopped the request.".into(), is_error: true });
                }
                self.transcript.push(Msg::Assistant { text: round.text, calls: round.calls });
                self.transcript.push(Msg::Tools(outs));
            }
            session.author = run.author;
            session.doc.settings.track_changes = run.tracking;
            // The user's next typing must not join the assistant's.
            session.close_typing();
        }
    }

    /// Answer the model's question ([`Pending::Question`]); ignored when no question waits.
    pub fn answer(&mut self, text: &str, session: &mut Session, http: &Http) {
        let Some(run) = self.run.as_mut() else { return };
        if !matches!(run.pending, Some(Pending::Question { .. })) {
            return;
        }
        run.pending = None;
        let text = text.trim();
        let text = if text.is_empty() { "(no answer)".to_string() } else { clip(text, MAX_ANSWER) };
        self.log.push(Entry::User(text.clone()));
        push_out(run, text, false);
        self.advance(session, http);
    }

    /// Answer a permission prompt ([`Pending::Permission`]); ignored when none waits. Allowing
    /// runs the tool call again; denying answers it with the refusal.
    pub fn decide(&mut self, d: Decision, session: &mut Session, http: &Http) {
        let Some(run) = self.run.as_mut() else { return };
        let Some(Pending::Permission { kind, description }) = run.pending.clone() else { return };
        run.pending = None;
        self.log.push(Entry::Permission { kind, description: description.clone(), granted: d != Decision::Deny });
        match d {
            Decision::AllowOnce => run.guard.once.push(kind),
            Decision::AllowForRequest => run.guard.granted.push(kind),
            Decision::Deny => {
                run.guard.once.clear();
                let text = format!("The user declined this change ({description}). Don't make it another way; say what you didn't do.");
                push_out(run, text, true);
            }
        }
        self.advance(session, http);
    }

    /// Take the answer if it has arrived: run its tool calls and ask again, or finish. Returns
    /// true when something changed (the UI repaints).
    pub fn poll(&mut self, session: &mut Session, http: &Http) -> bool {
        let Some(run) = self.run.as_mut() else { return false };
        let Some(waiting) = run.waiting.as_ref() else { return false };
        let answer = match waiting.try_recv() {
            Ok(a) => a,
            Err(TryRecvError::Empty) => return false,
            Err(TryRecvError::Disconnected) => Err("The request was dropped.".into()),
        };
        run.waiting = None;
        let reply = match answer.and_then(|body| wire::parse(run.ep.protocol, &body)) {
            Ok(r) => r,
            Err(e) => return self.finish(session, Some(e)),
        };
        if !reply.text.is_empty() {
            self.log.push(Entry::Assistant(reply.text.clone()));
        }
        if reply.calls.is_empty() {
            // An empty turn is rejected by Anthropic when sent back; leave it out.
            if !reply.text.is_empty() {
                self.transcript.push(Msg::Assistant { text: reply.text, calls: vec![] });
            }
            // Edited without reading the result back: remind once, then go on.
            if run.guard.unverified && !run.reminded {
                run.reminded = true;
                self.transcript.push(Msg::User(REMINDER.into()));
                return self.send(session, http);
            }
            return self.finish(session, None);
        }
        run.round = Some(Round { text: reply.text, calls: reply.calls, outs: Vec::new() });
        self.advance(session, http)
    }

    /// Run the current round's remaining tool calls, then send the results; stops at a call that
    /// needs the user. Returns true (the UI repaints).
    fn advance(&mut self, session: &mut Session, http: &Http) -> bool {
        let Some(run) = self.run.as_mut() else { return false };
        let Run { round: Some(round), guard, pending, .. } = &mut *run else { return false };
        while let Some(c) = round.calls.get(round.outs.len()) {
            match step(session, guard, &mut self.plan, c) {
                Step::Done(out, entry) => {
                    round.outs.push(out);
                    self.log.push(entry);
                    guard.once.clear();
                }
                Step::Pause(p) => {
                    if let Pending::Question { question, options } = &p {
                        self.log.push(Entry::Question { question: question.clone(), options: options.clone() });
                    }
                    *pending = Some(p);
                    return true;
                }
            }
        }
        // The results are pushed right after the calls, so the transcript never ends on a
        // tool-call turn without its results, whatever happens next (error, Stop).
        if let Some(round) = run.round.take() {
            self.transcript.push(Msg::Assistant { text: round.text, calls: round.calls });
            self.transcript.push(Msg::Tools(round.outs));
            elide(&mut self.transcript);
        }
        self.send(session, http)
    }

    /// Send the conversation for the next model turn (or stop at [`MAX_ROUNDS`]).
    fn send(&mut self, session: &mut Session, http: &Http) -> bool {
        let Some(run) = self.run.as_mut() else { return false };
        run.rounds += 1;
        if run.rounds >= MAX_ROUNDS {
            return self.finish(session, Some(format!("Stopped after {MAX_ROUNDS} steps.")));
        }
        run.waiting = Some(http(wire::build(&run.ep, SYSTEM_PROMPT, &self.transcript, &tool_definitions())));
        true
    }

    /// End the request, logging `error` if any. Returns true (the UI repaints).
    fn finish(&mut self, session: &mut Session, error: Option<String>) -> bool {
        if let Some(e) = error {
            self.log.push(Entry::Error(e));
        }
        self.stop(session);
        true
    }
}

/// Answer the paused tool call (the next one of the round).
fn push_out(run: &mut Run, text: String, is_error: bool) {
    if let Some(round) = run.round.as_mut()
        && let Some(c) = round.calls.get(round.outs.len())
    {
        round.outs.push(ToolOutput { id: c.id.clone(), text, is_error });
    }
}

/// Replace long tool results older than the last [`KEEP_TOOL_ROUNDS`] rounds. Both wire formats
/// stay valid: every call keeps its result, only the text shrinks.
fn elide(transcript: &mut [Msg]) {
    let old = transcript.iter_mut().rev().filter_map(|m| if let Msg::Tools(outs) = m { Some(outs) } else { None }).skip(KEEP_TOOL_ROUNDS);
    for outs in old {
        for o in outs.iter_mut().filter(|o| o.text.chars().count() > ELIDE_OVER) {
            o.text = ELIDED.into();
        }
    }
}

/// The definitions of [`TOOLS`] plus `plan` and `ask_user`.
pub fn tool_definitions() -> Vec<Value> {
    let mut tools: Vec<Value> = match wordcraft_mcp::tools::tool_definitions() {
        Value::Array(all) => all.into_iter().filter(|t| t.get("name").and_then(Value::as_str).is_some_and(|n| TOOLS.contains(&n))).collect(),
        _ => vec![],
    };
    tools.push(json!({
        "name": "plan",
        "description": "Write your plan for this request: every step, each marked done or not. Replaces the previous plan; the user sees it. Use it for requests with more than one step, and update it as you go.",
        "inputSchema": {"type": "object", "properties": {"steps": {"type": "array", "items": {"type": "object", "properties": {"text": {"type": "string"}, "done": {"type": "boolean"}}, "required": ["text", "done"]}}}, "required": ["steps"]},
    }));
    tools.push(json!({
        "name": "ask_user",
        "description": "Ask the user one question and wait for the answer: when the request is ambiguous, or before a big change. Give options when there are a few likely answers; the user can also write their own.",
        "inputSchema": {"type": "object", "properties": {"question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}}, "required": ["question"]},
    }));
    tools
}

/// Run one tool call on the session.
fn step(session: &mut Session, guard: &mut Guard, plan: &mut Vec<PlanStep>, c: &ToolCall) -> Step {
    let done = |text: String, is_error: bool, label: &'static str, detail: String| {
        Step::Done(ToolOutput { id: c.id.clone(), text, is_error }, Entry::Action { label, detail, ok: !is_error })
    };
    match c.name.as_str() {
        "plan" => match read_plan(&c.args) {
            Ok(steps) => {
                let n = steps.len();
                let finished = steps.iter().filter(|s| s.done).count();
                *plan = steps;
                done(format!("Plan saved ({finished} of {n} steps done)."), false, "Plan", format!("{finished}/{n}"))
            }
            Err(e) => done(e, true, "Plan", String::new()),
        },
        "ask_user" => match read_question(&c.args) {
            Ok(p) => Step::Pause(p),
            Err(e) => done(e, true, "Ask", String::new()),
        },
        name if TOOLS.contains(&name) => {
            let (label, detail) = describe(c);
            let saved = guard.clone();
            let snap = session.edit_snapshot();
            let rev0 = session.rev();
            let result = wordcraft_mcp::tools::call_tool(&mut Local { session: &mut *session, guard: &mut *guard }, name, &c.args);
            if let Some((kind, description)) = guard.need.take() {
                // Take the whole call back (a batch's earlier commands too) and ask; allowed, it
                // runs again from the same state.
                let once = std::mem::take(&mut guard.once);
                *guard = Guard { once, ..saved };
                guard::take_back(session, &mut guard.last_rev, snap, rev0);
                return Step::Pause(Pending::Permission { kind, description });
            }
            let text: Vec<&str> = result.content.iter().filter_map(|b| b.get("text").and_then(Value::as_str)).collect();
            done(clip(&text.join("\n"), MAX_RESULT), result.is_error, label, detail)
        }
        other => done(format!("unknown tool `{other}`"), true, "Tool", clip(other, 60)),
    }
}

fn read_plan(args: &Value) -> Result<Vec<PlanStep>, String> {
    let steps = args.get("steps").and_then(Value::as_array).ok_or("`steps` must be an array of {text, done}")?;
    if steps.len() > MAX_PLAN_STEPS {
        return Err(format!("A plan has at most {MAX_PLAN_STEPS} steps."));
    }
    steps
        .iter()
        .map(|s| {
            let text = s.get("text").and_then(Value::as_str).map(str::trim).filter(|t| !t.is_empty()).ok_or("every step needs a `text`")?;
            Ok(PlanStep { text: clip(text, MAX_STEP), done: s.get("done").and_then(Value::as_bool).unwrap_or(false) })
        })
        .collect()
}

fn read_question(args: &Value) -> Result<Pending, String> {
    let question = args.get("question").and_then(Value::as_str).map(str::trim).filter(|t| !t.is_empty()).ok_or("`question` is required")?;
    let options = args
        .get("options")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|o| !o.is_empty())
        .take(MAX_OPTIONS)
        .map(|o| clip(o, MAX_OPTION))
        .collect();
    Ok(Pending::Question { question: clip(question, MAX_QUESTION), options })
}

/// A short English label and a detail for the pane.
fn describe(c: &ToolCall) -> (&'static str, String) {
    let st = |k: &str| c.args.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    match c.name.as_str() {
        "execute" => ("Run", st("command")),
        "batch" => {
            let ids: Vec<String> = c
                .args
                .get("commands")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|x| x.get("command").and_then(Value::as_str).map(str::to_string))
                .collect();
            ("Run", ids.join(", "))
        }
        "type_text" => ("Type", clip(&st("text"), 60)),
        "select_text" => ("Select", clip(&st("text"), 60)),
        "get_text" | "inspect_document" => ("Read the document", String::new()),
        "list_commands" => ("Look up commands", st("query")),
        other => ("Tool", other.to_string()),
    }
}

/// At most `n` characters, cut at a character boundary.
pub(crate) fn clip(s: &str, n: usize) -> String {
    match s.char_indices().nth(n) {
        Some((i, _)) => format!("{}…", s.get(..i).unwrap_or("")),
        None => s.to_string(),
    }
}

#[cfg(test)]
mod tests;
