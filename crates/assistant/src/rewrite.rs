//! Prompt on selection: one request, no tools. The model gets some of the document as context,
//! the selection and the user's instruction, and answers with the replacement text only, which
//! is typed over the selection as one tracked change by "AI (model)" and one undo step.
//!
//! Pure parts (scopes, clipping, the request, cleaning the answer) are unit tested here; the host
//! sends the request through the same [`Http`] hook as the assistant.

use std::sync::mpsc::{Receiver, RecvTimeoutError, TryRecvError};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use wordcraft_engine::doc::{Block, StoryRef};
use wordcraft_engine::{Selection, Session};

use crate::providers::Settings;
use crate::wire::{self, HttpRequest, Msg};
use crate::{Http, Protocol};

/// How much of the document goes with the selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Scope {
    /// The whole document (clipped to the cap around the selection).
    #[default]
    Whole,
    /// The section between the nearest headings around the selection.
    Section,
    /// The selected paragraphs and `scope_paragraphs` on each side.
    Paragraphs,
    /// No context: the selection only.
    SelectionOnly,
}

impl Scope {
    pub const ALL: [Scope; 4] = [Scope::Whole, Scope::Section, Scope::Paragraphs, Scope::SelectionOnly];

    pub fn id(self) -> &'static str {
        match self {
            Scope::Whole => "whole",
            Scope::Section => "section",
            Scope::Paragraphs => "paragraphs",
            Scope::SelectionOnly => "selectionOnly",
        }
    }

    pub fn parse(s: &str) -> Option<Scope> {
        Scope::ALL.into_iter().find(|x| x.id().eq_ignore_ascii_case(s))
    }

    /// The German label the settings and the popover show.
    pub fn label(self) -> &'static str {
        match self {
            Scope::Whole => "Ganzes Dokument",
            Scope::Section => "Abschnitt",
            Scope::Paragraphs => "Absätze ringsum",
            Scope::SelectionOnly => "Nur Auswahl",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct RewritePreset {
    pub name: String,
    pub prompt: String,
}

pub const DEFAULT_SYSTEM_PROMPT: &str = "Du bist ein Lektoratswerkzeug in einem Textverarbeitungsprogramm. \
Du bekommst einen Ausschnitt des Dokuments als Kontext (<document>), die markierte Auswahl (<selection>) und eine Anweisung. \
Wende die Anweisung nur auf die Auswahl an und gib AUSSCHLIESSLICH den Ersatztext für die Auswahl zurück: \
keine Einleitung, keine Erklärung, keine Anführungszeichen, kein Markdown, keine Codeblöcke. \
Behalte die Sprache der Auswahl bei, außer die Anweisung verlangt ausdrücklich eine andere. \
Trenne Absätze mit einem Zeilenumbruch. Der Kontext dient nur dem Verständnis: gib ihn nie wieder.";

/// The prompt-on-selection settings, stored in `assistant.json` beside the provider settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct RewriteSettings {
    pub scope: Scope,
    pub scope_paragraphs: usize,
    pub max_context_chars: usize,
    pub system_prompt: String,
    pub presets: Vec<RewritePreset>,
}

impl Default for RewriteSettings {
    fn default() -> Self {
        let p = |name: &str, prompt: &str| RewritePreset { name: name.into(), prompt: prompt.into() };
        RewriteSettings {
            scope: Scope::Whole,
            scope_paragraphs: 3,
            max_context_chars: 60_000,
            system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
            presets: vec![
                p("Schärfen", "Schärfe den Text: präziser, klarer und aktiver formuliert, ohne den Inhalt zu ändern."),
                p("Kürzen", "Kürze den Text deutlich, ohne wesentliche Information zu verlieren."),
                p("Formeller", "Formuliere den Text formeller und professioneller, bei gleichem Inhalt."),
                p("Englisch → Deutsch", "Übersetze den Text ins Deutsche. Gib nur die Übersetzung zurück."),
            ],
        }
    }
}

impl RewriteSettings {
    /// A preset's name stands for its prompt (`"Kürzen"` → the Kürzen preset's text); anything
    /// else is the prompt itself.
    pub fn resolve_prompt(&self, prompt: &str) -> String {
        let p = prompt.trim();
        self.presets.iter().find(|x| x.name.trim().eq_ignore_ascii_case(p)).map_or_else(|| p.to_string(), |x| x.prompt.clone())
    }

    fn system(&self) -> &str {
        if self.system_prompt.trim().is_empty() { DEFAULT_SYSTEM_PROMPT } else { &self.system_prompt }
    }
}

/// One top-level block of the body as the scopes see it.
#[derive(Clone, Debug, PartialEq)]
pub struct Para {
    pub text: String,
    pub heading: bool,
}

/// The body's top-level blocks: paragraphs (tracked deletions left out) and tables (cells by tab,
/// rows by line).
pub fn outline(s: &Session) -> Vec<Para> {
    s.doc
        .body
        .iter()
        .map(|b| match &**b {
            Block::Para(p) => Para { text: p.final_text(), heading: s.doc.styles.resolve_para(&p.props).outline_level.is_some_and(|l| l < 9) },
            Block::Table(t) => Para {
                text: t
                    .rows
                    .iter()
                    .map(|r| {
                        r.cells
                            .iter()
                            .map(|c| c.blocks.iter().filter_map(|b| b.as_para()).map(|p| p.final_text()).collect::<Vec<_>>().join(" "))
                            .collect::<Vec<_>>()
                            .join("\t")
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                heading: false,
            },
        })
        .collect()
}

/// The blocks a scope takes, given the selected blocks `first..=last`.
pub fn scope_range(paras: &[Para], first: usize, last: usize, scope: Scope, n: usize) -> (usize, usize) {
    let end = paras.len().saturating_sub(1);
    let (first, last) = (first.min(end), last.min(end).max(first.min(end)));
    match scope {
        Scope::Whole => (0, end),
        Scope::SelectionOnly => (first, last),
        Scope::Paragraphs => (first.saturating_sub(n), last.saturating_add(n).min(end)),
        Scope::Section => {
            let lo = (0..=first).rev().find(|&i| paras.get(i).is_some_and(|p| p.heading)).unwrap_or(0);
            let hi = (last + 1..=end).find(|&i| paras.get(i).is_some_and(|p| p.heading)).map_or(end, |i| i.saturating_sub(1).max(last));
            (lo, hi)
        }
    }
}

/// The text of blocks `lo..=hi`, clipped to `cap` characters around blocks `first..=last` (the
/// selection), not from the start. Clipped ends are marked with "…".
pub fn context_text(paras: &[Para], lo: usize, hi: usize, first: usize, last: usize, cap: usize) -> String {
    let mut text = String::new();
    let (mut sel_a, mut sel_b) = (0usize, 0usize);
    for i in lo..=hi.min(paras.len().saturating_sub(1)) {
        if i > lo {
            text.push('\n');
        }
        if i == first {
            sel_a = text.chars().count();
        }
        text.push_str(paras.get(i).map_or("", |p| p.text.as_str()));
        if i == last {
            sel_b = text.chars().count();
        }
    }
    if sel_b < sel_a {
        sel_b = sel_a;
    }
    clip_around(&text, sel_a, sel_b, cap)
}

/// `text` cut to `cap` characters, keeping characters `a..b` and as much on each side as fits,
/// split evenly (what one side doesn't use goes to the other).
pub fn clip_around(text: &str, a: usize, b: usize, cap: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();
    if total <= cap {
        return text.to_string();
    }
    let (a, b) = (a.min(total), b.min(total).max(a.min(total)));
    let budget = cap.saturating_sub(b - a);
    let mut before = a.min(budget / 2);
    let after = (total - b).min(budget - before);
    before = a.min(budget - after);
    let (start, end) = (a - before, (b + after).min(total));
    let (start, end) = if b - a > cap { (a, a + cap) } else { (start, end) };
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.extend(chars.get(start..end).unwrap_or_default());
    if end < total {
        out.push('…');
    }
    out
}

/// The user message: the context, the selection, the instruction.
pub fn user_message(scope: Scope, context: &str, selection: &str, prompt: &str) -> String {
    let mut m = String::new();
    if scope != Scope::SelectionOnly && !context.trim().is_empty() {
        m.push_str(&format!("<document scope=\"{}\">\n{}\n</document>\n\n", scope.id(), context));
    }
    m.push_str(&format!("<selection>\n{selection}\n</selection>\n\n{prompt}"));
    m
}

/// One request, no tools. An empty API key sends no authorisation header (local servers).
pub fn build_request(settings: &Settings, system: &str, user: &str) -> Result<(HttpRequest, String, Protocol), String> {
    let ep = settings.endpoint()?;
    let mut req = wire::build(&ep, system, &[Msg::User(user.to_string())], &[]);
    if let Ok(mut body) = serde_json::from_str::<Value>(&req.body) {
        if let Some(o) = body.as_object_mut() {
            o.remove("tools");
        }
        req.body = body.to_string();
    }
    if ep.api_key.is_empty() {
        req.headers.retain(|(k, _)| k != "authorization" && k != "x-api-key");
    }
    Ok((req, ep.model, ep.protocol))
}

/// The model's answer as replacement text: no reasoning block, code fence or wrapping quotes.
pub fn clean_answer(s: &str) -> String {
    let mut t = s.trim().to_string();
    // Reasoning models may put their thinking in the content.
    while let (Some(a), Some(b)) = (t.find("<think>"), t.find("</think>")) {
        if b < a {
            break;
        }
        t.replace_range(a..b + "</think>".len(), "");
        t = t.trim().to_string();
    }
    if t.starts_with("```") && t.ends_with("```") && t.len() >= 6 {
        let inner = &t[3..t.len() - 3];
        // The fence's language tag runs to the first line break.
        let inner = match inner.find('\n') {
            Some(i) if !inner[..i].contains(' ') => &inner[i + 1..],
            _ => inner,
        };
        t = inner.trim().to_string();
    }
    for (open, close) in [("\"", "\""), ("„", "“"), ("“", "”"), ("»", "«"), ("«", "»"), ("'", "'"), ("‚", "‘")] {
        if t.len() > open.len() + close.len() && t.starts_with(open) && t.ends_with(close) {
            let inner = &t[open.len()..t.len() - close.len()];
            // Only quotes around the whole answer: none of the same kind inside.
            if !inner.contains(open) && !inner.contains(close) {
                t = inner.trim().to_string();
                break;
            }
        }
    }
    t
}

/// What the selection was when the request went out: an answer only applies if it is unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub document: u64,
    pub rev: u64,
    pub sel: Selection,
}

impl Snapshot {
    pub fn of(s: &Session) -> Self {
        Snapshot { document: s.document_id(), rev: s.rev(), sel: s.sel.clone() }
    }
    pub fn holds(&self, s: &Session) -> bool {
        *self == Snapshot::of(s)
    }
}

/// A request in flight.
pub struct Job {
    rx: Receiver<Result<String, String>>,
    pub model: String,
    pub protocol: Protocol,
    pub snapshot: Snapshot,
    /// The selection ended with a paragraph mark: the replacement keeps one.
    pub ends_paragraph: bool,
    pub started_ms: f64,
    pub scope: Scope,
}

pub enum Outcome {
    Waiting,
    /// The cleaned replacement text.
    Answer(String),
    Failed(String),
}

/// Start a request for the current selection. `prompt` may be a preset's name.
pub fn start(settings: &Settings, scope: Scope, prompt: &str, session: &Session, http: &Http, now_ms: f64) -> Result<Job, String> {
    let rs = &settings.rewrite;
    let prompt = rs.resolve_prompt(prompt);
    if prompt.is_empty() {
        return Err("Keine Anweisung.".into());
    }
    if session.sel.is_collapsed() {
        return Err("Nichts ausgewählt.".into());
    }
    let selection = session.selected_text();
    if selection.trim().is_empty() {
        return Err("Die Auswahl enthält keinen Text.".into());
    }
    let context = scope_context(session, scope, rs.scope_paragraphs, rs.max_context_chars);
    let user = user_message(scope, &context, selection.trim_end_matches('\n'), &prompt);
    let (req, model, protocol) = build_request(settings, rs.system(), &user)?;
    Ok(Job {
        rx: http(req),
        model,
        protocol,
        snapshot: Snapshot::of(session),
        ends_paragraph: selection.ends_with('\n'),
        started_ms: now_ms,
        scope,
    })
}

/// The context text for `scope` around the session's selection (body only; elsewhere, none).
pub fn scope_context(session: &Session, scope: Scope, n: usize, cap: usize) -> String {
    let (a, b) = session.sel.ordered();
    if scope == Scope::SelectionOnly || a.story != StoryRef::Body || b.story != StoryRef::Body {
        return String::new();
    }
    let paras = outline(session);
    let first = a.path.0.first().copied().unwrap_or(0) as usize;
    let last = b.path.0.first().copied().unwrap_or(0) as usize;
    let (lo, hi) = scope_range(&paras, first, last, scope, n);
    context_text(&paras, lo, hi, first, last, cap)
}

impl Job {
    fn take(&self, got: Result<String, String>) -> Outcome {
        match got.and_then(|body| wire::parse(self.protocol, &body)) {
            Ok(reply) => {
                let mut text = clean_answer(&reply.text);
                if text.is_empty() {
                    return Outcome::Failed("Das Modell hat keinen Text geliefert.".into());
                }
                if self.ends_paragraph && !text.ends_with('\n') {
                    text.push('\n');
                }
                Outcome::Answer(text)
            }
            Err(e) => Outcome::Failed(format!("Modell nicht erreichbar: {e}")),
        }
    }

    pub fn poll(&self) -> Outcome {
        match self.rx.try_recv() {
            Ok(got) => self.take(got),
            Err(TryRecvError::Empty) => Outcome::Waiting,
            Err(TryRecvError::Disconnected) => Outcome::Failed("Modell nicht erreichbar: die Anfrage brach ab.".into()),
        }
    }

    /// Block until the answer arrives (headless callers).
    pub fn wait(&self, timeout: std::time::Duration) -> Outcome {
        match self.rx.recv_timeout(timeout) {
            Ok(got) => self.take(got),
            Err(RecvTimeoutError::Timeout) => Outcome::Failed("Modell nicht erreichbar: Zeitüberschreitung.".into()),
            Err(RecvTimeoutError::Disconnected) => Outcome::Failed("Modell nicht erreichbar: die Anfrage brach ab.".into()),
        }
    }
}

/// Type `text` over the selection as the assistant does: author "AI (model)", tracked, one undo
/// step; the user's author and tracking setting come back afterwards. Returns the characters typed.
pub fn apply(session: &mut Session, text: &str, model: &str) -> Result<usize, String> {
    let author = std::mem::replace(&mut session.author, format!("AI ({model})"));
    let tracking = std::mem::replace(&mut session.doc.settings.track_changes, true);
    // The replacement's typing must not join the user's, nor the user's next typing join it.
    session.close_typing();
    let r = session.run("text.insert", &json!({"text": text, "raw": true, "paragraphs": true}));
    session.close_typing();
    session.author = author;
    session.doc.settings.track_changes = tracking;
    r.map(|_| text.chars().count()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wordcraft_engine::doc::{Document, Path, Pos};

    fn p(text: &str, heading: bool) -> Para {
        Para { text: text.into(), heading }
    }

    /// H1 | a b | H2 | c d e | H3 | f
    fn fixture() -> Vec<Para> {
        vec![p("Eins", true), p("a", false), p("b", false), p("Zwei", true), p("c", false), p("d", false), p("e", false), p("Drei", true), p("f", false)]
    }

    #[test]
    fn whole_takes_everything() {
        assert_eq!(scope_range(&fixture(), 5, 5, Scope::Whole, 3), (0, 8));
    }

    #[test]
    fn section_is_bounded_by_the_nearest_headings() {
        let f = fixture();
        assert_eq!(scope_range(&f, 5, 5, Scope::Section, 3), (3, 6));
        assert_eq!(scope_range(&f, 1, 1, Scope::Section, 3), (0, 2));
        assert_eq!(scope_range(&f, 8, 8, Scope::Section, 3), (7, 8));
        // A selection across a heading takes both sections.
        assert_eq!(scope_range(&f, 2, 4, Scope::Section, 3), (0, 6));
        // No heading before: from the start.
        let plain = vec![p("x", false), p("y", false), p("H", true)];
        assert_eq!(scope_range(&plain, 1, 1, Scope::Section, 3), (0, 1));
    }

    #[test]
    fn paragraphs_take_n_on_each_side_within_the_document() {
        let f = fixture();
        assert_eq!(scope_range(&f, 5, 5, Scope::Paragraphs, 1), (4, 6));
        assert_eq!(scope_range(&f, 1, 2, Scope::Paragraphs, 3), (0, 5));
        assert_eq!(scope_range(&f, 8, 8, Scope::Paragraphs, 3), (5, 8));
    }

    #[test]
    fn selection_only_is_the_selected_blocks() {
        assert_eq!(scope_range(&fixture(), 4, 6, Scope::SelectionOnly, 3), (4, 6));
    }

    #[test]
    fn context_text_joins_the_scope() {
        let f = fixture();
        assert_eq!(context_text(&f, 3, 6, 5, 5, 1000), "Zwei\nc\nd\ne");
    }

    #[test]
    fn clipping_keeps_the_selection_and_splits_the_rest() {
        // 100 x's, the selection "SEL", 100 y's: cap 23 keeps SEL and 10 on each side.
        let text = format!("{}SEL{}", "x".repeat(100), "y".repeat(100));
        let out = clip_around(&text, 100, 103, 23);
        assert_eq!(out, format!("…{}SEL{}…", "x".repeat(10), "y".repeat(10)));
        // Near the start, what the start doesn't use goes after.
        let text = format!("xxSEL{}", "y".repeat(100));
        assert_eq!(clip_around(&text, 2, 5, 13), format!("xxSEL{}…", "y".repeat(8)));
        // Under the cap: untouched.
        assert_eq!(clip_around("abc", 1, 2, 10), "abc");
        // A selection longer than the cap: its start.
        assert_eq!(clip_around("aaSELECTIONbb", 2, 11, 4), "…SELE…");
        // The selection deep inside a long document is never clipped away.
        let paras: Vec<Para> = (0..2000).map(|i| p(&format!("Absatz {i:04} mit etwas Text."), false)).collect();
        let ctx = context_text(&paras, 0, 1999, 1500, 1500, 600);
        assert!(ctx.contains("Absatz 1500"), "{ctx}");
        assert!(!ctx.contains("Absatz 0000"));
        assert!(ctx.chars().count() <= 602);
    }

    #[test]
    fn clean_answer_strips_fences_quotes_and_thinking() {
        assert_eq!(clean_answer("  Hallo Welt \n"), "Hallo Welt");
        assert_eq!(clean_answer("```\nHallo\n```"), "Hallo");
        assert_eq!(clean_answer("```text\nZeile 1\nZeile 2\n```"), "Zeile 1\nZeile 2");
        assert_eq!(clean_answer("\"Hallo Welt\""), "Hallo Welt");
        assert_eq!(clean_answer("„Hallo Welt“"), "Hallo Welt");
        assert_eq!(clean_answer("»Hallo«"), "Hallo");
        assert_eq!(clean_answer("<think>hmm</think>\nKurz."), "Kurz.");
        // Quotes that aren't around the whole answer stay.
        assert_eq!(clean_answer("\"A\" und \"B\""), "\"A\" und \"B\"");
    }

    #[test]
    fn presets_resolve_by_name_and_old_settings_load() {
        let r = RewriteSettings::default();
        assert_eq!(r.presets.len(), 4);
        assert!(r.resolve_prompt("kürzen").starts_with("Kürze"));
        assert_eq!(r.resolve_prompt("Mach es lustig"), "Mach es lustig");
        let s: Settings = serde_json::from_str(r#"{"provider":"claude"}"#).unwrap_or_default();
        assert_eq!(s.rewrite, RewriteSettings::default());
        let s: Settings = serde_json::from_str(r#"{"rewrite":{"scope":"section","scopeParagraphs":5}}"#).unwrap_or_default();
        assert_eq!((s.rewrite.scope, s.rewrite.scope_paragraphs, s.rewrite.max_context_chars), (Scope::Section, 5, 60_000));
    }

    fn strata() -> Settings {
        let mut s = Settings { provider: crate::providers::CUSTOM.into(), ..Default::default() };
        let c = s.current_mut();
        c.base_url = "http://127.0.0.1:8080/v1".into();
        c.model = "qwen3.8-flash-next".into();
        s
    }

    #[test]
    fn the_request_has_no_tools_and_no_empty_key() {
        let s = strata();
        let user = user_message(Scope::Section, "Kontext", "Auswahl", "Kürzen");
        assert!(user.starts_with("<document scope=\"section\">\nKontext\n</document>"));
        assert!(user.contains("<selection>\nAuswahl\n</selection>\n\nKürzen"));
        let (req, model, protocol) = build_request(&s, "SYS", &user).unwrap();
        assert_eq!((model.as_str(), protocol), ("qwen3.8-flash-next", Protocol::OpenAi));
        assert_eq!(req.url, "http://127.0.0.1:8080/v1/chat/completions");
        let body: Value = serde_json::from_str(&req.body).unwrap();
        assert!(body.get("tools").is_none());
        assert_eq!(body["messages"][0]["content"], "SYS");
        assert_eq!(body["messages"][1]["content"], user);
        assert!(req.headers.iter().all(|(k, _)| k != "authorization"));
        assert!(!user_message(Scope::SelectionOnly, "Kontext", "A", "P").contains("<document"));
    }

    fn session(text: &str) -> Session {
        Session::new(Document::from_text(text))
    }

    fn select(s: &mut Session, block: usize) {
        let len = s.doc.para(StoryRef::Body, &Path::top(block)).map_or(0, |p| p.text.len());
        s.sel = Selection { anchor: Pos::new(StoryRef::Body, Path::top(block), 0), focus: Pos::new(StoryRef::Body, Path::top(block), len) };
    }

    #[test]
    fn scope_context_reads_the_session() {
        let mut s = session("Eins\nzwei\ndrei\nvier");
        select(&mut s, 2);
        assert_eq!(scope_context(&s, Scope::Paragraphs, 1, 1000), "zwei\ndrei\nvier");
        assert_eq!(scope_context(&s, Scope::Whole, 1, 1000), "Eins\nzwei\ndrei\nvier");
        assert_eq!(scope_context(&s, Scope::SelectionOnly, 1, 1000), "");
    }

    #[test]
    fn apply_is_one_tracked_undo_step_by_the_model() {
        let mut s = session("Erster Absatz.\nDer zweite Absatz ist viel zu lang geraten.");
        s.author = "Philip".into();
        select(&mut s, 1);
        let depth = s.undo_depth();
        let snap = Snapshot::of(&s);
        assert!(snap.holds(&s));
        let n = apply(&mut s, "Zweiter Absatz, kurz.", "qwen").unwrap();
        assert_eq!(n, 21);
        assert!(!snap.holds(&s), "the document changed");
        assert_eq!(s.undo_depth(), depth + 1, "one undo step");
        assert_eq!((s.author.as_str(), s.doc.settings.track_changes), ("Philip", false), "author and tracking restored");
        assert!(s.doc.final_text(StoryRef::Body).contains("Zweiter Absatz, kurz."));
        let changes = s.run("review.changes", &json!({})).unwrap();
        let authors: Vec<String> = changes
            .as_array()
            .or_else(|| changes.get("changes").and_then(Value::as_array))
            .into_iter()
            .flatten()
            .filter_map(|c| c.get("author").and_then(Value::as_str).map(str::to_string))
            .collect();
        assert!(!authors.is_empty() && authors.iter().all(|a| a == "AI (qwen)"), "{changes}");
        assert!(s.undo());
        assert_eq!(s.doc.final_text(StoryRef::Body).trim_end(), "Erster Absatz.\nDer zweite Absatz ist viel zu lang geraten.");
    }

    #[test]
    fn a_job_answers_cleans_and_fails_without_hanging() {
        let mut s = session("Ein Satz.");
        select(&mut s, 0);
        let ok = |_r: HttpRequest| {
            let (tx, rx) = std::sync::mpsc::channel();
            let _ = tx.send(Ok(r#"{"choices":[{"message":{"content":"```\nKurz.\n```"}}]}"#.to_string()));
            rx
        };
        let job = start(&strata(), Scope::Whole, "Kürzen", &s, &ok, 0.0).unwrap();
        assert!(matches!(job.poll(), Outcome::Answer(t) if t == "Kurz."));
        let down = |_r: HttpRequest| {
            let (tx, rx) = std::sync::mpsc::channel();
            let _ = tx.send(Err("network error: connection failed".to_string()));
            rx
        };
        let job = start(&strata(), Scope::Whole, "Kürzen", &s, &down, 0.0).unwrap();
        assert!(matches!(job.wait(std::time::Duration::from_secs(1)), Outcome::Failed(e) if e.starts_with("Modell nicht erreichbar: network")));
        let dropped = |_r: HttpRequest| std::sync::mpsc::channel().1;
        let job = start(&strata(), Scope::Whole, "Kürzen", &s, &dropped, 0.0).unwrap();
        assert!(matches!(job.poll(), Outcome::Failed(_)));
        s.sel = Selection::caret(s.sel.focus.clone());
        assert!(start(&strata(), Scope::Whole, "Kürzen", &s, &ok, 0.0).is_err(), "nothing selected");
    }
}
