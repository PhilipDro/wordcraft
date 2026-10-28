//! The harness rules enforced on every command the model runs (not only asked for in the prompt):
//! read before edit, permission before removing content or changing styles, and staying inside
//! the user's selection.
//!
//! Every tool reaches the document through [`Local`], so the rules sit in one place whatever
//! tool (execute, batch, type_text…) the model uses.

use std::sync::Arc;

use serde_json::{Value, json};
use wordcraft_engine::Session;
use wordcraft_engine::doc::para::OBJ;
use wordcraft_engine::doc::{Block, Document, StoryRef};
use wordcraft_mcp::Backend;

use crate::clip;

/// What a permission prompt is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermissionKind {
    /// Removing content: see [`removal`] for when an edit counts.
    Delete,
    /// Changing fonts, sizes, colours, highlighting or the theme: see [`style_change`].
    Style,
}

// Deletion thresholds. Counts are non-whitespace characters of the document as it reads with
// every change accepted (tracked deletions don't count). "Original" text is everything but what
// the assistant itself inserted during this request, so it may rework its own text freely.

/// One command losing this many characters net (original text removed minus text inserted) asks.
pub const DELETE_NET_CHARS: usize = 120;
/// One command removing at least this much original text…
pub const SHRINK_MIN_CHARS: usize = 60;
/// …and inserting less than 1/`SHRINK_KEEP_DIV` of it (a long passage replaced by a much shorter
/// one) asks.
pub const SHRINK_KEEP_DIV: usize = 3;
/// A paragraph with text disappearing asks once the command removed at least this much original
/// text (merging two paragraphs removes none and doesn't ask).
pub const PARAGRAPH_MIN_CHARS: usize = 10;
/// The whole request losing this many characters net since it started (or since the user last
/// allowed a deletion) asks, so many small deletions can't add up unnoticed.
pub const REQUEST_NET_CHARS: usize = 300;

const READ_FIRST: &str = "Refused: read the document first (get_text or inspect_document). Edits are allowed once you have read it in this request.";
const PAUSED: &str = "Waiting for the user's permission.";
const OUTSIDE: &str = "Refused: this edit reaches outside the paragraphs the user selected. Work only inside the selection.";

/// Per-request harness state.
#[derive(Clone, Default)]
pub(crate) struct Guard {
    /// The model read the document since the user's message.
    pub read: bool,
    /// The model edited since it last read the document.
    pub unverified: bool,
    /// Permissions given for the rest of the request.
    pub granted: Vec<PermissionKind>,
    /// Permissions given for the tool call being run again.
    pub once: Vec<PermissionKind>,
    /// Set by a command that needs a permission it doesn't have: the tool call is taken back and
    /// the request pauses.
    pub need: Option<(PermissionKind, String)>,
    pub scope: Option<Scope>,
    /// The document revision after the assistant's last edit: while nobody else edits, the next
    /// edit joins the same Undo step.
    pub last_rev: Option<u64>,
    /// Total characters at the request start or the last allowed deletion ([`REQUEST_NET_CHARS`]).
    pub base_total: usize,
}

impl Guard {
    fn allows(&self, k: PermissionKind) -> bool {
        self.granted.contains(&k) || self.once.contains(&k)
    }
}

/// The paragraphs the user selected when the request started, as top-level blocks of one story:
/// `start` blocks before them and `tail` blocks after them must stay as they were. Both counts
/// are fixed for the request, so paragraphs inserted or removed inside the selection move its
/// end without any bookkeeping.
#[derive(Clone, Debug)]
pub(crate) struct Scope {
    story: StoryRef,
    start: usize,
    tail: usize,
}

impl Scope {
    /// The scope of a non-collapsed selection.
    pub fn of(s: &Session) -> Option<Scope> {
        if s.sel.is_collapsed() {
            return None;
        }
        let (a, b) = s.sel.ordered();
        let n = s.doc.story(a.story)?.len();
        let first = a.path.0.first().copied().unwrap_or(0) as usize;
        let mut last = b.path.0.first().copied().unwrap_or(0) as usize;
        // A selection that ends at the very start of a paragraph (a triple-click takes the
        // paragraph mark) doesn't include that paragraph.
        if b.story == a.story && b.off == 0 && b.path.0.len() == 1 && last > first {
            last -= 1;
        }
        if b.story != a.story {
            last = n.saturating_sub(1);
        }
        Some(Scope { story: a.story, start: first.min(n), tail: n.saturating_sub(last.saturating_add(1)) })
    }

    /// Whether going from `before` to `after` changed only the selected blocks.
    fn check(&self, id: &str, before: &Document, after: &Document) -> Result<(), String> {
        let (Some(b), Some(a)) = (before.story(self.story), after.story(self.story)) else { return Err(OUTSIDE.into()) };
        let keep = self.start.saturating_add(self.tail);
        if a.len() < keep || b.len() < keep {
            return Err(OUTSIDE.into());
        }
        let prefix = same(b.get(..self.start), a.get(..self.start));
        let suffix = same(b.get(b.len() - self.tail..), a.get(a.len() - self.tail..));
        // Other stories: the body, and parts that existed before (a part the edit removed — a
        // deleted note or text box in the selection — is fine; new ones are too).
        let body = self.story == StoryRef::Body || same(Some(&before.body), Some(&after.body));
        let parts = before
            .parts
            .iter()
            .all(|(k, p)| StoryRef::Part(*k) == self.story || after.parts.get(k).is_none_or(|q| same(Some(&p.blocks), Some(&q.blocks))));
        let doc_wide = id.starts_with("design.") || matches!(id, "styles.modify" | "styles.updateToMatch" | "styles.delete");
        if prefix && suffix && body && parts && !doc_wide && before.last_section == after.last_section { Ok(()) } else { Err(OUTSIDE.into()) }
    }
}

fn same(a: Option<&[Arc<Block>]>, b: Option<&[Arc<Block>]>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| Arc::ptr_eq(x, y) || x == y),
        _ => false,
    }
}

/// Commands that change fonts, sizes, colours, highlighting, shading or the theme, and so ask
/// first (plus `format.set` when it sets one of those). Bold, italic, underline, paragraph styles
/// (Heading 1…), lists and alignment are ordinary formatting and don't.
pub const STYLE_COMMANDS: &[&str] = &[
    "format.font",
    "format.size",
    "format.growFont",
    "format.shrinkFont",
    "format.growFont1",
    "format.shrinkFont1",
    "format.color",
    "format.highlight",
    "format.shading",
    "format.clear",
    "para.shading",
    "table.shading",
    "design.theme",
    "design.themes",
    "design.themeFonts",
    "design.themeColors",
    "design.styleSet",
    "design.pageColor",
    "design.setDefault",
    "styles.modify",
    "styles.updateToMatch",
];

/// Whether running `id` with `params` changes a style (see [`STYLE_COMMANDS`]).
pub fn style_change(id: &str, params: &Value) -> bool {
    if id == "format.set" {
        return params.get("props").and_then(Value::as_object).is_some_and(|p| {
            p.keys()
                .any(|k| k.starts_with("font") || k.starts_with("size") || matches!(k.as_str(), "color" | "highlight" | "shading" | "underlineColor"))
        });
    }
    STYLE_COMMANDS.contains(&id)
}

/// What the document holds, for telling a removal from a rewrite.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Measure {
    /// Characters that aren't the assistant's own insertions.
    pub original: usize,
    /// The assistant's own inserted characters.
    pub own: usize,
    /// Paragraphs with any text.
    pub paras: usize,
    /// Pictures, shapes, equations, fields… (not bookmark or comment markers).
    pub objects: usize,
}

impl Measure {
    pub fn total(&self) -> usize {
        self.original.saturating_add(self.own)
    }
}

/// Measure every story (body, headers, footers, notes, text boxes).
// ponytail: walks the whole document twice per edit; fine for book-length documents, measure only
// the touched stories if it ever shows in a profile.
pub(crate) fn measure(doc: &Document, author: &str) -> Measure {
    let mut m = Measure::default();
    let own = |rev: Option<u32>| rev.and_then(|r| doc.revisions.get(r as usize)).is_some_and(|r| r.author == author);
    let stories = std::iter::once(StoryRef::Body).chain(doc.parts.keys().map(|k| StoryRef::Part(*k)));
    for story in stories {
        for path in doc.para_paths(story) {
            let Some(p) = doc.para(story, &path) else { continue };
            let mut k = 0;
            let mut any = false;
            for (range, props) in p.run_ranges() {
                let gone = props.del.is_some();
                let mine = own(props.ins);
                for c in p.text.get(range).unwrap_or("").chars() {
                    if c == OBJ {
                        if !gone && p.objects.get(k).is_some_and(|o| !o.is_marker()) {
                            m.objects += 1;
                        }
                        k += 1;
                    } else if !gone && !c.is_whitespace() {
                        any = true;
                        if mine { m.own += 1 } else { m.original += 1 }
                    }
                }
            }
            m.paras += usize::from(any);
        }
    }
    m
}

/// Why an edit from `before` to `after` needs the user's permission to remove content, if it
/// does. `selected` is the text that was selected before the edit.
pub(crate) fn removal(before: &Measure, after: &Measure, base_total: usize, selected: &str) -> Option<String> {
    let removed = before.original.saturating_sub(after.original);
    let inserted = after.own.saturating_sub(before.own);
    let net = removed.saturating_sub(inserted);
    let quote = if selected.trim().is_empty() { String::new() } else { format!(": “{}”", clip(selected.trim(), 160)) };
    if after.objects < before.objects {
        let n = before.objects - after.objects;
        return Some(format!("Delete {n} picture(s) or other object(s){quote}"));
    }
    let paras = before.paras.saturating_sub(after.paras);
    if net >= DELETE_NET_CHARS
        || (removed >= SHRINK_MIN_CHARS && inserted.saturating_mul(SHRINK_KEEP_DIV) < removed)
        || (paras > 0 && removed >= PARAGRAPH_MIN_CHARS)
    {
        let p = if paras > 0 { format!(" ({paras} paragraph(s))") } else { String::new() };
        return Some(format!("Delete about {removed} characters of text{p}{quote}"));
    }
    let total = base_total.saturating_sub(after.total());
    if total >= REQUEST_NET_CHARS {
        return Some(format!("Delete more text: about {total} characters removed in this request so far{quote}"));
    }
    None
}

/// Commands the assistant may not run: file I/O, switching tracking or protection off, undo
/// (it would undo the user's work), macros.
pub(crate) fn blocked(id: &str) -> bool {
    id.starts_with("file.")
        || id.starts_with("review.restrict")
        || matches!(id, "review.trackChanges" | "edit.undo" | "edit.redo" | "edit.repeat" | "tools.recordMacro" | "tools.macros")
}

/// The open session as an MCP backend, limited to what the assistant may do.
pub(crate) struct Local<'a> {
    pub session: &'a mut Session,
    pub guard: &'a mut Guard,
}

impl Local<'_> {
    fn execute(&mut self, id: &str, p: &Value) -> Result<Value, String> {
        let Some(spec) = self.session.registry.get(id) else { return self.session.run(id, p).map_err(|e| e.to_string()) };
        let (mutates, label) = (spec.mutates, spec.label);
        if !mutates {
            let r = self.session.run(id, p).map_err(|e| e.to_string());
            if id == "document.text" && r.is_ok() {
                self.read();
            }
            return r;
        }
        if !self.guard.read {
            return Err(READ_FIRST.into());
        }
        let selected = self.session.selected_text();
        if style_change(id, p) && !self.guard.allows(PermissionKind::Style) {
            let params = if p.as_object().is_some_and(|o| !o.is_empty()) { format!(" {}", clip(&p.to_string(), 120)) } else { String::new() };
            let on = if selected.trim().is_empty() { String::new() } else { format!(" on “{}”", clip(selected.trim(), 80)) };
            return self.ask(PermissionKind::Style, format!("{label}{params}{on}"));
        }
        let author = self.session.author.clone();
        let before = measure(&self.session.doc, &author);
        let snap = self.session.edit_snapshot();
        let rev0 = self.session.rev();
        if self.guard.last_rev == Some(rev0) {
            self.session.join_next_undo();
        }
        let r = self.session.run(id, p).map_err(|e| e.to_string());
        if self.session.rev() == rev0 {
            return r;
        }
        if let Some(scope) = &self.guard.scope
            && let Err(e) = scope.check(id, snap.doc(), &self.session.doc)
        {
            self.take_back(snap, rev0);
            return Err(e);
        }
        let after = measure(&self.session.doc, &author);
        if let Some(why) = removal(&before, &after, self.guard.base_total, &selected) {
            if !self.guard.allows(PermissionKind::Delete) {
                self.take_back(snap, rev0);
                return self.ask(PermissionKind::Delete, why);
            }
            self.guard.base_total = after.total();
        }
        self.guard.unverified = true;
        self.guard.last_rev = Some(self.session.rev());
        r
    }

    fn read(&mut self) {
        self.guard.read = true;
        self.guard.unverified = false;
    }

    fn ask(&mut self, kind: PermissionKind, why: String) -> Result<Value, String> {
        if self.guard.need.is_none() {
            self.guard.need = Some((kind, why));
        }
        Err(PAUSED.into())
    }

    /// Undo a command without a trace, keeping the assistant's Undo step joinable.
    fn take_back(&mut self, snap: wordcraft_engine::EditSnapshot, rev0: u64) {
        take_back(self.session, &mut self.guard.last_rev, snap, rev0);
    }
}

/// Put `snap` (taken at revision `rev0`) back; if the assistant's last edit was the state then,
/// the next edit still joins its Undo step.
pub(crate) fn take_back(s: &mut Session, last_rev: &mut Option<u64>, snap: wordcraft_engine::EditSnapshot, rev0: u64) {
    s.restore(snap);
    if *last_rev == Some(rev0) {
        *last_rev = Some(s.rev());
    }
}

impl Backend for Local<'_> {
    fn call(&mut self, method: &str, params: Value) -> Result<Value, String> {
        if self.guard.need.is_some() {
            return Err(PAUSED.into());
        }
        match method {
            "engine.execute" => {
                let id = params.get("command").and_then(Value::as_str).ok_or("missing `command`")?;
                if blocked(id) {
                    return Err(format!("`{id}` isn't available to the assistant"));
                }
                let p = params.get("params").cloned().filter(|v| !v.is_null()).unwrap_or(json!({}));
                self.execute(id, &p)
            }
            "engine.commands" => match self.session.registry.describe() {
                Value::Array(all) => {
                    Ok(Value::Array(all.into_iter().filter(|c| c.get("id").and_then(Value::as_str).is_some_and(|id| !blocked(id))).collect()))
                }
                v => Ok(v),
            },
            "document.inspect" => {
                let r = self.session.run("document.inspect", &params).map_err(|e| e.to_string());
                if r.is_ok() {
                    self.read();
                }
                r
            }
            m => Err(format!("`{m}` isn't available to the assistant")),
        }
    }
    fn has_ui(&self) -> bool {
        false
    }
    fn describe(&self) -> String {
        "assistant".into()
    }
}
