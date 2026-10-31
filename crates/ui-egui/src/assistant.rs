//! The Assistant pane: a chat with an AI model that edits the open document through the command
//! registry (`wordcraft-assistant`). Its edits are tracked changes, and one request is one Undo
//! step. The provider settings (API key included) live here and in the host's own file — never in
//! `UiState`, logs, the control channel or MCP output.
//!
//! A request can pause for the user: the model asks a question (a card with its options; the
//! input box takes a free answer) or an edit needs permission (a card with Allow once / Allow for
//! this request / Deny). The model's plan shows as a checklist above the chat.

use egui::{Align, Color32, Layout, Rect, RichText, Stroke, Ui, vec2};
use serde_json::{Value, json};
use wordcraft_assistant::{Agent, Decision, Entry, PRESETS, Pending, PermissionKind, PlanStep, Protocol, Settings, providers::CUSTOM, wire};

use crate::WordApp;
use crate::theme::{Tokens, regular, semibold};
use crate::widgets::icon_text_button;

#[derive(Default)]
pub struct AssistantPane {
    pub open: bool,
    pub agent: Agent,
    /// Provider, keys and models. The host loads and saves them; never in `UiState`.
    pub settings: Settings,
    /// The user edited `settings` in the pane: the host saves them and resets the flag.
    pub settings_changed: bool,
    /// The message being typed.
    pub input: String,
    /// The providers' model lists, fetched on demand. In memory only: not in `settings`.
    pub models: Models,
    /// Start with the Settings section open (tests and screenshots).
    pub settings_open: bool,
    /// Prompt on selection (ai.rewriteSelection): its state lives here; see
    /// [crate::rewrite_popover].
    pub rewrite: crate::rewrite_popover::RewritePopover,
}

/// A fetched model list and what it was fetched for.
#[derive(Default)]
struct ModelList {
    /// Protocol, URL and key it was fetched with: any change makes the list stale.
    key: String,
    list: Vec<String>,
    error: Option<String>,
}

struct Flight {
    provider: String,
    key: String,
    protocol: Protocol,
    rx: std::sync::mpsc::Receiver<Result<String, String>>,
}

/// Model lists per provider, one fetch in flight at a time.
#[derive(Default)]
pub struct Models {
    lists: std::collections::BTreeMap<String, ModelList>,
    flight: Option<Flight>,
    filter: String,
}

/// Any API key in `text` (a provider echoing it in an error) is blanked.
fn blank_keys(settings: &Settings, text: &str) -> String {
    settings.providers.values().map(|p| p.api_key.trim()).filter(|k| k.len() >= 4).fold(text.to_string(), |s, k| s.replace(k, "•••"))
}

fn cache_key(ep: &wordcraft_assistant::Endpoint) -> String {
    format!("{:?}|{}|{}", ep.protocol, ep.base_url, ep.api_key)
}

/// Ask the chosen provider for its models, unless a list for the same key and URL is cached (or
/// `force`), a fetch is running, the key is missing or there is no HTTP client.
fn fetch_models(app: &mut WordApp, force: bool) {
    let Some(http) = app.services.http.as_deref() else { return };
    let pane = &mut app.assistant;
    if pane.models.flight.is_some() {
        return;
    }
    let Ok(ep) = pane.settings.connection() else { return };
    let provider = pane.settings.provider_id().to_string();
    let key = cache_key(&ep);
    if !force && pane.models.lists.get(&provider).is_some_and(|l| l.key == key) {
        return;
    }
    pane.models.flight = Some(Flight { provider, key, protocol: ep.protocol, rx: http(wire::models_request(&ep)) });
}

/// Take the model list's answer if it has arrived. True while a fetch is still running.
fn poll_models(app: &mut WordApp) -> bool {
    let models = &mut app.assistant.models;
    let Some(f) = &models.flight else { return false };
    let res = match f.rx.try_recv() {
        Ok(r) => r,
        Err(std::sync::mpsc::TryRecvError::Empty) => return true,
        Err(std::sync::mpsc::TryRecvError::Disconnected) => Err("the request failed".into()),
    };
    let Some(f) = models.flight.take() else { return false };
    let (list, error) = match res.and_then(|body| wire::parse_models(f.protocol, &body)) {
        Ok(list) => (list, None),
        Err(e) => (Vec::new(), Some(e)),
    };
    models.lists.insert(f.provider, ModelList { key: f.key, list, error });
    false
}

/// The chosen provider's list, if it was fetched with the current key and URL.
fn current_models(pane: &AssistantPane) -> Option<&ModelList> {
    let key = cache_key(&pane.settings.connection().ok()?);
    pane.models.lists.get(pane.settings.provider_id()).filter(|l| l.key == key)
}

/// `ui.assistant {"models": true}`'s answer; never the key.
fn models_state(pane: &AssistantPane) -> Value {
    let loading = pane.models.flight.as_ref().is_some_and(|f| f.provider == pane.settings.provider_id());
    let cur = current_models(pane);
    let list = cur.map_or(&[][..], |l| &l.list[..]);
    let error = cur.and_then(|l| l.error.as_deref()).map(|e| blank_keys(&pane.settings, e));
    json!({"loading": loading, "list": list, "error": error})
}

/// The user's reply to a paused request.
enum Reply {
    Answer(String),
    Decide(Decision),
}

/// `ui.assistant`: `open` (omit to toggle), `ask`, `stop`, `clear`, `answer` (text for the
/// model's question), `decide` (`allowOnce`, `allowRequest` or `deny` for a permission prompt).
/// Returns `open`, `busy`, the chat `log`, `pending` (`{kind: "question", question, options}`,
/// `{kind: "permission", permissionKind: "delete"|"style", description}` or null) and `plan`
/// (`[{text, done}]`) — never the settings or the key. `models: true` fetches the chosen provider's
/// model list (or reuses it) and adds `models: {loading, list, error}`.
pub fn command(app: &mut WordApp, p: &Value) -> Result<Value, String> {
    let flag = |k: &str| p.get(k).and_then(Value::as_bool) == Some(true);
    let ask = p.get("ask").and_then(Value::as_str);
    let answer = p.get("answer").and_then(Value::as_str);
    let decide = match p.get("decide") {
        None | Some(Value::Null) => None,
        Some(v) => Some(match v.as_str() {
            Some("allowOnce") => Decision::AllowOnce,
            Some("allowRequest") => Decision::AllowForRequest,
            Some("deny") => Decision::Deny,
            _ => return Err("`decide` must be allowOnce, allowRequest or deny".into()),
        }),
    };
    if flag("models") {
        fetch_models(app, false);
        poll_models(app);
    }
    if flag("clear") {
        app.assistant.agent.clear(&mut app.session);
    }
    if flag("stop") {
        app.assistant.agent.stop(&mut app.session);
    }
    let replying = answer.is_some() || decide.is_some();
    match p.get("open").and_then(Value::as_bool) {
        Some(v) => app.assistant.open = v,
        None if ask.is_some() || replying => app.assistant.open = true,
        None if !flag("clear") && !flag("stop") && !flag("models") => app.assistant.open = !app.assistant.open,
        None => {}
    }
    if let Some(text) = ask {
        let Some(http) = app.services.http.as_deref() else { return Err(unavailable().into()) };
        app.assistant.agent.ask(text, &app.assistant.settings, &mut app.session, http)?;
    }
    if let Some(text) = answer {
        if !matches!(app.assistant.agent.pending(), Some(Pending::Question { .. })) {
            return Err("no question is waiting for an answer".into());
        }
        reply(app, Reply::Answer(text.to_string()));
    }
    if let Some(d) = decide {
        if !matches!(app.assistant.agent.pending(), Some(Pending::Permission { .. })) {
            return Err("no permission prompt is waiting".into());
        }
        reply(app, Reply::Decide(d));
    }
    let mut out = state(&app.assistant);
    if flag("models") {
        out["models"] = models_state(&app.assistant);
    }
    Ok(out)
}

fn reply(app: &mut WordApp, r: Reply) {
    let Some(http) = app.services.http.as_deref() else { return };
    let (agent, session) = (&mut app.assistant.agent, &mut app.session);
    match r {
        Reply::Answer(text) => agent.answer(&text, session, http),
        Reply::Decide(d) => agent.decide(d, session, http),
    }
}

fn unavailable() -> &'static str {
    "The assistant isn't available here."
}

fn kind_id(k: PermissionKind) -> &'static str {
    match k {
        PermissionKind::Delete => "delete",
        PermissionKind::Style => "style",
    }
}

/// What `ui.assistant` reports. Any API key that turns up in a text (a provider echoing it in an
/// error, a model quoting it) is blanked.
fn state(pane: &AssistantPane) -> Value {
    let keys: Vec<&str> = pane.settings.providers.values().map(|p| p.api_key.trim()).filter(|k| k.len() >= 4).collect();
    let clean = |s: &str| keys.iter().fold(s.to_string(), |s, k| s.replace(k, "•••"));
    let cleans = |v: &[String]| v.iter().map(|s| clean(s)).collect::<Vec<_>>();
    let log: Vec<Value> = pane
        .agent
        .log
        .iter()
        .map(|e| match e {
            Entry::User(t) => json!({"kind": "user", "text": clean(t)}),
            Entry::Assistant(t) => json!({"kind": "assistant", "text": clean(t)}),
            Entry::Action { label, detail, ok } => json!({"kind": "action", "text": label, "detail": clean(detail), "ok": ok}),
            Entry::Error(t) => json!({"kind": "error", "text": clean(t)}),
            Entry::Question { question, options } => json!({"kind": "question", "text": clean(question), "options": cleans(options)}),
            Entry::Permission { kind, description, granted } => {
                json!({"kind": "permission", "permissionKind": kind_id(*kind), "text": clean(description), "ok": granted})
            }
        })
        .collect();
    let pending = match pane.agent.pending() {
        Some(Pending::Question { question, options }) => json!({"kind": "question", "question": clean(question), "options": cleans(options)}),
        Some(Pending::Permission { kind, description }) => {
            json!({"kind": "permission", "permissionKind": kind_id(*kind), "description": clean(description)})
        }
        None => Value::Null,
    };
    let plan: Vec<Value> = pane.agent.plan().iter().map(|s| json!({"text": clean(&s.text), "done": s.done})).collect();
    json!({"open": pane.open, "busy": pane.agent.busy(), "log": log, "pending": pending, "plan": plan})
}

/// Each frame: take the model's answer if it has arrived (runs its tool calls).
pub(crate) fn poll(app: &mut WordApp, ctx: &egui::Context) {
    if poll_models(app) {
        ctx.request_repaint_after(std::time::Duration::from_millis(150));
    }
    // Idle, or paused for the user: nothing in flight, nothing to animate.
    if !app.assistant.agent.busy() || app.assistant.agent.pending().is_some() {
        return;
    }
    if let Some(http) = app.services.http.as_deref()
        && app.assistant.agent.poll(&mut app.session, http)
    {
        ctx.request_repaint();
    }
    // The host wakes the UI when an answer arrives; this keeps the spinner turning meanwhile.
    // Not once the answer paused the request.
    if app.assistant.agent.pending().is_none() {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
}

pub fn show(app: &mut WordApp, ui: &mut Ui) {
    if !app.assistant.open {
        return;
    }
    let t = Tokens::get(ui.ctx());
    egui::Panel::right("assistant_pane")
        .default_size(320.0)
        .resizable(true)
        .frame(egui::Frame::NONE.fill(t.panel).inner_margin(10).stroke(Stroke::new(1.0, t.border)))
        .show(ui, |ui| pane(app, ui, &t));
}

fn pane(app: &mut WordApp, ui: &mut Ui, t: &Tokens) {
    if header(ui, "Assistant") {
        app.assistant.open = false;
        return;
    }
    let available = app.services.http.is_some();
    ui.horizontal(|ui| {
        let busy = app.assistant.agent.busy();
        if ui.add_enabled(!app.assistant.agent.log.is_empty() || busy, egui::Button::new(tl!("New chat"))).clicked() {
            app.assistant.agent.clear(&mut app.session);
        }
    });
    if let Some(refresh) = settings(&mut app.assistant, ui) {
        fetch_models(app, refresh);
        ui.ctx().request_repaint();
    }
    ui.separator();
    plan(ui, app.assistant.agent.plan(), t);
    let pending = app.assistant.agent.pending().cloned();
    let mut act = None;
    // The chat log fills what the input area leaves.
    let input_h = 118.0;
    egui::ScrollArea::vertical()
        .id_salt("assistant_log")
        .auto_shrink([false, false])
        .max_height((ui.available_height() - input_h).max(60.0))
        .stick_to_bottom(true)
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 4.0;
            // Action rows are one small line, not a button's height.
            ui.spacing_mut().interact_size.y = 16.0;
            let log = &app.assistant.agent.log;
            if !available {
                ui.label(RichText::new(tl!(unavailable())).color(t.text_dim));
            } else if log.is_empty() {
                ui.label(
                    RichText::new(tl!("Ask for changes to your document, such as \"Make the title bold\" or \"Shorten the second paragraph\"."))
                        .color(t.text_dim),
                );
            }
            // The question being asked is the card below, not a bubble too.
            let asking = matches!(pending, Some(Pending::Question { .. })) && matches!(log.last(), Some(Entry::Question { .. }));
            for e in log.iter().take(log.len().saturating_sub(usize::from(asking))) {
                entry(ui, e, t);
            }
            match &pending {
                Some(Pending::Question { question, options }) => act = question_card(ui, question, options, t),
                Some(Pending::Permission { kind, description }) => act = permission_card(ui, *kind, description, t),
                None => {}
            }
            if pending.is_some() {
                ui.add_space(4.0);
                ui.label(RichText::new(tl!("Waiting for you")).font(regular(11.5)).color(t.text_dim));
            } else if app.assistant.agent.busy() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new(tl!("Working…")).color(t.text_dim));
                });
            }
        });
    if let Some(r) = act {
        reply(app, r);
    }
    ui.separator();
    input(app, ui, available, t);
}

/// The model's plan as a checklist; collapsible, and scrolls when long.
fn plan(ui: &mut Ui, steps: &[PlanStep], t: &Tokens) {
    if steps.is_empty() {
        return;
    }
    let done = steps.iter().filter(|s| s.done).count();
    let title = format!("{} {done}/{}", tl!("Plan"), steps.len());
    egui::CollapsingHeader::new(RichText::new(title).font(semibold(12.5))).id_salt("assistant_plan").default_open(true).show(ui, |ui| {
        egui::ScrollArea::vertical().id_salt("assistant_plan_steps").max_height(132.0).show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 3.0;
            for s in steps {
                ui.horizontal_top(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    let (r, _) = ui.allocate_exact_size(vec2(12.0, 14.0), egui::Sense::hover());
                    let b = Rect::from_center_size(r.center(), vec2(11.0, 11.0));
                    if s.done {
                        ui.painter().rect_filled(b, 2, t.accent);
                        crate::icons::paint(ui.painter(), b.shrink(1.0), "check", t.on_accent, t.on_accent);
                    } else {
                        ui.painter().rect_stroke(b, 2, Stroke::new(1.0, t.border_strong), egui::StrokeKind::Inside);
                    }
                    let text = RichText::new(&s.text).font(regular(12.0));
                    let text = if s.done { text.color(t.text_dim).strikethrough() } else { text.color(t.text) };
                    ui.add(egui::Label::new(text).wrap());
                });
            }
        });
    });
    ui.separator();
}

/// A card in the chat: `tone` tints its fill and draws its border.
fn card(tone: Color32) -> egui::Frame {
    egui::Frame::NONE.fill(tone.gamma_multiply(0.10)).stroke(Stroke::new(1.0, tone)).corner_radius(8).inner_margin(egui::Margin::same(10))
}

fn question_card(ui: &mut Ui, question: &str, options: &[String], t: &Tokens) -> Option<Reply> {
    ui.add_space(8.0);
    let mut out = None;
    card(t.accent).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.spacing_mut().item_spacing.y = 6.0;
        ui.label(RichText::new(tl!("Question")).font(semibold(11.5)).color(t.accent_text));
        ui.add(egui::Label::new(RichText::new(question).color(t.text)).wrap().selectable(true));
        let w = ui.available_width();
        for o in options {
            let b = egui::Button::new(RichText::new(o).color(t.text)).wrap().min_size(vec2(w, 26.0));
            if ui.add(b).clicked() {
                out = Some(Reply::Answer(o.clone()));
            }
        }
        let hint = if options.is_empty() { tl!("Type your answer below.") } else { tl!("Or type your own answer below.") };
        ui.label(RichText::new(hint).font(regular(11.0)).color(t.text_dim));
    });
    out
}

fn permission_card(ui: &mut Ui, kind: PermissionKind, description: &str, t: &Tokens) -> Option<Reply> {
    ui.add_space(8.0);
    let mut out = None;
    card(t.orange).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.spacing_mut().item_spacing.y = 6.0;
        ui.horizontal(|ui| {
            // A drawn exclamation mark in a circle.
            let (r, _) = ui.allocate_exact_size(vec2(18.0, 18.0), egui::Sense::hover());
            let c = r.center();
            ui.painter().circle_filled(c, 8.0, t.orange);
            ui.painter().line_segment([c + vec2(0.0, -4.5), c + vec2(0.0, 1.2)], Stroke::new(1.8, t.on_accent));
            ui.painter().circle_filled(c + vec2(0.0, 4.0), 1.1, t.on_accent);
            let title = match kind {
                PermissionKind::Delete => tl!("Delete content?"),
                PermissionKind::Style => tl!("Change formatting style?"),
            };
            ui.label(RichText::new(title).font(semibold(13.0)).color(t.text));
        });
        // The harness's own description of the change (English, quoting the document).
        ui.add(egui::Label::new(RichText::new(description).color(t.text)).wrap().selectable(true));
        ui.horizontal_wrapped(|ui| {
            // Allow and Deny differ by their drawn mark (check, cross), not by colour alone.
            for (icon, label, d) in [
                ("check", "Allow once", Decision::AllowOnce),
                ("check", "Allow for this request", Decision::AllowForRequest),
                ("close", "Deny", Decision::Deny),
            ] {
                if icon_text_button(ui, icon, tl!(label), regular(12.5)).clicked() {
                    out = Some(Reply::Decide(d));
                }
            }
        });
    });
    out
}

/// A drawn check or cross, then a small dim line.
fn mark_line(ui: &mut Ui, ok: bool, text: String, t: &Tokens) -> egui::Response {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing = vec2(4.0, 0.0);
        // Drawn marks: the check and cross symbols aren't in every interface font.
        let (r, resp) = ui.allocate_exact_size(vec2(12.0, 12.0), egui::Sense::hover());
        let (icon, c) = if ok { ("check", t.green) } else { ("close", t.red) };
        crate::icons::paint(ui.painter(), Rect::from_center_size(r.center(), vec2(11.0, 11.0)), icon, c, c);
        ui.add(egui::Label::new(RichText::new(text).font(regular(11.5)).color(t.text_dim)).truncate());
        resp
    })
    .inner
}

fn entry(ui: &mut Ui, e: &Entry, t: &Tokens) {
    // Action lines sit close together under the message that led to them.
    if !matches!(e, Entry::Action { .. } | Entry::Permission { .. }) {
        ui.add_space(8.0);
    }
    match e {
        Entry::User(text) => {
            ui.with_layout(Layout::top_down(Align::Max), |ui| {
                let max = ui.available_width() * 0.85;
                egui::Frame::NONE.fill(t.checked).corner_radius(8).inner_margin(egui::Margin::symmetric(9, 6)).show(ui, |ui| {
                    ui.set_max_width(max);
                    ui.add(egui::Label::new(RichText::new(text).color(t.text)).wrap().halign(Align::Min));
                });
            });
        }
        Entry::Assistant(text) => {
            ui.add(egui::Label::new(RichText::new(text).color(t.text)).wrap().selectable(true));
        }
        Entry::Action { label, detail, ok } => {
            let text = if detail.is_empty() { tl!(label).to_string() } else { format!("{} {detail}", tl!(label)) };
            let resp = mark_line(ui, *ok, text, t);
            if !ok {
                resp.on_hover_text(tl!("Failed"));
            }
        }
        Entry::Error(text) => {
            ui.add(egui::Label::new(RichText::new(text).color(t.red)).wrap());
        }
        // What the model asked; the answer follows as the user's bubble.
        Entry::Question { question, options } => {
            let max = ui.available_width() * 0.85;
            egui::Frame::NONE.stroke(Stroke::new(1.0, t.border_strong)).corner_radius(8).inner_margin(egui::Margin::symmetric(9, 6)).show(ui, |ui| {
                ui.set_max_width(max);
                ui.spacing_mut().item_spacing.y = 2.0;
                ui.label(RichText::new(tl!("Asked")).font(semibold(11.0)).color(t.text_dim));
                ui.add(egui::Label::new(RichText::new(question).color(t.text)).wrap());
                for o in options {
                    ui.horizontal_top(|ui| {
                        ui.add_space(8.0);
                        ui.add(egui::Label::new(RichText::new(o).font(regular(11.5)).color(t.text_dim)).wrap());
                    });
                }
            });
        }
        Entry::Permission { description, granted, .. } => {
            let verdict = if *granted { tl!("Allowed:") } else { tl!("Denied:") };
            mark_line(ui, *granted, format!("{verdict} {description}"), t).on_hover_text(description);
        }
    }
}

fn input(app: &mut WordApp, ui: &mut Ui, available: bool, t: &Tokens) {
    let busy = app.assistant.agent.busy();
    let pending = app.assistant.agent.pending();
    // A question takes the typed text as its answer; a permission prompt takes only its buttons.
    let answering = matches!(pending, Some(Pending::Question { .. }));
    let deciding = matches!(pending, Some(Pending::Permission { .. }));
    let hint = if answering { tl!("Type your answer…") } else { tl!("Ask the assistant…") };
    let edit = egui::TextEdit::multiline(&mut app.assistant.input)
        .id_salt("assistant_input")
        .hint_text(hint)
        .desired_rows(3)
        .desired_width(f32::INFINITY)
        // Enter sends; Shift+Enter starts a new line.
        .return_key(Some(egui::KeyboardShortcut::new(egui::Modifiers::SHIFT, egui::Key::Enter)));
    let resp = ui.add_enabled(available, edit);
    let enter = resp.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
    let mut send = false;
    ui.horizontal(|ui| {
        if !busy || answering || deciding {
            let can = available && !app.assistant.input.trim().is_empty() && (!busy || answering);
            send = ui.add_enabled(can, egui::Button::new(tl!("Send"))).clicked() || (enter && can);
        }
        if busy && ui.button(tl!("Stop")).clicked() {
            app.assistant.agent.stop(&mut app.session);
            send = false;
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.add(egui::Label::new(RichText::new(tl!("Edits are tracked changes.")).font(regular(11.0)).color(t.text_dim)).truncate());
        });
    });
    if !send {
        return;
    }
    let text = std::mem::take(&mut app.assistant.input);
    if answering {
        reply(app, Reply::Answer(text));
    } else if let Some(http) = app.services.http.as_deref() {
        // A refusal (no key yet…) is shown in the log; keep what was typed.
        if app.assistant.agent.ask(&text, &app.assistant.settings, &mut app.session, http).is_err() {
            app.assistant.input = text;
        }
    }
    resp.request_focus();
}

/// The Settings section. `Some(refresh)` while it is open: the models should be fetched (always
/// when `refresh`, else only if not cached yet).
fn settings(pane: &mut AssistantPane, ui: &mut Ui) -> Option<bool> {
    let open = pane.settings_open;
    egui::CollapsingHeader::new(RichText::new(tl!("Settings")).font(semibold(12.5)))
        .id_salt("assistant_settings")
        .default_open(open)
        .show(ui, |ui| {
            let id = pane.settings.provider_id().to_string();
            let preset = PRESETS.iter().find(|p| p.id == id);
            let name = preset.map_or("", |p| p.name);
            ui.label(tl!("Provider"));
            craft_ui::ComboBox::from_id_salt("assistant_provider").selected_text(tl!(name)).width(ui.available_width()).show_ui(ui, |ui| {
                for p in PRESETS {
                    if ui.selectable_label(p.id == id, tl!(p.name)).clicked() && p.id != id {
                        pane.settings.provider = p.id.to_string();
                        pane.settings_changed = true;
                    }
                }
            });
            let id = pane.settings.provider_id().to_string();
            let preset = PRESETS.iter().find(|p| p.id == id);
            let custom = id == CUSTOM;
            // What the model dropdown shows: the cached list, or why there is none.
            let loading = pane.models.flight.as_ref().is_some_and(|f| f.provider == id);
            let cached = current_models(pane);
            let list: Vec<String> = cached.map(|l| l.list.clone()).unwrap_or_default();
            let status = if loading {
                Some((false, tl!("Loading models…").to_string()))
            } else {
                cached.map(|l| match &l.error {
                    Some(e) => (true, blank_keys(&pane.settings, e)),
                    None => (false, format!("{} {}", l.list.len(), tl!("models"))),
                })
            };
            let t = Tokens::get(ui.ctx());
            let mut refresh = false;
            let cur = pane.settings.current_mut();
            let mut changed = false;
            let field = |ui: &mut Ui, label: &str, value: &mut String, hint: &str, password: bool| {
                ui.label(tl!(label));
                ui.add(egui::TextEdit::singleline(value).hint_text(hint).password(password).desired_width(f32::INFINITY)).changed()
            };
            changed |= field(ui, "API key", &mut cur.api_key, "", true);
            ui.label(tl!("Model"));
            let mut pick = None;
            ui.horizontal(|ui| {
                let w = (ui.available_width() - 2.0 * 26.0 - 2.0 * ui.spacing().item_spacing.x).max(60.0);
                let hint = preset.map_or("", |p| p.model);
                changed |= ui.add(egui::TextEdit::singleline(&mut cur.model).hint_text(hint).desired_width(w)).changed();
                let drop = ui.add_enabled(!list.is_empty(), egui::Button::new("▾").min_size(vec2(26.0, 22.0)));
                let drop = drop.on_hover_text(tl!("Choose from the provider's models"));
                egui::Popup::menu(&drop).show(|ui| {
                    ui.set_width(300.0);
                    ui.add(egui::TextEdit::singleline(&mut pane.models.filter).hint_text(tl!("Filter")).desired_width(f32::INFINITY));
                    let q = pane.models.filter.trim().to_lowercase();
                    let shown: Vec<String> = list.iter().filter(|m| q.is_empty() || m.to_lowercase().contains(&q)).cloned().collect();
                    if shown.is_empty() {
                        ui.label(RichText::new(tl!("No matching models")).color(t.text_dim));
                    } else {
                        // Long ids are cut (the full id is the tooltip); rows stay one line so only the
                        // ones in view are built.
                        let row_h = ui.spacing().interact_size.y;
                        egui::ScrollArea::vertical().max_height(260.0).show_rows(ui, row_h, shown.len(), |ui, range| {
                            for m in shown.get(range).unwrap_or_default() {
                                let r = ui.add_sized([ui.available_width(), row_h], egui::Button::selectable(*m == cur.model, m.as_str()).truncate());
                                if r.on_hover_text(m.as_str()).clicked() {
                                    pick = Some(m.clone());
                                    ui.close();
                                }
                            }
                        });
                    }
                });
                refresh = crate::widgets::icon_button(ui, "update", tl!("Refresh the model list")).clicked();
            });
            if let Some(m) = pick {
                cur.model = m;
                changed = true;
            }
            if let Some((is_error, text)) = status {
                let color = if is_error { t.red } else { t.text_dim };
                ui.add(egui::Label::new(RichText::new(text).font(regular(11.0)).color(color)).wrap());
            }
            let url_hint = preset.map(|p| p.base_url).filter(|u| !u.is_empty()).unwrap_or("https://");
            changed |= field(ui, if custom { "Base URL (required)" } else { "Base URL" }, &mut cur.base_url, url_hint, false);
            if custom {
                ui.label(tl!("Protocol"));
                let now = cur.protocol.unwrap_or(Protocol::OpenAi);
                ui.horizontal(|ui| {
                    for (p, label) in [(Protocol::OpenAi, "OpenAI-compatible"), (Protocol::Anthropic, "Anthropic")] {
                        if ui.radio(now == p, tl!(label)).clicked() && now != p {
                            cur.protocol = Some(p);
                            changed = true;
                        }
                    }
                });
            }
            pane.settings_changed |= changed;
            ui.add_space(2.0);
            crate::rewrite_popover::settings_ui(pane, ui);
            refresh
        })
        .body_returned
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    use serde_json::json;
    use wordcraft_assistant::ProviderSettings;

    use crate::{Services, WordApp};

    const KEY: &str = "sk-test-secret-0123";

    /// An app whose HTTP client answers with `replies` in order, as an Anthropic server would.
    fn app_with(replies: &[&str]) -> WordApp {
        let queue: Rc<RefCell<VecDeque<String>>> = Rc::new(RefCell::new(replies.iter().map(|s| s.to_string()).collect()));
        let http = move |_req: wordcraft_assistant::HttpRequest| {
            let (tx, rx) = std::sync::mpsc::channel();
            let _ = tx.send(queue.borrow_mut().pop_front().ok_or_else(|| "no more replies".to_string()));
            rx
        };
        let services = Services { http: Some(Box::new(http)), ..Default::default() };
        let mut app = WordApp::new(wordcraft_engine::Session::new(wordcraft_doc::Document::new()), services);
        app.assistant.settings.provider = "claude".into();
        app.assistant.settings.providers.insert("claude".into(), ProviderSettings { api_key: KEY.into(), ..Default::default() });
        app
    }

    fn frames(app: &mut WordApp, n: usize) {
        let ctx = egui::Context::default();
        for _ in 0..n {
            ctx.run_ui(egui::RawInput::default(), |ui| app.logic(ui.ctx())).drop_without_applying_deltas();
        }
    }

    #[test]
    fn ui_assistant_toggles_the_pane() {
        let mut a = WordApp::new(wordcraft_engine::Session::new(wordcraft_doc::Document::new()), Services::default());
        assert_eq!(a.run("ui.assistant", json!({})).unwrap()["open"], true);
        assert_eq!(a.run("ui.assistant", json!({})).unwrap()["open"], false);
        assert_eq!(a.run("ui.assistant", json!({"open": true})).unwrap()["open"], true);
        assert_eq!(a.run("ui.assistant", json!({"open": true})).unwrap()["open"], true);
        assert_eq!(a.run("ui.assistant", json!({"stop": true})).unwrap()["open"], true, "stop leaves the pane as it is");
        // No HTTP client (the web): asking is an error, not a crash.
        assert!(a.run("ui.assistant", json!({"ask": "Hello"})).is_err());
        for junk in [json!({"open": 3, "ask": 5, "stop": "x", "clear": null}), json!([1, 2]), json!(null), json!({"ask": ""})] {
            let _ = a.run("ui.assistant", junk);
        }
    }

    #[test]
    fn asking_drives_the_agent_to_an_edit_through_logic_frames() {
        let mut a = app_with(&[
            r#"{"content":[{"type":"tool_use","id":"t0","name":"get_text","input":{}},{"type":"tool_use","id":"t1","name":"type_text","input":{"text":"Hi"}}]}"#,
            r#"{"content":[{"type":"tool_use","id":"t2","name":"get_text","input":{}}]}"#,
            r#"{"content":[{"type":"text","text":"Done"}]}"#,
        ]);
        let r = a.run("ui.assistant", json!({"ask": "Say hi"})).unwrap();
        assert_eq!((r["open"].clone(), r["busy"].clone()), (json!(true), json!(true)));
        frames(&mut a, 6);
        let r = a.run("ui.assistant", json!({"open": true})).unwrap();
        assert_eq!(r["busy"], false);
        assert!(a.session.doc.plain_text(wordcraft_doc::StoryRef::Body).contains("Hi"));
        let kinds: Vec<&str> = r["log"].as_array().unwrap().iter().map(|e| e["kind"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["user", "action", "action", "action", "assistant"]);
        assert_eq!(r["log"][2]["ok"], true);
        assert_eq!(r["log"][4]["text"], "Done");
        let all = r.to_string();
        assert!(!all.contains(KEY) && !all.contains("apiKey") && !all.contains("claude"), "no settings in {all}");
        // New chat.
        assert_eq!(a.run("ui.assistant", json!({"clear": true})).unwrap()["log"], json!([]));
    }

    #[test]
    fn a_key_echoed_in_an_error_is_blanked() {
        let mut a = app_with(&[]);
        a.assistant.agent.log.push(wordcraft_assistant::Entry::Error(format!("invalid key {KEY}")));
        let r = a.run("ui.assistant", json!({"open": true})).unwrap();
        assert!(!r.to_string().contains(KEY));
    }

    #[test]
    fn prefs_keep_the_users_name_while_a_request_runs() {
        let mut a = app_with(&[r#"{"content":[{"type":"text","text":"Done"}]}"#]);
        a.session.author = "Ana Writer".into();
        a.run("ui.assistant", json!({"ask": "Hello"})).unwrap();
        assert_ne!(a.session.author, "Ana Writer", "edits are by the AI while it runs");
        assert_eq!(a.prefs().author.as_deref(), Some("Ana Writer"));
        frames(&mut a, 3);
        assert!(!a.assistant.agent.busy());
        assert_eq!(a.session.author, "Ana Writer");
        assert_eq!(a.prefs().author.as_deref(), Some("Ana Writer"));
    }

    #[test]
    fn the_pane_draws_with_a_log() {
        let mut a = app_with(&[]);
        a.assistant.open = true;
        a.assistant.agent.log = vec![
            wordcraft_assistant::Entry::User("Make it bold".into()),
            wordcraft_assistant::Entry::Action { label: "Run", detail: "format.bold".into(), ok: true },
            wordcraft_assistant::Entry::Action { label: "Select", detail: "nope".into(), ok: false },
            wordcraft_assistant::Entry::Assistant("Done.".into()),
            wordcraft_assistant::Entry::Error("Oops".into()),
        ];
        let ctx = egui::Context::default();
        for _ in 0..4 {
            ctx.run_ui(egui::RawInput::default(), |ui| {
                a.logic(ui.ctx());
                a.ui(ui);
            })
            .drop_without_applying_deltas();
        }
        assert!(a.assistant.open);
    }

    /// Frames that draw the pane too (the cards and the checklist).
    fn draw(app: &mut WordApp) {
        let ctx = egui::Context::default();
        crate::theme::install_fonts(&ctx);
        for _ in 0..3 {
            ctx.run_ui(egui::RawInput::default(), |ui| {
                app.logic(ui.ctx());
                app.ui(ui);
            })
            .drop_without_applying_deltas();
        }
    }

    fn doc_text(a: &mut WordApp) -> String {
        a.session.run("document.text", &json!({})).unwrap()["text"].as_str().unwrap().trim().to_string()
    }

    const PARA: &str = "This paragraph is long enough that removing it asks first.";

    #[test]
    fn the_plan_shows_in_ui_assistant() {
        let mut a = app_with(&[
            r#"{"content":[{"type":"tool_use","id":"p","name":"plan","input":{"steps":[{"text":"Read the document","done":true},{"text":"Bold the title","done":false}]}}]}"#,
            r#"{"content":[{"type":"text","text":"Done"}]}"#,
        ]);
        a.run("ui.assistant", json!({"ask": "Bold the title"})).unwrap();
        draw(&mut a);
        let r = a.run("ui.assistant", json!({"open": true})).unwrap();
        assert_eq!(r["busy"], false);
        assert_eq!(r["pending"], json!(null));
        assert_eq!(r["plan"], json!([{"text": "Read the document", "done": true}, {"text": "Bold the title", "done": false}]));
    }

    #[test]
    fn a_question_pauses_until_answered() {
        let mut a = app_with(&[
            r#"{"content":[{"type":"tool_use","id":"q","name":"ask_user","input":{"question":"Which tone?","options":["Option A","Option B"]}}]}"#,
            r#"{"content":[{"type":"text","text":"Done"}]}"#,
        ]);
        a.run("ui.assistant", json!({"ask": "Rewrite it"})).unwrap();
        draw(&mut a);
        let r = a.run("ui.assistant", json!({"open": true})).unwrap();
        assert_eq!(r["busy"], true, "paused, not finished");
        assert_eq!(r["pending"], json!({"kind": "question", "question": "Which tone?", "options": ["Option A", "Option B"]}));
        assert!(a.run("ui.assistant", json!({"decide": "deny"})).is_err(), "no permission prompt waits");
        let r = a.run("ui.assistant", json!({"answer": "Option B"})).unwrap();
        assert_eq!(r["pending"], json!(null));
        draw(&mut a);
        let r = a.run("ui.assistant", json!({"open": true})).unwrap();
        assert_eq!(r["busy"], false);
        let kinds: Vec<&str> = r["log"].as_array().unwrap().iter().map(|e| e["kind"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["user", "question", "user", "assistant"]);
        assert_eq!(r["log"][2]["text"], "Option B");
        assert!(a.run("ui.assistant", json!({"answer": "late"})).is_err(), "nothing waits any more");
    }

    /// Ask to delete the paragraph; the model reads, selects it whole and deletes it.
    fn deleting() -> WordApp {
        let select = format!(
            r#"{{"content":[{{"type":"tool_use","id":"a","name":"get_text","input":{{}}}},{{"type":"tool_use","id":"b","name":"select_text","input":{{"text":"{PARA}"}}}},{{"type":"tool_use","id":"c","name":"execute","input":{{"command":"text.delete","params":{{}}}}}}]}}"#
        );
        let done = r#"{"content":[{"type":"text","text":"Done"}]}"#;
        let mut a = app_with(&[&select, done, done, done]);
        a.session.run("text.insert", &json!({"text": PARA})).unwrap();
        a.run("ui.assistant", json!({"ask": "Delete the paragraph"})).unwrap();
        draw(&mut a);
        let r = a.run("ui.assistant", json!({"open": true})).unwrap();
        assert_eq!(r["busy"], true);
        assert_eq!(r["pending"]["kind"], "permission", "{r}");
        assert_eq!(r["pending"]["permissionKind"], "delete");
        assert!(r["pending"]["description"].as_str().unwrap().starts_with("Delete"));
        assert_eq!(doc_text(&mut a), PARA, "nothing removed while it asks");
        assert!(a.run("ui.assistant", json!({"answer": "yes"})).is_err(), "no question waits");
        assert!(a.run("ui.assistant", json!({"decide": "maybe"})).is_err());
        a
    }

    #[test]
    fn denying_a_deletion_keeps_the_text() {
        let mut a = deleting();
        a.run("ui.assistant", json!({"decide": "deny"})).unwrap();
        draw(&mut a);
        let r = a.run("ui.assistant", json!({"open": true})).unwrap();
        assert_eq!(r["busy"], false);
        a.session.run("review.acceptAll", &json!({})).unwrap();
        assert_eq!(doc_text(&mut a), PARA, "no deletion to accept");
        let perm = r["log"].as_array().unwrap().iter().find(|e| e["kind"] == "permission").unwrap();
        assert_eq!((perm["ok"].clone(), perm["permissionKind"].clone()), (json!(false), json!("delete")));
    }

    #[test]
    fn allowing_a_deletion_removes_the_text_as_a_tracked_change() {
        let mut a = deleting();
        a.run("ui.assistant", json!({"decide": "allowOnce"})).unwrap();
        draw(&mut a);
        let r = a.run("ui.assistant", json!({"open": true})).unwrap();
        assert_eq!(r["busy"], false, "{r}");
        // document.text reads the final text (upstream); the tracked deletion is still in the plain text.
        assert_eq!(doc_text(&mut a), "", "deleted in the final text");
        assert_eq!(a.session.doc.plain_text(wordcraft_doc::StoryRef::Body).trim(), PARA, "still there as a tracked deletion");
        a.session.run("review.acceptAll", &json!({})).unwrap();
        assert_eq!(doc_text(&mut a), "", "accepting the change removes it");
    }

    type Sent = Rc<RefCell<Vec<(wordcraft_assistant::Method, String, Vec<(String, String)>)>>>;

    /// An app whose HTTP client answers every request with `body` and records what was asked.
    fn app_listing(body: &str, key: &str) -> (WordApp, Sent) {
        let sent: Sent = Rc::default();
        let (log, body) = (sent.clone(), body.to_string());
        let http = move |req: wordcraft_assistant::HttpRequest| {
            log.borrow_mut().push((req.method, req.url, req.headers));
            let (tx, rx) = std::sync::mpsc::channel();
            let _ = tx.send(Ok(body.clone()));
            rx
        };
        let services = Services { http: Some(Box::new(http)), ..Default::default() };
        let mut app = WordApp::new(wordcraft_engine::Session::new(wordcraft_doc::Document::new()), services);
        app.assistant.settings.provider = "claude".into();
        app.assistant.settings.providers.insert("claude".into(), ProviderSettings { api_key: key.into(), ..Default::default() });
        app.assistant.open = true;
        app.assistant.settings_open = true;
        (app, sent)
    }

    const LIST: &str = r#"{"data":[{"id":"claude-b"},{"id":"claude-a"}]}"#;

    #[test]
    fn opening_settings_fetches_the_models_once() {
        let (mut a, sent) = app_listing(LIST, KEY);
        draw(&mut a);
        draw(&mut a);
        {
            let s = sent.borrow();
            assert_eq!(s.len(), 1, "one fetch, then cached");
            assert_eq!(s[0].0, wordcraft_assistant::Method::Get);
            assert_eq!(s[0].1, "https://api.anthropic.com/v1/models?limit=1000");
            assert!(s[0].2.contains(&("x-api-key".to_string(), KEY.to_string())));
        }
        let r = a.run("ui.assistant", json!({"models": true})).unwrap();
        assert_eq!(r["models"], json!({"loading": false, "list": ["claude-a", "claude-b"], "error": null}));
        assert!(!r.to_string().contains(KEY));
        assert_eq!(sent.borrow().len(), 1, "the control call reused the list");
        // A new key makes the list stale.
        a.assistant.settings.providers.get_mut("claude").unwrap().api_key = "sk-other-key-9".into();
        draw(&mut a);
        assert_eq!(sent.borrow().len(), 2);
    }

    #[test]
    fn no_fetch_without_a_key() {
        let (mut a, sent) = app_listing(LIST, "");
        draw(&mut a);
        let r = a.run("ui.assistant", json!({"models": true})).unwrap();
        assert!(sent.borrow().is_empty());
        assert_eq!(r["models"]["list"], json!([]));
    }

    #[test]
    fn a_models_error_hides_the_key() {
        let body = format!(r#"{{"error":{{"message":"bad key {KEY}"}}}}"#);
        let (mut a, _) = app_listing(&body, KEY);
        let r = a.run("ui.assistant", json!({"models": true})).unwrap();
        assert_eq!(r["models"]["list"], json!([]));
        let e = r["models"]["error"].as_str().unwrap();
        assert!(e.contains("bad key") && !e.contains(KEY), "{e}");
        draw(&mut a);
    }

    #[test]
    fn picking_a_model_sets_it() {
        use egui_kittest::kittest::Queryable;
        let (mut a, _) = app_listing(LIST, KEY);
        a.assistant.settings_open = true;
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1440.0, 900.0)).build_ui_state(
            |ui, app: &mut WordApp| {
                let ctx = ui.ctx().clone();
                app.logic(&ctx);
                app.ui(ui);
            },
            a,
        );
        h.run_steps(4);
        h.get_by_label("▾").click();
        h.run_steps(4);
        h.get_by_label("claude-b").click();
        h.run_steps(4);
        let a = h.state();
        assert_eq!(a.assistant.settings.providers["claude"].model, "claude-b");
        assert!(a.assistant.settings_changed);
    }
}

/// The pane title with a close button (upstream dropped the shared helper when panes moved to craft_ui).
fn header(ui: &mut Ui, title: &str) -> bool {
    let mut close = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(tl!(title)).font(semibold(15.0)));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if crate::widgets::icon_button(ui, "close", tl!("Close")).clicked() {
                close = true;
            }
        });
    });
    ui.add_space(4.0);
    close
}
