//! Prompt on selection (`ai.rewriteSelection`): a small popover at the selection's end with a
//! preset list, a free prompt (Enter sends) and a scope for this one request. The answer is typed
//! over the selection as a tracked change by "AI (model)", one undo step
//! ([`wordcraft_assistant::rewrite::apply`]). If the document or the selection changed while the
//! model was working, the popover asks before applying. Settings live with the assistant's
//! (`assistant.json`); their editor is [`settings_ui`].

use egui::{Align, Layout, Order, Pos2, RichText, Ui, vec2};
use serde_json::{Value, json};
use wordcraft_assistant::rewrite::{self, Job, Outcome, RewritePreset, Scope};

use crate::WordApp;
use crate::assistant::AssistantPane;
use crate::theme::{Tokens, regular, semibold};

/// Longest a headless (control port) request waits; below the control channel's 60 s.
const HEADLESS_WAIT_S: u64 = 55;

#[derive(Default)]
pub struct RewritePopover {
    pub open: bool,
    pub prompt: String,
    /// The scope for the next request (`None`: the settings').
    pub scope: Option<Scope>,
    /// Where the popover sits (screen), taken when it opens.
    anchor: Option<Pos2>,
    job: Option<Job>,
    /// An answer that arrived after the selection changed: (text, model, snapshot when sent).
    conflict: Option<(String, String)>,
    pub error: Option<String>,
    /// Focus the prompt on the next frame.
    focus: bool,
}

impl RewritePopover {
    pub fn busy(&self) -> bool {
        self.job.is_some()
    }
}

/// The command: with `prompt`, run headless (block until the answer, apply, report); without,
/// open the popover. `{"status": true}` reports the popover's state.
pub fn command(app: &mut WordApp, p: &Value) -> Result<Value, String> {
    if p.get("status").and_then(Value::as_bool) == Some(true) {
        let r = &app.assistant.rewrite;
        return Ok(json!({"open": r.open, "busy": r.busy(), "error": r.error, "conflict": r.conflict.is_some()}));
    }
    let scope = match p.get("scope").and_then(Value::as_str) {
        Some(s) => Some(Scope::parse(s).ok_or("`scope` must be whole, section, paragraphs or selectionOnly")?),
        None => None,
    };
    match p.get("prompt").and_then(Value::as_str) {
        Some(prompt) => headless(app, prompt, scope),
        None => {
            open(app, scope);
            Ok(json!({"open": true}))
        }
    }
}

fn open(app: &mut WordApp, scope: Option<Scope>) {
    let anchor = crate::canvas::caret_screen(app).map(|(p, h)| p + vec2(0.0, h + 4.0));
    let r = &mut app.assistant.rewrite;
    r.open = true;
    r.anchor = anchor;
    r.scope = scope;
    r.error = None;
    r.conflict = None;
    r.focus = true;
}

fn close(app: &mut WordApp) {
    let r = &mut app.assistant.rewrite;
    r.open = false;
    r.job = None;
    r.conflict = None;
    r.error = None;
    app.canvas.want_focus = true;
}

fn headless(app: &mut WordApp, prompt: &str, scope: Option<Scope>) -> Result<Value, String> {
    let Some(http) = app.services.http.as_ref() else { return Err("The assistant isn't available here.".into()) };
    let settings = &app.assistant.settings;
    let scope = scope.unwrap_or(settings.rewrite.scope);
    let t0 = crate::clock::now_ms();
    let job = rewrite::start(settings, scope, prompt, &app.session, http, t0)?;
    let text = match job.wait(std::time::Duration::from_secs(HEADLESS_WAIT_S)) {
        Outcome::Answer(t) => t,
        Outcome::Failed(e) => return Err(e),
        Outcome::Waiting => return Err("Modell nicht erreichbar: keine Antwort.".into()),
    };
    let ms = (crate::clock::now_ms() - t0).round() as u64;
    if !job.snapshot.holds(&app.session) {
        return Ok(json!({"applied": false, "reason": "selection changed", "ms": ms, "model": job.model}));
    }
    let chars = rewrite::apply(&mut app.session, &text, &job.model)?;
    Ok(json!({"applied": true, "chars": chars, "ms": ms, "model": job.model, "scope": scope.id(), "text": text}))
}

fn send(app: &mut WordApp) {
    let r = &app.assistant.rewrite;
    let prompt = r.prompt.trim().to_string();
    if prompt.is_empty() || r.busy() {
        return;
    }
    let Some(http) = app.services.http.as_ref() else {
        app.assistant.rewrite.error = Some("Der Assistent ist hier nicht verfügbar.".into());
        return;
    };
    let scope = r.scope.unwrap_or(app.assistant.settings.rewrite.scope);
    match rewrite::start(&app.assistant.settings, scope, &prompt, &app.session, http, crate::clock::now_ms()) {
        Ok(job) => {
            app.assistant.rewrite.error = None;
            app.assistant.rewrite.job = Some(job);
        }
        Err(e) => app.assistant.rewrite.error = Some(e),
    }
}

/// Every frame: take the answer when it has arrived.
pub fn poll(app: &mut WordApp, ctx: &egui::Context) {
    let Some(job) = app.assistant.rewrite.job.as_ref() else { return };
    match job.poll() {
        Outcome::Waiting => ctx.request_repaint_after(std::time::Duration::from_millis(200)),
        Outcome::Failed(e) => {
            app.assistant.rewrite.job = None;
            app.assistant.rewrite.error = Some(e);
            ctx.request_repaint();
        }
        Outcome::Answer(text) => {
            let (model, holds) = (job.model.clone(), job.snapshot.holds(&app.session));
            app.assistant.rewrite.job = None;
            if holds {
                match rewrite::apply(&mut app.session, &text, &model) {
                    Ok(_) => close(app),
                    Err(e) => app.assistant.rewrite.error = Some(e),
                }
            } else {
                app.assistant.rewrite.conflict = Some((text, model));
            }
            ctx.request_repaint();
        }
    }
}

/// The popover, drawn over the page.
pub fn show(app: &mut WordApp, ctx: &egui::Context) {
    if !app.assistant.rewrite.open {
        return;
    }
    let t = Tokens::get(ctx);
    let at = app.assistant.rewrite.anchor.unwrap_or_else(|| ctx.content_rect().center() - vec2(180.0, 80.0));
    let mut action = None;
    egui::Area::new(egui::Id::unique("wc_rewrite_popover")).order(Order::Foreground).fixed_pos(at).constrain(true).show(ctx, |ui| {
        egui::Frame::popup(ui.style()).show(ui, |ui| {
            ui.set_width(380.0);
            action = body(app, ui, &t);
        });
    });
    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) && !app.assistant.rewrite.busy() {
        action = Some(Action::Close);
    }
    match action {
        Some(Action::Send) => send(app),
        Some(Action::Stop) => app.assistant.rewrite.job = None,
        Some(Action::Close) => close(app),
        Some(Action::ApplyAnyway) => {
            if let Some((text, model)) = app.assistant.rewrite.conflict.take() {
                match rewrite::apply(&mut app.session, &text, &model) {
                    Ok(_) => close(app),
                    Err(e) => app.assistant.rewrite.error = Some(e),
                }
            }
        }
        Some(Action::Discard) => app.assistant.rewrite.conflict = None,
        None => {}
    }
}

enum Action {
    Send,
    Stop,
    Close,
    ApplyAnyway,
    Discard,
}

fn body(app: &mut WordApp, ui: &mut Ui, t: &Tokens) -> Option<Action> {
    let mut action = None;
    ui.horizontal(|ui| {
        ui.label(RichText::new("Mit KI bearbeiten").font(semibold(13.5)));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if crate::widgets::icon_button(ui, "close", "Schließen").clicked() {
                action = Some(Action::Close);
            }
        });
    });
    let model = app.assistant.settings.endpoint().map(|e| e.model).unwrap_or_default();
    let r = &mut app.assistant.rewrite;
    if let Some(job) = r.job.as_ref() {
        let secs = ((crate::clock::now_ms() - job.started_ms) / 1000.0).max(0.0);
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(format!("{model} arbeitet … {secs:.0} s"));
            if ui.button("Stopp").clicked() {
                action = Some(Action::Stop);
            }
        });
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
        return action;
    }
    if r.conflict.is_some() {
        ui.label(RichText::new("Auswahl geändert").color(t.red));
        ui.horizontal(|ui| {
            if ui.button("Trotzdem anwenden").clicked() {
                action = Some(Action::ApplyAnyway);
            }
            if ui.button("Verwerfen").clicked() {
                action = Some(Action::Discard);
            }
        });
        return action;
    }
    let settings = &app.assistant.settings.rewrite;
    ui.horizontal(|ui| {
        craft_ui::ComboBox::from_id_salt("wc_rewrite_preset").selected_text("Vorlage").width(150.0).show_ui(ui, |ui| {
            for p in &settings.presets {
                if ui.selectable_label(false, &p.name).on_hover_text(&p.prompt).clicked() {
                    r.prompt = p.prompt.clone();
                    r.focus = true;
                }
            }
        });
        let scope = r.scope.unwrap_or(settings.scope);
        craft_ui::ComboBox::from_id_salt("wc_rewrite_scope").selected_text(scope.label()).width(150.0).show_ui(ui, |ui| {
            for s in Scope::ALL {
                if ui.selectable_label(s == scope, s.label()).clicked() {
                    r.scope = Some(s);
                }
            }
        });
    });
    let edit = ui.add(egui::TextEdit::singleline(&mut r.prompt).hint_text("Anweisung, z. B. „Kürzen“ — Enter sendet").desired_width(f32::INFINITY));
    if std::mem::take(&mut r.focus) {
        edit.request_focus();
    }
    let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
    ui.horizontal(|ui| {
        if ui.add_enabled(!r.prompt.trim().is_empty(), egui::Button::new("Senden")).clicked() || enter {
            action = Some(Action::Send);
        }
        ui.label(RichText::new(model).font(regular(11.0)).color(t.text_dim));
    });
    if let Some(e) = &r.error {
        ui.add(egui::Label::new(RichText::new(e).color(t.red)).wrap());
    }
    action
}

/// The "Mit KI bearbeiten" settings, under the assistant's provider settings (same endpoint).
pub fn settings_ui(pane: &mut AssistantPane, ui: &mut Ui) {
    let mut changed = false;
    egui::CollapsingHeader::new(RichText::new("Mit KI bearbeiten").font(semibold(12.5))).id_salt("rewrite_settings").show(ui, |ui| {
        let s = &mut pane.settings.rewrite;
        ui.label("Kontext");
        craft_ui::ComboBox::from_id_salt("rewrite_scope_setting").selected_text(s.scope.label()).width(ui.available_width()).show_ui(ui, |ui| {
            for x in Scope::ALL {
                if ui.selectable_label(x == s.scope, x.label()).clicked() && x != s.scope {
                    s.scope = x;
                    changed = true;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Absätze ringsum");
            changed |= ui.add(egui::DragValue::new(&mut s.scope_paragraphs).range(0..=50)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Höchstens Zeichen Kontext");
            changed |= ui.add(egui::DragValue::new(&mut s.max_context_chars).range(500..=1_000_000).speed(100)).changed();
        });
        ui.label("Systemprompt");
        changed |= ui.add(egui::TextEdit::multiline(&mut s.system_prompt).desired_rows(4).desired_width(f32::INFINITY)).changed();
        if ui.small_button("Standard wiederherstellen").clicked() {
            s.system_prompt = rewrite::DEFAULT_SYSTEM_PROMPT.into();
            changed = true;
        }
        ui.label("Vorlagen");
        let mut remove = None;
        for (i, p) in s.presets.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                changed |= ui.add(egui::TextEdit::singleline(&mut p.name).desired_width(110.0)).changed();
                let w = (ui.available_width() - 28.0).max(60.0);
                changed |= ui.add(egui::TextEdit::singleline(&mut p.prompt).desired_width(w)).changed();
                if ui.small_button("−").on_hover_text("Vorlage entfernen").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            s.presets.remove(i);
            changed = true;
        }
        if ui.small_button("+ Vorlage").clicked() {
            s.presets.push(RewritePreset { name: "Neu".into(), prompt: String::new() });
            changed = true;
        }
    });
    pane.settings_changed |= changed;
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use serde_json::json;
    use wordcraft_assistant::ProviderSettings;
    use wordcraft_assistant::providers::CUSTOM;

    use crate::{Services, WordApp};

    /// An app whose HTTP client answers every request with `reply` (OpenAI-compatible), counting them.
    fn app_with(reply: Result<&str, &str>) -> (WordApp, Rc<RefCell<Vec<String>>>) {
        let sent: Rc<RefCell<Vec<String>>> = Rc::default();
        let (log, reply) = (sent.clone(), reply.map(str::to_string).map_err(str::to_string));
        let http = move |req: wordcraft_assistant::HttpRequest| {
            log.borrow_mut().push(req.body.clone());
            let (tx, rx) = std::sync::mpsc::channel();
            let body = reply.clone().map(|t| json!({"choices": [{"message": {"content": t}}]}).to_string());
            let _ = tx.send(body);
            rx
        };
        let services = Services { http: Some(Box::new(http)), ..Default::default() };
        let mut app = WordApp::new(wordcraft_engine::Session::new(wordcraft_doc::Document::from_text("Kopf\nDer Absatz ist viel zu lang und umständlich.")), services);
        app.assistant.settings.provider = CUSTOM.into();
        app.assistant
            .settings
            .providers
            .insert(CUSTOM.into(), ProviderSettings { base_url: "http://127.0.0.1:8080/v1".into(), model: "qwen".into(), ..Default::default() });
        app.session.author = "Philip".into();
        (app, sent)
    }

    fn select_second(app: &mut WordApp) {
        app.session.run("caret.docEnd", &json!({})).ok();
        app.session.run("select.paragraph", &json!({})).unwrap();
    }

    fn frames(app: &mut WordApp, n: usize) {
        let ctx = egui::Context::default();
        crate::theme::install_fonts(&ctx);
        for _ in 0..n {
            ctx.run_ui(egui::RawInput::default(), |ui| {
                app.logic(ui.ctx());
                app.ui(ui);
            })
            .drop_without_applying_deltas();
        }
    }

    fn final_text(app: &WordApp) -> String {
        app.session.doc.final_text(wordcraft_doc::StoryRef::Body).trim_end().to_string()
    }

    #[test]
    fn the_command_is_registered_with_a_free_shortcut() {
        let reg = wordcraft_engine::cmd::registry();
        let spec = reg.get("ai.rewriteSelection").unwrap();
        assert_eq!(spec.shortcut, "Mod+Shift+I");
        assert_eq!(reg.by_shortcut("Mod+Shift+I").map(|c| c.id), Some("ai.rewriteSelection"));
        assert_eq!(reg.by_shortcut("Mod+Shift+E").map(|c| c.id), Some("review.trackChanges"), "Track Changes keeps its key");
    }

    #[test]
    fn headless_rewrite_is_one_tracked_undo_step_by_the_model() {
        let (mut a, sent) = app_with(Ok("„Der Absatz ist kurz.“"));
        select_second(&mut a);
        let depth = a.session.undo_depth();
        let r = a.run("ai.rewriteSelection", json!({"prompt": "Kürzen", "scope": "section"})).unwrap();
        assert_eq!((r["applied"].clone(), r["model"].clone(), r["scope"].clone()), (json!(true), json!("qwen"), json!("section")), "{r}");
        assert_eq!(final_text(&a), "Kopf\nDer Absatz ist kurz.");
        assert_eq!(a.session.undo_depth(), depth + 1);
        assert_eq!(a.session.author, "Philip");
        let body: serde_json::Value = serde_json::from_str(&sent.borrow()[0]).unwrap();
        let user = body["messages"][1]["content"].as_str().unwrap();
        assert!(user.contains("<selection>\nDer Absatz ist viel zu lang und umständlich.\n</selection>"), "{user}");
        assert!(user.ends_with("Kürze den Text deutlich, ohne wesentliche Information zu verlieren."), "the preset's prompt: {user}");
        assert!(body.get("tools").is_none());
        let changes = a.session.run("review.changes", &json!({})).unwrap().to_string();
        assert!(changes.contains("AI (qwen)") && !changes.contains("Philip"), "{changes}");
        assert!(a.session.undo());
        assert_eq!(final_text(&a), "Kopf\nDer Absatz ist viel zu lang und umständlich.");
    }

    #[test]
    fn an_unreachable_model_is_an_error_not_a_hang() {
        let (mut a, _) = app_with(Err("network error: connection failed"));
        select_second(&mut a);
        let e = a.run("ai.rewriteSelection", json!({"prompt": "Kürzen"})).unwrap_err();
        assert!(e.starts_with("Modell nicht erreichbar"), "{e}");
        assert_eq!(final_text(&a), "Kopf\nDer Absatz ist viel zu lang und umständlich.");
        assert!(a.run("ai.rewriteSelection", json!({"prompt": "x", "scope": "nope"})).is_err());
    }

    #[test]
    fn the_popover_sends_applies_and_closes() {
        let (mut a, _) = app_with(Ok("Kurz."));
        select_second(&mut a);
        a.run("ai.rewriteSelection", json!({})).unwrap();
        assert!(a.assistant.rewrite.open);
        frames(&mut a, 2);
        a.assistant.rewrite.prompt = "Kürzen".into();
        super::send(&mut a);
        frames(&mut a, 2);
        assert!(!a.assistant.rewrite.open, "closed after applying: {:?}", a.assistant.rewrite.error);
        assert_eq!(final_text(&a), "Kopf\nKurz.");
    }

    #[test]
    fn a_changed_selection_asks_before_applying() {
        let (mut a, _) = app_with(Ok("Kurz."));
        select_second(&mut a);
        a.run("ai.rewriteSelection", json!({})).unwrap();
        a.assistant.rewrite.prompt = "Kürzen".into();
        super::send(&mut a);
        a.session.run("caret.docStart", &json!({})).unwrap();
        frames(&mut a, 2);
        assert!(a.assistant.rewrite.conflict.is_some());
        assert_eq!(a.run("ai.rewriteSelection", json!({"status": true})).unwrap()["conflict"], true);
        assert_eq!(final_text(&a), "Kopf\nDer Absatz ist viel zu lang und umständlich.", "nothing applied yet");
    }
}
