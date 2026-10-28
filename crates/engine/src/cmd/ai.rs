//! `ai.rewriteSelection` (prompt on selection). The app runs it: it needs the model's endpoint
//! and the network, which the engine doesn't have. It is registered here so the keymap, the
//! context menu, macros, `engine.commands` and agents all find it.

use serde_json::Value;

use crate::{CmdError, CmdResult, CommandSpec, Session};

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec::new("ai.rewriteSelection", "Rewrite with AI", "Home › Assistant", run)
            .key("Mod+Shift+I")
            .params(
                r#"{"prompt"?: string (a preset name such as "Shorten", or an instruction; omitted, the app opens its popover), "scope"?: "whole"|"section"|"paragraphs"|"selectionOnly"} → {applied, chars, ms, model}"#,
            )
            .when(needs_selection),
    ]
}

fn needs_selection(s: &Session) -> Option<&'static str> {
    if s.sel.is_collapsed() { Some("select some text first") } else { None }
}

fn run(_: &mut Session, _: &Value) -> CmdResult {
    Err(CmdError::Failed("Rewrite with AI runs in the WordCraft app (it needs the model's endpoint).".into()))
}
