//! Fork-only (Philip's AI assistant): `assistant.json` load/save, kept out of `main.rs` so upstream
//! merges touch only the few call lines there. Uses `main.rs`'s `prefs_path` / `prefs_enabled`.
use crate::{prefs_enabled, prefs_path};
use wordcraft_ui_egui::WordApp;

/// `assistant.json` next to the preferences: provider, API keys, models, and the
/// prompt-on-selection settings. Not in `ui.json`, which is shared with the MCP bridge and easy to
/// attach to a bug report.
fn assistant_settings_path() -> Option<std::path::PathBuf> {
    Some(prefs_path()?.parent()?.join("assistant.json"))
}

pub(crate) fn load_assistant_settings(app: &mut WordApp) {
    if !prefs_enabled() {
        return;
    }
    if let Some(p) = assistant_settings_path()
        && let Some(s) = read_settings(&p)
    {
        app.assistant.settings = s;
        // German defaults saved by an earlier build become the English ones (edits are kept).
        app.assistant.settings_changed |= app.assistant.settings.rewrite.upgrade_old_defaults();
    }
}

pub(crate) fn save_assistant_settings(app: &WordApp) {
    if prefs_enabled()
        && let Some(p) = assistant_settings_path()
    {
        write_settings(&p, &app.assistant.settings);
    }
}

fn read_settings(p: &std::path::Path) -> Option<wordcraft_assistant::Settings> {
    serde_json::from_slice(&std::fs::read(p).ok()?).ok()
}

/// Best effort, like the preferences. Owner-only on Unix (it holds API keys); on Windows the
/// per-user AppData folder is private enough. Never logs the contents.
fn write_settings(p: &std::path::Path, s: &wordcraft_assistant::Settings) {
    use std::io::Write;
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let Ok(bytes) = serde_json::to_vec_pretty(s) else { return };
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    if let Ok(mut f) = o.open(p) {
        let _ = f.write_all(&bytes);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn settings_round_trip() {
        let dir = std::env::temp_dir().join(format!("wordcraft-as-{}", std::process::id()));
        let p = dir.join("sub").join("assistant.json");
        let mut s = wordcraft_assistant::Settings::default();
        super::write_settings(&p, &s);
        assert!(super::read_settings(&p).is_some());
        s = super::read_settings(&p).unwrap_or_default();
        assert_eq!(serde_json::to_value(&s).ok(), serde_json::to_value(wordcraft_assistant::Settings::default()).ok());
        assert!(super::read_settings(&dir.join("missing.json")).is_none());
        // An old file without the rewrite section still loads, with the defaults.
        std::fs::write(&p, br#"{"provider":"claude","providers":{}}"#).ok();
        let old = super::read_settings(&p).unwrap_or_default();
        assert_eq!(old.provider, "claude");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
