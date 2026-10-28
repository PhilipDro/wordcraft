//! Provider presets and the user's settings for them.
//!
//! Every provider but Anthropic speaks the OpenAI-compatible `chat/completions` API (Gemini
//! through its `/v1beta/openai` endpoint), so a provider is a row here, not code.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The wire format a provider speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Protocol {
    /// Anthropic Messages API (`POST {base}/messages`).
    Anthropic,
    /// OpenAI-compatible Chat Completions (`POST {base}/chat/completions`).
    OpenAi,
}

pub struct Preset {
    pub id: &'static str,
    pub name: &'static str,
    pub protocol: Protocol,
    pub base_url: &'static str,
    /// Suggested model; the user can type any other. Model names go stale: update them here.
    pub model: &'static str,
}

pub const CUSTOM: &str = "custom";
/// Philip's Hermes gateway (api_server, multiplexed profiles). The model field names the Hermes
/// profile; the gateway serves each profile at `/p/<profile>/v1` (OpenAI-compatible), so the
/// Claude-subscription profiles (golem, sonnet, haiku, …) work without an Anthropic API key.
pub const HERMES: &str = "hermes";

pub const PRESETS: &[Preset] = &[
    Preset { id: "claude", name: "Claude", protocol: Protocol::Anthropic, base_url: "https://api.anthropic.com/v1", model: "claude-sonnet-5-5" },
    Preset {
        id: "gemini",
        name: "Gemini",
        protocol: Protocol::OpenAi,
        base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
        model: "gemini-2.5-flash",
    },
    Preset { id: "openai", name: "OpenAI", protocol: Protocol::OpenAi, base_url: "https://api.openai.com/v1", model: "gpt-5" },
    Preset {
        id: "nvidia",
        name: "Nvidia",
        protocol: Protocol::OpenAi,
        base_url: "https://integrate.api.nvidia.com/v1",
        model: "meta/llama-3.3-70b-instruct",
    },
    Preset {
        id: "fireworks",
        name: "Fireworks",
        protocol: Protocol::OpenAi,
        base_url: "https://api.fireworks.ai/inference/v1",
        model: "accounts/fireworks/models/llama-v3p3-70b-instruct",
    },
    Preset { id: "deepseek", name: "DeepSeek", protocol: Protocol::OpenAi, base_url: "https://api.deepseek.com/v1", model: "deepseek-chat" },
    Preset { id: "grok", name: "Grok", protocol: Protocol::OpenAi, base_url: "https://api.x.ai/v1", model: "grok-4" },
    Preset { id: HERMES, name: "Hermes", protocol: Protocol::OpenAi, base_url: "http://127.0.0.1:8642", model: "sonnet" },
    Preset { id: CUSTOM, name: "Custom", protocol: Protocol::OpenAi, base_url: "", model: "" },
];

pub fn preset(id: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.id == id)
}

/// What the user set for one provider. Empty fields fall back to the preset. No `Debug`: it holds
/// the API key, which must never reach a log.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ProviderSettings {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
    /// Custom providers only: the wire format (default OpenAI-compatible).
    pub protocol: Option<Protocol>,
}

/// The assistant's settings, stored by the host in their own file — never in `UiState`, which
/// agents can read through the control channel.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    /// The chosen preset id.
    pub provider: String,
    pub providers: BTreeMap<String, ProviderSettings>,
    /// Prompt on selection (): scope, cap, system prompt, presets. Missing in
    /// older files: the defaults.
    pub rewrite: crate::rewrite::RewriteSettings,
}

/// Everything one request needs.
#[derive(Clone)]
pub struct Endpoint {
    pub protocol: Protocol,
    pub base_url: String,
    pub model: String,
    pub api_key: String,
}

impl Settings {
    /// The chosen provider's id (the first preset when none is chosen yet).
    pub fn provider_id(&self) -> &str {
        if preset(&self.provider).is_some() { &self.provider } else { PRESETS.first().map_or(CUSTOM, |p| p.id) }
    }

    /// The chosen provider's settings, created on first use.
    pub fn current_mut(&mut self) -> &mut ProviderSettings {
        let id = self.provider_id().to_string();
        self.providers.entry(id).or_default()
    }

    /// The endpoint to call, or why the settings aren't usable yet.
    pub fn endpoint(&self) -> Result<Endpoint, String> {
        let ep = self.connection()?;
        if ep.model.is_empty() {
            return Err("Choose a model.".into());
        }
        Ok(ep)
    }

    /// Like `endpoint`, but the model may be empty (enough to list the provider's models).
    pub fn connection(&self) -> Result<Endpoint, String> {
        let id = self.provider_id();
        let p = preset(id).ok_or("unknown provider")?;
        let user = self.providers.get(id).cloned().unwrap_or_default();
        let pick = |user: &str, preset: &str| if user.trim().is_empty() { preset.to_string() } else { user.trim().to_string() };
        let mut ep = Endpoint {
            protocol: if id == CUSTOM { user.protocol.unwrap_or(Protocol::OpenAi) } else { p.protocol },
            base_url: pick(&user.base_url, p.base_url).trim_end_matches('/').to_string(),
            model: pick(&user.model, p.model),
            api_key: user.api_key.trim().to_string(),
        };
        if id == HERMES {
            ep.base_url = hermes_base(&ep.base_url, &ep.model);
            if ep.api_key.is_empty() {
                ep.api_key = hermes_key_from_file();
            }
        }
        // A custom endpoint on this machine may be a local server that needs no key (llama.cpp,
        // Ollama, vLLM, or one forwarded to 127.0.0.1).
        let local = ["http://127.0.0.1", "http://localhost", "http://[::1]"].iter().any(|h| ep.base_url.starts_with(h));
        if ep.api_key.is_empty() && !(id == CUSTOM && local) {
            return Err(format!("Add your {} API key in the assistant's settings.", p.name));
        }
        if !(ep.base_url.starts_with("https://") || ep.base_url.starts_with("http://")) {
            return Err("Set the provider's base URL (https://…).".into());
        }
        Ok(ep)
    }
}

/// The gateway URL for one Hermes profile: `<gateway>/p/<profile>/v1`, or `<gateway>/v1` for the
/// default profile (or none). A base URL the user already pointed at a `/v1` is taken as is.
fn hermes_base(base: &str, profile: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/v1") {
        return base.to_string();
    }
    let profile = profile.trim();
    if profile.is_empty() || profile == "default" || profile == "hermes-agent" {
        format!("{base}/v1")
    } else {
        format!("{base}/p/{profile}/v1")
    }
}

/// The gateway key kept outside `assistant.json` (so a settings rewrite never drops it):
/// `~/.config/wordcraft/hermes-gateway.key`. Empty when missing.
fn hermes_key_from_file() -> String {
    std::env::var_os("HOME")
        .map(|h| std::path::Path::new(&h).join(".config/wordcraft/hermes-gateway.key"))
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|k| k.trim().to_string())
        .unwrap_or_default()
}
