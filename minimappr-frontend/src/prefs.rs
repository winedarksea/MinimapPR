//! localStorage wrapper for persisted UI prefs (theme, density, filters…).
//!
//! All keys are namespaced under `mmp.*` to avoid clashes with other apps
//! that might be served from the same origin in future dev setups.

use wasm_bindgen::JsCast;
use web_sys::{HtmlElement, Storage};

pub const KEY_THEME: &str = "mmp.theme";
pub const KEY_WORKSPACE: &str = "mmp.workspace.v1";
pub const KEY_LAYERS: &str = "mmp.layers.v1";
pub const KEY_DEVICES: &str = "mmp.devices.v1";

fn storage() -> Option<Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

pub fn set(key: &str, value: &str) {
    if let Some(s) = storage() {
        let _ = s.set_item(key, value);
    }
}

pub fn get_json<T>(key: &str) -> Option<T>
where
    T: serde::de::DeserializeOwned,
{
    let value = storage()?.get_item(key).ok().flatten()?;
    serde_json::from_str(&value).ok()
}

pub fn set_json<T>(key: &str, value: &T)
where
    T: serde::Serialize,
{
    if let Ok(serialized) = serde_json::to_string(value) {
        set(key, &serialized);
    }
}

/// Read the current `data-theme` attribute on `<html>`; defaults to `"dark"`.
pub fn current_theme() -> String {
    root_attr("data-theme").unwrap_or_else(|| "dark".into())
}

fn root_attr(name: &str) -> Option<String> {
    let doc = web_sys::window()?.document()?;
    let root = doc.document_element()?;
    root.get_attribute(name)
}

fn set_root_attr(name: &str, value: &str) {
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        if let Some(root) = doc.document_element() {
            if let Some(html) = root.dyn_ref::<HtmlElement>() {
                let _ = html.set_attribute(name, value);
            } else {
                let _ = root.set_attribute(name, value);
            }
        }
    }
}

/// Persist `theme` ("dark" | "light") and apply it to `<html data-theme>`.
pub fn apply_theme(theme: &str) {
    set_root_attr("data-theme", theme);
    set(KEY_THEME, theme);
}

/// Toggle between dark and light; returns the new theme string.
pub fn toggle_theme() -> String {
    let next = if current_theme() == "light" {
        "dark"
    } else {
        "light"
    };
    apply_theme(next);
    next.to_string()
}
