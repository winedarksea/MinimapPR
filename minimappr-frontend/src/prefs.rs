//! localStorage wrapper for persisted UI prefs (theme, density, filters…).
//!
//! All keys are namespaced under `mmp.*` to avoid clashes with other apps
//! that might be served from the same origin in future dev setups.

use wasm_bindgen::JsCast;
use web_sys::{HtmlElement, Storage};

pub const KEY_THEME:   &str = "mmp.theme";
pub const KEY_DENSITY: &str = "mmp.density";

fn storage() -> Option<Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

pub fn get(key: &str) -> Option<String> {
    storage()?.get_item(key).ok().flatten()
}

pub fn set(key: &str, value: &str) {
    if let Some(s) = storage() {
        let _ = s.set_item(key, value);
    }
}

/// Read the current `data-theme` attribute on `<html>`; defaults to `"dark"`.
pub fn current_theme() -> String {
    root_attr("data-theme").unwrap_or_else(|| "dark".into())
}

/// Read the current `data-density` attribute on `<html>`; defaults to `"comfortable"`.
pub fn current_density() -> String {
    root_attr("data-density").unwrap_or_else(|| "comfortable".into())
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
    let next = if current_theme() == "light" { "dark" } else { "light" };
    apply_theme(next);
    next.to_string()
}
