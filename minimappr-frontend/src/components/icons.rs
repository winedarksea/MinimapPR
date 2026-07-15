//! Inline SVG array-type icons, keyed by `NodeType` value. `fill="currentColor"`
//! so they inherit text color in both themes. Introduced for the pipeline DAG
//! source cards; reusable in the nodes list / COP node markers later.

use leptos::prelude::*;

/// Renders a small array-type glyph for the given `node_type` string
/// (`"point"`, `"sirith_tetra"`, `"sirith_planar"`, `"array"`, `"gateway"`), falling back to the
/// Material `hub` icon for unknown/gateway types.
#[component]
pub fn ArrayTypeIcon(node_type: String, #[prop(default = 18)] size: u32) -> impl IntoView {
    let size_str = size.to_string();
    match node_type.as_str() {
        "point" => view! {
            <svg width=size_str.clone() height=size_str viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <circle cx="12" cy="12" r="3" fill="currentColor" />
                <path d="M12 4 A8 8 0 0 1 12 20" stroke="currentColor" stroke-width="1.4" opacity="0.55" fill="none" />
                <path d="M12 7 A5 5 0 0 1 12 17" stroke="currentColor" stroke-width="1.4" opacity="0.85" fill="none" />
            </svg>
        }.into_any(),
        "sirith_tetra" => view! {
            <svg width=size_str.clone() height=size_str viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M12 3 L21 19 L3 19 Z" stroke="currentColor" stroke-width="1.4" fill="none" />
                <circle cx="12" cy="3" r="1.6" fill="currentColor" />
                <circle cx="21" cy="19" r="1.6" fill="currentColor" />
                <circle cx="3" cy="19" r="1.6" fill="currentColor" />
                <circle cx="12" cy="14" r="1.6" fill="currentColor" />
            </svg>
        }.into_any(),
        "sirith_planar" => view! {
            <svg width=size_str.clone() height=size_str viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <rect x="5" y="5" width="14" height="14" rx="1" stroke="currentColor" stroke-width="1.4" fill="none" />
                <circle cx="5" cy="5" r="1.6" fill="currentColor" />
                <circle cx="19" cy="5" r="1.6" fill="currentColor" />
                <circle cx="5" cy="19" r="1.6" fill="currentColor" />
                <circle cx="19" cy="19" r="1.6" fill="currentColor" />
                <circle cx="12" cy="12" r="1.6" fill="currentColor" />
            </svg>
        }.into_any(),
        "array" => view! {
            <svg width=size_str.clone() height=size_str viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <circle cx="6" cy="6" r="1.6" fill="currentColor" />
                <circle cx="12" cy="6" r="1.6" fill="currentColor" />
                <circle cx="18" cy="6" r="1.6" fill="currentColor" />
                <circle cx="6" cy="12" r="1.6" fill="currentColor" />
                <circle cx="12" cy="12" r="1.6" fill="currentColor" />
                <circle cx="18" cy="12" r="1.6" fill="currentColor" />
                <circle cx="6" cy="18" r="1.6" fill="currentColor" />
                <circle cx="12" cy="18" r="1.6" fill="currentColor" />
                <circle cx="18" cy="18" r="1.6" fill="currentColor" />
            </svg>
        }.into_any(),
        _ => view! {
            <span class="material-symbols-rounded" aria-hidden="true" style=format!("font-size:{size}px")>"hub"</span>
        }.into_any(),
    }
}
