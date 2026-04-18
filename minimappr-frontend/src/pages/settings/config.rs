use crate::panels::config::ConfigPane;
use leptos::prelude::*;

/// Phase 2: reuses the existing ConfigPane — just wraps it in the settings page frame.
#[component]
pub fn ConfigView() -> impl IntoView {
    view! {
        <div class="tab-column">
            <ConfigPane />
        </div>
    }
}
