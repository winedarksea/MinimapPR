use crate::state::{AppState, Tab};
use crate::panels::{alerts::AlertsPane, config::ConfigPane, detections::DetectionsPane, tracks::TracksPane};
use leptos::prelude::*;

fn badge(count: impl Fn() -> usize + Send + Sync + Clone + 'static) -> impl IntoView {
    let count2 = count.clone();
    view! {
        <span class="tab-badge" class:zero=move || count() == 0>
            {move || { let n = count2(); if n > 0 { n.to_string() } else { String::new() } }}
        </span>
    }
}

#[component]
pub fn TabPanel() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");

    let active = state.active_tab;
    let alerts = state.alerts;
    let tracks = state.tracks;
    let dets   = state.detections;

    view! {
        <div class="tab-column">
            <div class="tab-strip">
                <button
                    class="tab-btn"
                    class:active=move || active.get() == Tab::Tracks
                    on:click=move |_| active.set(Tab::Tracks)
                >
                    "Tracks"
                    {badge(move || tracks.get().len())}
                </button>
                <button
                    class="tab-btn"
                    class:active=move || active.get() == Tab::Detections
                    on:click=move |_| active.set(Tab::Detections)
                >
                    "Detections"
                    {badge(move || dets.get().len())}
                </button>
                <button
                    class="tab-btn"
                    class:active=move || active.get() == Tab::Alerts
                    on:click=move |_| active.set(Tab::Alerts)
                >
                    "Alerts"
                    {badge(move || alerts.get().len())}
                </button>
                <button
                    class="tab-btn"
                    class:active=move || active.get() == Tab::Config
                    on:click=move |_| active.set(Tab::Config)
                >
                    "Config"
                </button>
            </div>
            <div class="tab-content">
                {move || match active.get() {
                    Tab::Tracks     => view! { <TracksPane /> }.into_any(),
                    Tab::Detections => view! { <DetectionsPane /> }.into_any(),
                    Tab::Alerts     => view! { <AlertsPane /> }.into_any(),
                    Tab::Config     => view! { <ConfigPane /> }.into_any(),
                }}
            </div>
        </div>
    }
}
