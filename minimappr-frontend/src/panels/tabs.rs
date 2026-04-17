use crate::state::{AppState, Tab};
use crate::panels::{alerts::AlertsPane, config::ConfigPane, detections::DetectionsPane, tracks::TracksPane};
use leptos::prelude::*;

#[component]
pub fn TabPanel() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");

    let active    = state.active_tab;
    let alerts    = state.alerts;
    let tracks    = state.tracks;
    let dets      = state.detections;

    let alert_count = move || {
        let n = alerts.get().len();
        view! {
            <span class="tab-badge" class:zero=move || n == 0>{if n > 0 { n.to_string() } else { String::new() }}</span>
        }
    };
    let track_count = move || {
        let n = tracks.get().len();
        view! {
            <span class="tab-badge" class:zero=move || n == 0>{if n > 0 { n.to_string() } else { String::new() }}</span>
        }
    };
    let det_count = move || {
        let n = dets.get().len();
        view! {
            <span class="tab-badge" class:zero=move || n == 0>{if n > 0 { n.to_string() } else { String::new() }}</span>
        }
    };

    let tab_btn = move |tab: Tab, label: &'static str| {
        let is_active = {
            let t = tab.clone();
            move || active.get() == t
        };
        let t = tab.clone();
        view! {
            <button
                class="tab-btn"
                class:active=is_active
                on:click=move |_| active.set(t.clone())
            >
                {label}
            </button>
        }
    };

    view! {
        <div class="tab-column">
            <div class="tab-strip">
                {tab_btn(Tab::Tracks, "Tracks")}
                {move || view! { {track_count()} }}
                {tab_btn(Tab::Detections, "Detections")}
                {move || view! { {det_count()} }}
                {tab_btn(Tab::Alerts, "Alerts")}
                {move || view! { {alert_count()} }}
                {tab_btn(Tab::Config, "Config")}
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
