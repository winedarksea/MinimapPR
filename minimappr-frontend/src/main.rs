mod api;
mod map;
mod panels;
mod state;
mod ws;

use leptos::prelude::*;
use state::AppState;

fn main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Debug).ok();
    mount_to_body(App);
}

#[component]
fn App() -> impl IntoView {
    let state = AppState::new();
    provide_context(state.clone());

    // Bootstrap polls + WS
    ws::start_ws(state.clone());
    api::start_polling(state.clone());

    view! {
        <Header state=state.clone() />
        <SystemStrip state=state.clone() />
        <main class="main-grid">
            <panels::node_status::NodeStatusPanel />
            <map::LeafletMapPanel />
            <panels::tabs::TabPanel />
        </main>
        <audio id="audio-player" />
    }
}

#[component]
fn Header(state: AppState) -> impl IntoView {
    let ws_status = state.ws_status;
    let pill_class = move || match ws_status.get() {
        state::WsStatus::Connected    => "ws-pill connected",
        state::WsStatus::Reconnecting => "ws-pill reconnecting",
        state::WsStatus::Disconnected => "ws-pill disconnected",
    };
    let pill_text = move || match ws_status.get() {
        state::WsStatus::Connected    => "Live",
        state::WsStatus::Reconnecting => "Reconnecting…",
        state::WsStatus::Disconnected => "Disconnected",
    };

    view! {
        <header class="header">
            <span class="header-title">"MinimapPR COP"</span>
            <span class=pill_class>{pill_text}</span>
        </header>
    }
}

#[component]
fn SystemStrip(state: AppState) -> impl IntoView {
    let cop = state.cop_status;

    view! {
        <div class="system-strip">
            <span class="strip-stat">
                <span class="label">"Nodes"</span>
                <span class="value">{move || cop.get().map(|c| c.active_nodes).unwrap_or(0)}</span>
            </span>
            <span class="strip-stat">
                <span class="label">"Degraded"</span>
                <span class="value">{move || cop.get().map(|c| c.degraded_nodes).unwrap_or(0)}</span>
            </span>
            <span class="strip-stat">
                <span class="label">"Tracks"</span>
                <span class="value">{move || cop.get().map(|c| c.active_tracks).unwrap_or(0)}</span>
            </span>
            <span class="strip-stat">
                <span class="label">"Alerts"</span>
                <span class="value">{move || cop.get().map(|c| c.open_alerts).unwrap_or(0)}</span>
            </span>
        </div>
    }
}
