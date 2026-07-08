mod api;
mod audio;
mod components;
mod inspector;
mod layers;
mod map;
mod mock;
mod pages;
mod panels;
mod prefs;
mod recording;
mod shell;
mod state;
mod ui;
mod workspace;
mod ws;

use leptos::prelude::*;
use leptos_router::components::{ParentRoute, Redirect, Route, Router, Routes};
use leptos_router::path;
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

    // Bootstrap WS + polling — live data flows regardless of which page is active.
    ws::start_ws(state.clone());
    api::start_polling(state.clone());
    mock::start_mock_feeds(state.clone());

    view! {
        <Router>
            <div class="app-shell">
                <shell::TopBar />
                <shell::StatusStrip />
                <Routes fallback=NotFound>
                    // Root → /cop
                    <Route path=path!("") view=|| view! { <Redirect path="/cop" /> } />

                    <Route path=path!("/cop") view=pages::cop::CopPage />

                    <ParentRoute path=path!("/analysis") view=pages::analysis::AnalysisLayout>
                        <Route path=path!("")         view=pages::analysis::daily::DailyDetectionsView />
                        <Route path=path!("daily")    view=pages::analysis::daily::DailyDetectionsView />
                        <Route path=path!("labels")   view=pages::analysis::labels::LabelSummaryView />
                        <Route path=path!("labels/:id") view=pages::analysis::labels::LabelDetailView />
                        <Route path=path!("heatmap")  view=pages::analysis::heatmap::GeoHeatmapView />
                    </ParentRoute>

                    <ParentRoute path=path!("/audio") view=pages::audio::AudioLayout>
                        <Route path=path!("")            view=|| view! { <Redirect path="/audio/record" /> } />
                        <Route path=path!("record")      view=pages::audio::recording::RecordingPage />
                        <Route path=path!("analysis")    view=pages::audio::AudioAnalysisPage />
                        <Route path=path!("d/:id")       view=pages::audio::AudioAnalysisPage />
                    </ParentRoute>

                    <ParentRoute path=path!("/settings") view=pages::settings::SettingsLayout>
                        <Route path=path!("")          view=pages::settings::config::ConfigView />
                        <Route path=path!("config")    view=pages::settings::config::ConfigView />
                        <Route path=path!("rules")     view=pages::settings::rules::RulesView />
                        <Route path=path!("overlays")  view=pages::settings::overlays::OverlaysView />
                        <Route path=path!("nodes")     view=pages::settings::nodes::NodesView />
                        <Route path=path!("nodes/:id") view=pages::settings::nodes::NodeDetailView />
                        <Route path=path!("integrations") view=pages::settings::integrations::IntegrationsView />
                        <Route path=path!("server")    view=pages::settings::server::ServerDiagnosticsView />
                        <Route path=path!("logs")      view=pages::settings::logs::ServerLogsView />
                    </ParentRoute>
                </Routes>
                <audio::drawer::AudioAnalysisDrawer />
            </div>
            <audio id="audio-player" />
        </Router>
    }
}

#[component]
fn NotFound() -> impl IntoView {
    view! {
        <div class="page-stub">
            <h2>"Page not found"</h2>
            <p>"That route doesn't exist — try COP, Analysis, Audio, or Settings above."</p>
        </div>
    }
}
