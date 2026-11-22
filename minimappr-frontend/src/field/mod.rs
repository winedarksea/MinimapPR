//! Field Mode PWA (feature `field-pwa`; built by `trunk build --config
//! Trunk.field.toml`). A phone node without a server: mic capture through an
//! AudioWorklet, `minimappr-core` Stage-1 + clip codec + protocol + sealing
//! in wasm, BirdNET V3 through ONNX Runtime Web, and the relay client
//! (`minimappr-transport` web adapter). Server-backed `api.rs`/`ws.rs` are
//! replaced by the local shim in `mock::field_shim`.

mod birdnet_input_resampler;
mod browser_bindings;
mod capture_controller;
mod field_configuration;
mod field_message_builders;
mod field_panels;
mod field_signals;
mod relay_message_presenter;
mod relay_session_controller;
mod stage1_capture_pipeline;

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

use crate::mock::field_shim;
use crate::state::AppState;
use field_signals::FieldSignals;

#[component]
pub fn FieldApp() -> impl IntoView {
    let app_state = AppState::new();
    provide_context(app_state.clone());
    field_shim::start_field_local_feeds(&app_state);
    let signals = FieldSignals::new();
    provide_context(signals);

    browser_bindings::register_service_worker(field_configuration::SERVICE_WORKER_URL);
    spawn_local(capture_controller::load_classifier(signals));

    view! {
        <main class="field-app">
            <header class="field-header">
                <h1>"MinimapPR Field"</h1>
                <field_panels::ConnectionBadge />
            </header>
            <field_panels::CapturePanel />
            <field_panels::ClassifierPanel />
            <field_panels::RelayPanel />
            <field_panels::DetectionFeedPanel />
        </main>
    }
}
