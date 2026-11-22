//! Field PWA panels. `data-testid` attributes are the contract the browser
//! checks (`tests/field-pwa.spec.js`) rely on.

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

use super::capture_controller::{device_sample_index_now, start_microphone, stop_microphone};
use super::field_message_builders::build_heartbeat;
use super::field_signals::FieldSignals;
use super::relay_session_controller::{host_new_session, join_session_by_code, local_peer_id, send_sealed_message};
use crate::state::{AppState, WsStatus};

fn field_signals() -> FieldSignals {
    expect_context::<FieldSignals>()
}

#[component]
pub fn ConnectionBadge() -> impl IntoView {
    let app_state = expect_context::<AppState>();
    let label = move || match app_state.ws_status.get() {
        WsStatus::Connected => "Relay online",
        WsStatus::Reconnecting => "Relay reconnecting",
        WsStatus::Disconnected => "Solo",
    };
    view! { <span class="field-badge" data-testid="connection-badge">{label}</span> }
}

#[component]
pub fn CapturePanel() -> impl IntoView {
    let signals = field_signals();
    let app_state = expect_context::<AppState>();
    let toggle = move |_| {
        let app_state = app_state.clone();
        if signals.capture_running.get_untracked() {
            spawn_local(stop_microphone(signals));
        } else {
            spawn_local(start_microphone(signals, app_state));
        }
    };
    view! {
        <section class="field-card">
            <h2>"Capture + Stage-1"</h2>
            <button data-testid="mic-toggle" on:click=toggle>
                {move || if signals.capture_running.get() { "Stop microphone" } else { "Start microphone" }}
            </button>
            <p data-testid="capture-status">{move || signals.capture_status.get()}</p>
            <dl class="field-stats">
                <dt>"Triggers"</dt><dd data-testid="stage1-trigger-count">{move || signals.stage1_trigger_count.get()}</dd>
                <dt>"Onsets"</dt><dd data-testid="stage1-onset-count">{move || signals.stage1_onset_count.get()}</dd>
                <dt>"Trigger"</dt><dd>{move || if signals.stage1_trigger_open.get() { "open" } else { "closed" }}</dd>
                <dt>"Contrast"</dt><dd>{move || format!("{:.1} dB", signals.stage1_last_contrast_db.get())}</dd>
                <dt>"Capture gaps"</dt><dd data-testid="capture-gap-count">{move || signals.capture_gap_count.get()}</dd>
                <dt>"Windows dropped"</dt><dd>{move || signals.windows_dropped_count.get()}</dd>
            </dl>
        </section>
    }
}

#[component]
pub fn ClassifierPanel() -> impl IntoView {
    let signals = field_signals();
    let latest = move || {
        signals.latest_classification.get().map(|classification| {
            format!(
                "{} ({}) {:.2} in {:.0} ms",
                classification.common_name,
                classification.scientific_name,
                classification.score,
                classification.inference_milliseconds
            )
        })
    };
    view! {
        <section class="field-card">
            <h2>"BirdNET V3"</h2>
            <p data-testid="birdnet-status">{move || signals.birdnet_status.get()}</p>
            <p data-testid="latest-label">{move || latest().unwrap_or_else(|| "No classification yet".to_string())}</p>
            <p>
                "Clip (FLAC+MMCL): "
                <span data-testid="last-clip-bytes">
                    {move || signals.last_clip_bytes.get().map_or("none".to_string(), |bytes| format!("{bytes} B"))}
                </span>
            </p>
        </section>
    }
}

#[component]
pub fn RelayPanel() -> impl IntoView {
    let signals = field_signals();
    let app_state = expect_context::<AppState>();
    let typed_code = RwSignal::new(String::new());
    let heartbeat_sequence = RwSignal::new(0_u64);
    let host_app_state = app_state.clone();
    let host = move |_| {
        let app_state = host_app_state.clone();
        spawn_local(async move {
            if let Err(error) = host_new_session(signals, app_state).await {
                signals.relay_status.set(format!("Host failed: {error}"));
            }
        });
    };
    let join = move |_| {
        let app_state = app_state.clone();
        let code = typed_code.get_untracked();
        spawn_local(async move {
            if let Err(error) = join_session_by_code(code, signals, app_state).await {
                signals.relay_status.set(format!("Join failed: {error}"));
            }
        });
    };
    let send_heartbeat = move |_| {
        let Some(peer_id) = local_peer_id() else { return };
        heartbeat_sequence.update(|sequence| *sequence += 1);
        let message = build_heartbeat(
            peer_id,
            heartbeat_sequence.get_untracked(),
            device_sample_index_now(),
            signals.capture_gap_count.get_untracked(),
        );
        match send_sealed_message(&message) {
            Ok(()) => signals.log_relay(format!("sent sealed Heartbeat seq {}", heartbeat_sequence.get_untracked())),
            Err(error) => signals.log_relay(format!("send failed: {error}")),
        }
    };
    view! {
        <section class="field-card">
            <h2>"Relay session"</h2>
            <div class="field-row">
                <button data-testid="host-session" on:click=host>"Host session"</button>
                <input
                    data-testid="join-code-input"
                    placeholder="K7QX-731904"
                    prop:value=move || typed_code.get()
                    on:input=move |event| typed_code.set(event_target_value(&event))
                />
                <button data-testid="join-session" on:click=join>"Join"</button>
            </div>
            <p>"Join code: " <strong data-testid="join-code">{move || signals.join_code_display.get().unwrap_or_default()}</strong></p>
            <p data-testid="relay-role">{move || signals.relay_role_description.get().unwrap_or_default()}</p>
            <p data-testid="relay-status">{move || signals.relay_status.get()}</p>
            <p>
                "Session key: "
                <span data-testid="session-key-state">
                    {move || if signals.relay_session_key_installed.get() { "installed" } else { "none" }}
                </span>
                " · sealed received: "
                <span data-testid="sealed-received-count">{move || signals.relay_sealed_messages_received.get()}</span>
            </p>
            <button data-testid="send-heartbeat" on:click=send_heartbeat>"Send sealed heartbeat"</button>
            <ul class="field-log" data-testid="relay-log">
                {move || signals.relay_message_log.get().into_iter().map(|line| view! { <li>{line}</li> }).collect_view()}
            </ul>
        </section>
    }
}

#[component]
pub fn DetectionFeedPanel() -> impl IntoView {
    let app_state = expect_context::<AppState>();
    view! {
        <section class="field-card">
            <h2>"Detections"</h2>
            <ul class="field-log" data-testid="detection-feed">
                {move || {
                    app_state
                        .detections
                        .get()
                        .into_iter()
                        .map(|detection| {
                            let label = detection.label.unwrap_or_default();
                            let confidence = detection.confidence.unwrap_or_default();
                            let node = detection.node_id.unwrap_or_default();
                            view! { <li>{format!("{label} {confidence:.2} · {node}")}</li> }
                        })
                        .collect_view()
                }}
            </ul>
        </section>
    }
}
