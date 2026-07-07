use crate::devices::schema::DeviceRecord;
use crate::map::bindings::pan_to;
use crate::state::{AppState, TranscriptLine};
use leptos::prelude::*;

#[component]
pub fn SpeechPanel() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let lines = state.modality.speech_lines;
    let devices = state.devices;

    view! {
        <div class="tab-pane modality-panel">
            <div class="future-modality-header">
                <span class="tone-badge neutral">"Speech"</span>
                <span class="mock-watermark">"MOCK - awaiting backend"</span>
            </div>
            {move || {
                let current_lines = lines.get();
                if current_lines.is_empty() {
                    return view! { <div class="empty-state">"Register a speech node to enable the transcript ticker"</div> }.into_any();
                }
                let current_devices = devices.get();
                view! {
                    <div class="speech-ticker">
                        {current_lines.into_iter().map(|line| {
                            let device = current_devices
                                .iter()
                                .find(|device| device.id == line.device_id)
                                .cloned();
                            view! { <TranscriptRow line device /> }
                        }).collect_view()}
                    </div>
                }.into_any()
            }}
        </div>
    }
}

#[component]
fn TranscriptRow(line: TranscriptLine, device: Option<DeviceRecord>) -> impl IntoView {
    let label = device
        .as_ref()
        .map(DeviceRecord::display_label)
        .unwrap_or_else(|| line.device_id.clone());
    let coordinates = device
        .as_ref()
        .and_then(|device| device.lat.zip(device.lon));
    let row_title = coordinates
        .map(|(lat, lon)| format!("Pan to {lat:.5}, {lon:.5}"))
        .unwrap_or_else(|| "Speech source has no registered position".to_string());

    view! {
        <button
            type="button"
            class="speech-ticker-row"
            title=row_title
            disabled=coordinates.is_none()
            on:click=move |_| {
                if let Some((lat, lon)) = coordinates {
                    pan_to(lat, lon);
                }
            }
        >
            <span class="speech-ticker-source">{label}</span>
            <span class="speech-ticker-text">{line.text}</span>
            <span class="conf-pill">{format!("{:.0}%", line.confidence * 100.0)}</span>
        </button>
    }
}
