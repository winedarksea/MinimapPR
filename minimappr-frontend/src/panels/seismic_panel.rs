use crate::components::{meter::RmsMeter, strip_chart::StripChart};
use crate::devices::schema::DeviceRecord;
use crate::state::{AppState, SeismicTrace};
use leptos::prelude::*;

#[component]
pub fn SeismicPanel() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let traces = state.modality.seismic_traces;
    let devices = state.devices;

    view! {
        <div class="tab-pane modality-panel">
            <div class="future-modality-header">
                <span class="tone-badge neutral">"Seismic"</span>
                <span class="mock-watermark">"MOCK - awaiting backend"</span>
            </div>
            {move || {
                let current_traces = traces.get();
                if current_traces.is_empty() {
                    return view! { <div class="empty-state">"Register a seismic device to enable strip charts"</div> }.into_any();
                }
                let current_devices = devices.get();
                view! {
                    <div class="future-modality-list">
                        {current_traces.into_iter().map(|trace| {
                            let label = device_label(&current_devices, &trace.device_id);
                            view! { <SeismicTraceCard trace label /> }
                        }).collect_view()}
                    </div>
                }.into_any()
            }}
        </div>
    }
}

#[component]
fn SeismicTraceCard(trace: SeismicTrace, label: String) -> impl IntoView {
    view! {
        <article class="future-modality-card seismic-trace-card">
            <div class="modality-card-head">
                <div>
                    <strong>{label}</strong>
                    <code>{trace.device_id.clone()}</code>
                </div>
                <span class="conf-pill">{format!("{:.1} mm/s", trace.peak_velocity_mms)}</span>
            </div>
            <StripChart samples=trace.samples.clone() width=260.0 height=44.0 color="var(--mmp-severity-medium)" />
            <RmsMeter samples=trace.samples label="velocity" />
            <div class="row-summary-meta">
                <span class="conf-pill">{format!("{} events", trace.event_count)}</span>
                <span class="conf-pill">"node-attached"</span>
            </div>
        </article>
    }
}

fn device_label(devices: &[DeviceRecord], device_id: &str) -> String {
    devices
        .iter()
        .find(|device| device.id == device_id)
        .map(DeviceRecord::display_label)
        .unwrap_or_else(|| device_id.to_string())
}
