use crate::state::{AppState, NodeStatus, RfSpectrumFrame};
use leptos::prelude::*;

#[component]
pub fn RfPanel() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let frames = state.modality.rf_spectra;
    let nodes = state.nodes;

    view! {
        <div class="tab-pane modality-panel">
            <div class="future-modality-header">
                <span class="tone-badge neutral">"RF"</span>
                <span class="mock-watermark">"MOCK - awaiting backend"</span>
            </div>
            {move || {
                let current_frames = frames.get();
                if current_frames.is_empty() {
                    return view! { <div class="empty-state">"Add a node with sdr or radar_24g capability to enable RF feeds"</div> }.into_any();
                }
                let current_nodes = nodes.get();
                view! {
                    <div class="future-modality-list">
                        {current_frames.into_iter().map(|frame| {
                            let label = node_label(&current_nodes, &frame.device_id);
                            view! { <RfFrameCard frame label /> }
                        }).collect_view()}
                    </div>
                }.into_any()
            }}
        </div>
    }
}

#[component]
fn RfFrameCard(frame: RfSpectrumFrame, label: String) -> impl IntoView {
    view! {
        <article class="future-modality-card rf-frame-card">
            <div class="modality-card-head">
                <div>
                    <strong>{label}</strong>
                    <code>{frame.device_id.clone()}</code>
                </div>
                <span class="conf-pill">{format!("{:.1} MHz", frame.center_mhz)}</span>
            </div>
            <div class="rf-spectrum" aria-label="RF spectrum">
                {frame.bins.into_iter().map(|bin| {
                    let height = format!("{:.1}%", (bin * 100.0).clamp(4.0, 100.0));
                    view! { <span style:height=height></span> }
                }).collect_view()}
            </div>
            <div class="rf-emitter-list">
                {frame.emitters.into_iter().map(|emitter| {
                    let position = match (emitter.lat, emitter.lon) {
                        (Some(lat), Some(lon)) => format!("{lat:.5}, {lon:.5}"),
                        _ => "position pending".to_string(),
                    };
                    view! {
                        <div class="rf-emitter-row">
                            <strong>{emitter.emitter_id}</strong>
                            <span>{format!("{:.0} dBm", emitter.rssi_dbm)}</span>
                            <span>{position}</span>
                            <span class="conf-pill">{format!("drone {:.0}%", emitter.drone_likelihood * 100.0)}</span>
                        </div>
                    }
                }).collect_view()}
            </div>
        </article>
    }
}

fn node_label(nodes: &[NodeStatus], node_id: &str) -> String {
    nodes
        .iter()
        .find(|node| node.node_id == node_id)
        .map(|node| node.node_id.clone())
        .unwrap_or_else(|| node_id.to_string())
}
