use crate::map::bindings::pan_to;
use crate::state::{AppState, NodeStatus, TranscriptLine};
use leptos::prelude::*;
use leptos_router::components::A;

#[component]
pub fn SpeechPanel() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let lines = state.modality.speech_lines;
    let nodes = state.nodes;

    view! {
        <div class="tab-pane modality-panel">
            <div class="future-modality-header">
                <span class="tone-badge neutral">"Speech"</span>
            </div>
            {move || {
                let current_lines = lines.get();
                if current_lines.is_empty() {
                    return view! { <div class="empty-state">"No captured transcripts yet"</div> }.into_any();
                }
                let current_nodes = nodes.get();
                view! {
                    <div class="speech-ticker">
                        {current_lines.into_iter().map(|line| {
                            let node = current_nodes
                                .iter()
                                .find(|node| node.node_id == line.device_id)
                                .cloned();
                            view! { <TranscriptRow line node /> }
                        }).collect_view()}
                    </div>
                }.into_any()
            }}
        </div>
    }
}

#[component]
fn TranscriptRow(line: TranscriptLine, node: Option<NodeStatus>) -> impl IntoView {
    let label = node
        .as_ref()
        .map(|node| node.node_id.clone())
        .unwrap_or_else(|| line.device_id.clone());
    let coordinates = node
        .as_ref()
        .and_then(|node| node.position_geo.as_ref().map(|geo| (geo.lat, geo.lon)));
    let row_title = coordinates
        .map(|(lat, lon)| format!("Pan to {lat:.5}, {lon:.5}"))
        .unwrap_or_else(|| "Speech source has no registered position".to_string());
    let transcript_href = format!("/audio/t/{}", js_sys::encode_uri_component(&line.line_id));

    view! {
        <article class="speech-ticker-row">
            <span class="speech-ticker-source">{label}</span>
            <span class="speech-ticker-text">{line.text}</span>
            <span class="conf-pill">{format!("{:.0}%", line.confidence * 100.0)}</span>
            <div class="speech-ticker-actions">
                <A href=transcript_href attr:class="btn-sm">"Review"</A>
                {if coordinates.is_some() {
                    view! {
                        <button
                            type="button"
                            class="btn-sm"
                            title=row_title
                            on:click=move |_| {
                                if let Some((lat, lon)) = coordinates {
                                    pan_to(lat, lon);
                                }
                            }
                        >
                            "Map"
                        </button>
                    }.into_any()
                } else {
                    ().into_any()
                }}
            </div>
        </article>
    }
}
