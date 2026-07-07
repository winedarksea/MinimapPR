use crate::state::AppState;
use leptos::prelude::*;

fn lag_tone(seconds_behind: Option<f64>) -> &'static str {
    match seconds_behind {
        Some(value) if value > 10.0 => "danger",
        Some(value) if value > 3.0 => "warn",
        Some(_) => "ok",
        None => "neutral",
    }
}

#[component]
pub fn StatusRibbon() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let cop_status = state.cop_status;
    let fusion_status = state.fusion_status;
    let ws_status = state.ws_status;

    view! {
        <section class="workspace-status-ribbon" aria-label="Live system status">
            <span class="strip-chip info">
                <span class="label">"WS"</span>
                <span class="value">{move || format!("{:?}", ws_status.get())}</span>
            </span>
            <span class="strip-chip ok">
                <span class="label">"Nodes"</span>
                <span class="value">{move || {
                    cop_status
                        .get()
                        .map(|status| status.active_nodes.to_string())
                        .unwrap_or_else(|| "-".to_string())
                }}</span>
            </span>
            <span class="strip-chip neutral">
                <span class="label">"Tracks"</span>
                <span class="value">{move || {
                    cop_status
                        .get()
                        .map(|status| status.active_tracks.to_string())
                        .unwrap_or_else(|| "-".to_string())
                }}</span>
            </span>
            <span class=move || {
                let lag = fusion_status
                    .get()
                    .and_then(|status| status.realtime.pipeline_seconds_behind_realtime);
                format!("strip-chip {}", lag_tone(lag))
            }>
                <span class="label">"Lag"</span>
                <span class="value">{move || {
                    fusion_status
                        .get()
                        .and_then(|status| status.realtime.pipeline_seconds_behind_realtime)
                        .map(|seconds| format!("{seconds:.1}s"))
                        .unwrap_or_else(|| "-".to_string())
                }}</span>
            </span>
        </section>
    }
}
