use crate::state::AppState;
use leptos::prelude::*;

/// Persistent below-top-bar strip — summary metrics always visible.
#[component]
pub fn StatusStrip() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let cop = state.cop_status;
    let tracks = state.tracks;
    let dets = state.detections;
    let alerts = state.alerts;
    let nodes = state.nodes;

    view! {
        <div class="system-strip" role="status">
            <span class="strip-stat">
                <span class="label">"Nodes"</span>
                <span class="value">
                    {move || cop.get().map(|c| c.active_nodes as i64)
                        .unwrap_or_else(|| nodes.get().len() as i64)}
                </span>
            </span>
            <span class="strip-stat">
                <span class="label">"Degraded"</span>
                <span class="value">
                    {move || cop.get().map(|c| c.degraded_nodes as i64).unwrap_or(0)}
                </span>
            </span>
            <span class="strip-stat">
                <span class="label">"Tracks"</span>
                <span class="value">
                    {move || cop.get().map(|c| c.active_tracks as i64)
                        .unwrap_or_else(|| tracks.get().len() as i64)}
                </span>
            </span>
            <span class="strip-stat">
                <span class="label">"Alerts"</span>
                <span class="value">
                    {move || cop.get().map(|c| c.open_alerts as i64)
                        .unwrap_or_else(|| alerts.get().len() as i64)}
                </span>
            </span>
            <span class="strip-stat">
                <span class="label">"Det/60s"</span>
                <span class="value">
                    {move || cop.get().map(|c| c.detections_last_60s as i64).unwrap_or(0)}
                </span>
            </span>
            <span class="strip-stat">
                <span class="label">"Last det"</span>
                <span class="value">{move || last_detection_age(&dets)}</span>
            </span>
        </div>
    }
}

fn last_detection_age(
    dets: &RwSignal<std::collections::VecDeque<crate::state::Detection>>,
) -> String {
    let ds = dets.get();
    let Some(d) = ds.front() else { return "—".into() };
    let Some(ns) = d.received_ns else { return "—".into() };
    let now_ms = js_sys::Date::now();
    let sample_ms = ns as f64 / 1_000_000.0;
    let age_s = ((now_ms - sample_ms) / 1000.0).max(0.0);
    if age_s < 60.0 {
        format!("{:.0}s", age_s)
    } else if age_s < 3600.0 {
        format!("{:.0}m", age_s / 60.0)
    } else {
        format!("{:.1}h", age_s / 3600.0)
    }
}
