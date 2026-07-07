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

fn warn_metric_tone(value: i64) -> &'static str {
    if value > 0 {
        "warn"
    } else {
        "neutral"
    }
}

fn danger_metric_tone(value: i64) -> &'static str {
    if value > 0 {
        "danger"
    } else {
        "neutral"
    }
}

#[component]
pub fn StatusRibbon() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let cop_status = state.cop_status;
    let fusion_status = state.fusion_status;
    let ws_status = state.ws_status;
    let nodes = state.nodes;
    let alerts = state.alerts;

    view! {
        <section class="workspace-status-ribbon" aria-label="Live system status">
            <span class="strip-chip info">
                <span class="label">"WS"</span>
                <span class="value">{move || format!("{:?}", ws_status.get())}</span>
            </span>
            <span class="strip-chip ok">
                <span class="label">"Online"</span>
                <span class="value">{move || {
                    cop_status
                        .get()
                        .map(|status| status.active_nodes as i64)
                        .unwrap_or_else(|| nodes.get().len() as i64)
                        .to_string()
                }}</span>
            </span>
            <span class=move || {
                let value = cop_status.get().map(|status| status.degraded_nodes as i64).unwrap_or(0);
                format!("strip-chip {}", warn_metric_tone(value))
            }>
                <span class="label">"Degraded"</span>
                <span class="value">{move || {
                    cop_status.get().map(|status| status.degraded_nodes.to_string()).unwrap_or_else(|| "0".to_string())
                }}</span>
            </span>
            <span class=move || {
                let value = cop_status.get().map(|status| status.offline_nodes as i64).unwrap_or(0);
                format!("strip-chip {}", danger_metric_tone(value))
            }>
                <span class="label">"Offline"</span>
                <span class="value">{move || {
                    cop_status.get().map(|status| status.offline_nodes.to_string()).unwrap_or_else(|| "0".to_string())
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
                let value = cop_status
                    .get()
                    .map(|status| status.open_alerts as i64)
                    .unwrap_or_else(|| alerts.get().len() as i64);
                format!("strip-chip {}", danger_metric_tone(value))
            }>
                <span class="label">"Alerts"</span>
                <span class="value">{move || {
                    cop_status
                        .get()
                        .map(|status| status.open_alerts as i64)
                        .unwrap_or_else(|| alerts.get().len() as i64)
                        .to_string()
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
