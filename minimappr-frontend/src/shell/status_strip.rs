use crate::state::AppState;
use leptos::prelude::*;

/// Persistent below-top-bar strip — summary metrics always visible.
#[component]
pub fn StatusStrip() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let cop = state.cop_status;
    let fusion = state.fusion_status;
    let tracks = state.tracks;
    let dets = state.detections;
    let alerts = state.alerts;
    let nodes = state.nodes;

    view! {
        <div class="system-strip" role="status">
            <StatusChip
                label="Online"
                value=Signal::derive(move || {
                    cop.get()
                        .map(|c| c.active_nodes as i64)
                        .unwrap_or_else(|| nodes.get().len() as i64)
                })
                tone=Signal::derive(move || "ok")
            />
            <StatusChip
                label="Degraded"
                value=Signal::derive(move || cop.get().map(|c| c.degraded_nodes as i64).unwrap_or(0))
                tone=Signal::derive(move || metric_tone(cop.get().map(|c| c.degraded_nodes as i64).unwrap_or(0)))
            />
            <StatusChip
                label="Offline"
                value=Signal::derive(move || cop.get().map(|c| c.offline_nodes as i64).unwrap_or(0))
                tone=Signal::derive(move || danger_metric_tone(cop.get().map(|c| c.offline_nodes as i64).unwrap_or(0)))
            />
            <StatusChip
                label="Tracks"
                value=Signal::derive(move || {
                    cop.get()
                        .map(|c| c.active_tracks as i64)
                        .unwrap_or_else(|| tracks.get().len() as i64)
                })
                tone=Signal::derive(move || "info")
            />
            <StatusChip
                label="Alerts"
                value=Signal::derive(move || {
                    cop.get()
                        .map(|c| c.open_alerts as i64)
                        .unwrap_or_else(|| alerts.get().len() as i64)
                })
                tone=Signal::derive(move || danger_metric_tone(
                    cop.get()
                        .map(|c| c.open_alerts as i64)
                        .unwrap_or_else(|| alerts.get().len() as i64)
                ))
            />
            <StatusChip
                label="Det/60s"
                value=Signal::derive(move || cop.get().map(|c| c.detections_last_60s as i64).unwrap_or(0))
                tone=Signal::derive(move || "neutral")
            />
            <StatusChipText
                label="Pipeline Lag"
                value=Signal::derive(move || pipeline_lag_text(&fusion))
                tone=Signal::derive(move || pipeline_lag_tone(&fusion))
            />
            <StatusChipText
                label="Last Detection"
                value=Signal::derive(move || last_detection_age(&dets))
                tone=Signal::derive(move || last_detection_tone(&dets))
            />
        </div>
    }
}

fn pipeline_lag_tone(
    fusion: &RwSignal<Option<crate::state::FusionStatus>>,
) -> &'static str {
    let lag_seconds = fusion
        .get()
        .and_then(|status| status.realtime.pipeline_seconds_behind_realtime);
    match lag_seconds {
        Some(seconds) if seconds >= 30.0 => "danger",
        Some(seconds) if seconds >= 5.0 => "warn",
        Some(_) => "ok",
        None => "neutral",
    }
}

fn pipeline_lag_text(
    fusion: &RwSignal<Option<crate::state::FusionStatus>>,
) -> String {
    let lag_seconds = fusion
        .get()
        .and_then(|status| status.realtime.pipeline_seconds_behind_realtime);
    match lag_seconds {
        Some(seconds) if seconds < 60.0 => format!("{seconds:.1}s"),
        Some(seconds) => format!("{:.1}m", seconds / 60.0),
        None => "—".into(),
    }
}

#[component]
fn StatusChip(
    label: &'static str,
    value: Signal<i64>,
    tone: Signal<&'static str>,
) -> impl IntoView {
    view! {
        <span class=move || format!("strip-chip {}", tone.get())>
            <span class="label">{label}</span>
            <span class="value">{move || value.get()}</span>
        </span>
    }
}

#[component]
fn StatusChipText(
    label: &'static str,
    value: Signal<String>,
    tone: Signal<&'static str>,
) -> impl IntoView {
    view! {
        <span class=move || format!("strip-chip {}", tone.get())>
            <span class="label">{label}</span>
            <span class="value">{move || value.get()}</span>
        </span>
    }
}

fn metric_tone(value: i64) -> &'static str {
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

fn last_detection_tone(
    dets: &RwSignal<std::collections::VecDeque<crate::state::Detection>>,
) -> &'static str {
    let ds = dets.get();
    let Some(detection) = ds.front() else {
        return "neutral";
    };
    let Some(received_ns) = detection.received_ns else {
        return "neutral";
    };
    let age_seconds = ((js_sys::Date::now() - received_ns as f64 / 1_000_000.0) / 1000.0).max(0.0);
    if age_seconds < 30.0 {
        "ok"
    } else if age_seconds < 120.0 {
        "warn"
    } else {
        "danger"
    }
}

fn last_detection_age(
    dets: &RwSignal<std::collections::VecDeque<crate::state::Detection>>,
) -> String {
    let ds = dets.get();
    let Some(d) = ds.front() else {
        return "—".into();
    };
    let Some(ns) = d.received_ns else {
        return "—".into();
    };
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
