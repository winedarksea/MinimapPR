use crate::state::AppState;
use leptos::prelude::*;

/// Persistent below-top-bar strip — summary metrics always visible.
#[component]
pub fn StatusStrip() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let cop = state.cop_status;
    let fusion = state.fusion_status;
    let tracks = state.tracks;
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
                href="/settings"
            />
            <StatusChip
                label="Degraded"
                value=Signal::derive(move || cop.get().map(|c| c.degraded_nodes as i64).unwrap_or(0))
                tone=Signal::derive(move || metric_tone(cop.get().map(|c| c.degraded_nodes as i64).unwrap_or(0)))
                href="/settings"
            />
            <StatusChip
                label="Offline"
                value=Signal::derive(move || cop.get().map(|c| c.offline_nodes as i64).unwrap_or(0))
                tone=Signal::derive(move || danger_metric_tone(cop.get().map(|c| c.offline_nodes as i64).unwrap_or(0)))
                href="/settings"
            />
            <StatusChip
                label="Tracks"
                value=Signal::derive(move || {
                    cop.get()
                        .map(|c| c.active_tracks as i64)
                        .unwrap_or_else(|| tracks.get().len() as i64)
                })
                tone=Signal::derive(move || "info")
                href="/cop"
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
                href="/cop"
            />
            <StatusChipText
                label="Pipeline"
                value=Signal::derive(move || pipeline_text(&fusion))
                tone=Signal::derive(move || pipeline_tone(&fusion))
                href="/settings/server"
            />
        </div>
    }
}

fn pipeline_tone(fusion: &RwSignal<Option<crate::state::FusionStatus>>) -> &'static str {
    let status = fusion.get();
    let lag = status
        .as_ref()
        .and_then(|s| s.realtime.pipeline_seconds_behind_realtime);
    let drought = status
        .as_ref()
        .and_then(|s| s.health.active_drought)
        .unwrap_or(false);

    if lag.map(|s| s >= 30.0).unwrap_or(false) {
        return "danger";
    }
    if lag.map(|s| s >= 5.0).unwrap_or(false) || drought {
        return "warn";
    }
    if lag.is_some() {
        return "ok";
    }
    "neutral"
}

fn pipeline_text(fusion: &RwSignal<Option<crate::state::FusionStatus>>) -> String {
    let status = fusion.get();
    let lag = status
        .as_ref()
        .and_then(|s| s.realtime.pipeline_seconds_behind_realtime);
    let drought = status
        .as_ref()
        .and_then(|s| s.health.active_drought)
        .unwrap_or(false);

    if drought && lag.map(|s| s < 5.0).unwrap_or(true) {
        return "drought".into();
    }
    match lag {
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
    #[prop(optional)] href: Option<&'static str>,
) -> impl IntoView {
    let inner = view! {
        <span class="label">{label}</span>
        <span class="value">{move || value.get()}</span>
    };
    if let Some(path) = href {
        view! {
            <a href=path class=move || format!("strip-chip {}", tone.get())>
                {inner}
            </a>
        }
        .into_any()
    } else {
        view! {
            <span class=move || format!("strip-chip {}", tone.get())>
                {inner}
            </span>
        }
        .into_any()
    }
}

#[component]
fn StatusChipText(
    label: &'static str,
    value: Signal<String>,
    tone: Signal<&'static str>,
    #[prop(optional)] href: Option<&'static str>,
) -> impl IntoView {
    let inner = view! {
        <span class="label">{label}</span>
        <span class="value">{move || value.get()}</span>
    };
    if let Some(path) = href {
        view! {
            <a href=path class=move || format!("strip-chip {}", tone.get())>
                {inner}
            </a>
        }
        .into_any()
    } else {
        view! {
            <span class=move || format!("strip-chip {}", tone.get())>
                {inner}
            </span>
        }
        .into_any()
    }
}

fn metric_tone(value: i64) -> &'static str {
    if value > 0 { "warn" } else { "neutral" }
}

fn danger_metric_tone(value: i64) -> &'static str {
    if value > 0 { "danger" } else { "neutral" }
}
