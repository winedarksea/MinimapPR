use crate::state::{
    AppState, CopItemKind, CopSelection, Detection, Effector, NodeStatus, Track, ZoneSpec,
};
use crate::ui::short_id;
use crate::workspace::layout::clamp_dock_width;
use leptos::prelude::*;

fn confidence_text(value: Option<f64>) -> String {
    value
        .map(|confidence| format!("{:.0}%", confidence * 100.0))
        .unwrap_or_else(|| "-".to_string())
}

fn geo_text(lat: f64, lon: f64, alt_m: Option<f64>) -> String {
    match alt_m {
        Some(alt) => format!("{lat:.5}, {lon:.5}, {alt:.1} m"),
        None => format!("{lat:.5}, {lon:.5}"),
    }
}

fn covariance_text(covariance: Option<&Vec<Vec<f64>>>) -> String {
    let Some(covariance) = covariance else {
        return "-".to_string();
    };
    if covariance.len() < 2 || covariance[0].len() < 2 || covariance[1].len() < 2 {
        return "provided".to_string();
    }
    format!(
        "[{:.2}, {:.2}; {:.2}, {:.2}]",
        covariance[0][0], covariance[0][1], covariance[1][0], covariance[1][1]
    )
}

fn selection_title(selection: &CopSelection) -> String {
    format!(
        "{} {}",
        selection.kind.as_label(),
        short_id(&selection.id, 12)
    )
}

#[component]
pub fn EntityInspector(right_dock_width_px: RwSignal<f64>) -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let selected = state.selected_cop_item;
    let inspector_style = move || {
        format!(
            "right: calc({:.0}px + var(--mmp-density-gap-lg) + var(--mmp-density-gap-sm));",
            clamp_dock_width(right_dock_width_px.get())
        )
    };

    view! {
        {move || {
            let Some(selection) = selected.get() else {
                return ().into_any();
            };
            if !selection.pinned {
                return ().into_any();
            }

            view! {
                <aside
                    class="entity-inspector"
                    style=inspector_style
                    aria-label="Selected entity inspector"
                >
                    <div class="entity-inspector-header">
                        <div>
                            <div class="entity-inspector-eyebrow">{selection.kind.as_label()}</div>
                            <h2>{selection_title(&selection)}</h2>
                        </div>
                        <button
                            class="topbar-iconbtn"
                            title="Close inspector"
                            on:click=move |_| selected.set(None)
                        >
                            <span class="material-symbols-rounded" aria-hidden="true">"close"</span>
                        </button>
                    </div>
                    {render_selection_body(&state, &selection)}
                </aside>
            }.into_any()
        }}
    }
}

fn render_selection_body(state: &AppState, selection: &CopSelection) -> AnyView {
    match selection.kind {
        CopItemKind::Track => state
            .tracks
            .with(|tracks| {
                tracks
                    .iter()
                    .find(|track| track.track_id == selection.id)
                    .cloned()
            })
            .map(render_track)
            .unwrap_or_else(|| render_missing(selection)),
        CopItemKind::Detection => state
            .detections
            .with(|detections| {
                detections
                    .iter()
                    .find(|detection| detection.event_id == selection.id)
                    .cloned()
            })
            .map(render_detection)
            .unwrap_or_else(|| render_missing(selection)),
        CopItemKind::Alert => state
            .alerts
            .with(|alerts| {
                alerts
                    .iter()
                    .find(|alert| alert.alert_id == selection.id)
                    .cloned()
            })
            .map(|alert| {
                let payload = if alert.payload.is_object()
                    && !alert.payload.as_object().unwrap().is_empty()
                {
                    serde_json::to_string_pretty(&alert.payload)
                        .unwrap_or_else(|_| "{}".to_string())
                } else {
                    "-".to_string()
                };
                view! {
                    <div class="entity-inspector-body">
                        <Kv label="Status" value=alert.status.unwrap_or_else(|| "-".to_string()) />
                        <Kv label="Severity" value=alert.severity.unwrap_or_else(|| "-".to_string()) />
                        <Kv label="Rule" value=alert.rule_name.unwrap_or_else(|| "-".to_string()) />
                        <Kv label="Detection" value=alert.detection_id.unwrap_or_else(|| "-".to_string()) />
                        <Kv label="Track" value=alert.track_id.unwrap_or_else(|| "-".to_string()) />
                        <Kv label="Destination" value=alert.destination.unwrap_or_else(|| "-".to_string()) />
                        <Kv label="Message" value=alert.message.unwrap_or_else(|| "-".to_string()) />
                        <Kv label="Payload" value=payload />
                    </div>
                }.into_any()
            })
            .unwrap_or_else(|| render_missing(selection)),
        CopItemKind::Node => state
            .nodes
            .with(|nodes| {
                nodes
                    .iter()
                    .find(|node| node.node_id == selection.id)
                    .cloned()
            })
            .map(render_node)
            .unwrap_or_else(|| render_missing(selection)),
        CopItemKind::Effector => state
            .effectors
            .with(|effectors| {
                effectors
                    .iter()
                    .find(|effector| effector.id == selection.id)
                    .cloned()
            })
            .map(render_effector)
            .unwrap_or_else(|| render_missing(selection)),
        CopItemKind::Zone => {
            let occupancy = state.zone_occupancy.get();
            state
                .zones
                .with(|zones| zones.iter().find(|zone| zone.id == selection.id).cloned())
                .map(|zone| render_zone(zone, &occupancy))
                .unwrap_or_else(|| render_missing(selection))
        }
    }
}

fn render_node(node: NodeStatus) -> AnyView {
    let geo = node
        .position_geo
        .as_ref()
        .map(|point| geo_text(point.lat, point.lon, point.alt_m))
        .unwrap_or_else(|| "-".to_string());
    let local_position = node
        .position_m
        .as_ref()
        .filter(|position| position.len() >= 3)
        .map(|position| {
            format!(
                "[{:.2}, {:.2}, {:.2}] m",
                position[0], position[1], position[2]
            )
        })
        .unwrap_or_else(|| "-".to_string());
    let audio_status = node
        .audio_debug
        .as_ref()
        .and_then(|audio| audio.status.clone())
        .unwrap_or_else(|| "-".to_string());
    let rms = node
        .audio_debug
        .as_ref()
        .and_then(|audio| audio.rms)
        .map(|value| format!("{value:.5}"))
        .unwrap_or_else(|| "-".to_string());
    let time_quality = node
        .latest_time_quality
        .clone()
        .unwrap_or_else(|| "-".to_string());

    view! {
        <div class="entity-inspector-body">
            <Kv label="Health" value=node.health />
            <Kv label="Type" value=node.node_type.unwrap_or_else(|| "-".to_string()) />
            <Kv label="Mobility" value=node.mobility.unwrap_or_else(|| "-".to_string()) />
            <Kv label="Geo" value=geo />
            <Kv label="Local" value=local_position />
            <Kv label="Audio" value=audio_status />
            <Kv label="RMS" value=rms />
            <Kv label="Clock" value=time_quality />
        </div>
    }
    .into_any()
}

fn render_effector(effector: Effector) -> AnyView {
    let geo = effector
        .position_geo
        .as_ref()
        .map(|point| geo_text(point.lat, point.lon, point.alt_m))
        .unwrap_or_else(|| "-".to_string());
    let state = effector
        .status
        .as_ref()
        .map(|status| status.state.clone())
        .unwrap_or_else(|| "-".to_string());
    let armed = effector
        .status
        .as_ref()
        .map(|status| status.armed.to_string())
        .unwrap_or_else(|| "-".to_string());
    let pan = effector
        .status
        .as_ref()
        .and_then(|status| status.pan_deg)
        .map(|value| format!("{value:.1} deg"))
        .unwrap_or_else(|| "-".to_string());
    let tilt = effector
        .status
        .as_ref()
        .and_then(|status| status.tilt_deg)
        .map(|value| format!("{value:.1} deg"))
        .unwrap_or_else(|| "-".to_string());

    view! {
        <div class="entity-inspector-body">
            <Kv label="Type" value=effector.effector_type />
            <Kv label="State" value=state />
            <Kv label="Armed" value=armed />
            <Kv label="Geo" value=geo />
            <Kv label="Pan" value=pan />
            <Kv label="Tilt" value=tilt />
            <Kv
                label="Capabilities"
                value=effector.capabilities.unwrap_or_default().join(", ")
            />
        </div>
    }
    .into_any()
}

fn render_zone(zone: ZoneSpec, occupancy: &[crate::state::ZoneOccupancyState]) -> AnyView {
    let occupancy = occupancy.iter().find(|state| state.zone_id == zone.id);
    let occupied = occupancy
        .map(|state| state.occupied.to_string())
        .unwrap_or_else(|| "false".to_string());
    let tracks = occupancy
        .map(|state| state.occupying_track_ids.join(", "))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "-".to_string());
    let labels = occupancy
        .map(|state| state.occupying_labels.join(", "))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "-".to_string());

    view! {
        <div class="entity-inspector-body">
            <Kv label="Name" value=zone.name />
            <Kv label="Type" value=zone.zone_type />
            <Kv label="Occupied" value=occupied />
            <Kv label="Tracks" value=tracks />
            <Kv label="Labels" value=labels />
            <Kv label="Vertices" value=zone.polygon_geo.len().to_string() />
        </div>
    }
    .into_any()
}

fn render_track(track: Track) -> AnyView {
    let geo = track
        .position_geo
        .as_ref()
        .map(|point| geo_text(point.lat, point.lon, point.alt_m))
        .unwrap_or_else(|| "-".to_string());
    view! {
        <div class="entity-inspector-body">
            <Kv label="Label" value=track.label.unwrap_or_else(|| "-".to_string()) />
            <Kv label="Confidence" value=confidence_text(track.confidence) />
            <Kv label="TQI" value=track.tqi.map(|value| format!("{value:.2}")).unwrap_or_else(|| "-".to_string()) />
            <Kv label="Status" value=track.status.unwrap_or_else(|| "-".to_string()) />
            <Kv label="Geo" value=geo />
            <Kv label="Covariance" value=covariance_text(track.position_covariance_m2.as_ref()) />
            <Kv label="Contributors" value=track.contributor_count.to_string() />
        </div>
    }.into_any()
}

fn render_detection(detection: Detection) -> AnyView {
    let display_mode = detection.spatial_display_mode().to_string();
    let geo = detection
        .position_geo
        .as_ref()
        .map(|point| geo_text(point.lat, point.lon, point.alt_m))
        .unwrap_or_else(|| "-".to_string());
    view! {
        <div class="entity-inspector-body">
            <Kv label="Label" value=detection.label.unwrap_or_else(|| "-".to_string()) />
            <Kv label="Confidence" value=confidence_text(detection.label_confidence.or(detection.confidence)) />
            <Kv label="Mode" value=display_mode />
            <Kv label="Method" value=detection.localization_method.unwrap_or_else(|| "-".to_string()) />
            <Kv label="Capability" value=detection.capability_tier.unwrap_or_else(|| "-".to_string()) />
            <Kv label="Node" value=detection.node_id.unwrap_or_else(|| "-".to_string()) />
            <Kv label="Geo" value=geo />
            <Kv label="Covariance" value=covariance_text(detection.position_covariance_m2.as_ref()) />
        </div>
    }.into_any()
}

fn render_missing(selection: &CopSelection) -> AnyView {
    view! {
        <div class="entity-inspector-body">
            <p class="muted">
                {format!("{} {} is no longer in the live feed.", selection.kind.as_label(), selection.id)}
            </p>
        </div>
    }.into_any()
}

#[component]
fn Kv(label: &'static str, value: String) -> impl IntoView {
    view! {
        <div class="entity-inspector-kv">
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
    }
}
