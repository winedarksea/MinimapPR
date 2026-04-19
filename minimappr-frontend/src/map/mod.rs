pub mod bindings;

use crate::state::AppState;
use bindings::*;
use leptos::prelude::*;

#[component]
pub fn LeafletMapPanel() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let nodes = state.nodes;
    let tracks = state.tracks;
    let detections = state.detections;
    let alerts = state.alerts;
    let cop = state.cop_status;
    let theme = state.theme;

    // Init Leaflet once after the component first mounts.
    Effect::new(move |init_done: Option<bool>| {
        if init_done.is_none() {
            init(44.987, -93.258, 17);
        }
        true
    });

    // Sync nodes → map markers (geodetic mode only; position_m is local-frame, use position_geo)
    {
        let config = state.config;
        Effect::new(move |_| {
            let _ = theme.get();
            let ns = nodes.get();
            let is_geo = config
                .get()
                .map(|c| c.coordinate_mode == "geodetic")
                .unwrap_or(false);
            if is_geo {
                for n in &ns {
                    if let Some(geo) = &n.position_geo {
                        set_node_marker(&n.node_id, geo.lat, geo.lon, &n.health);
                    }
                }
            }
        });
    }

    // Sync tracks → map markers
    {
        Effect::new(move |_| {
            let _ = theme.get();
            let ts = tracks.get();
            for t in &ts {
                if let Some(geo) = &t.position_geo {
                    let label = t.label.as_deref().unwrap_or("");
                    let tqi = t.tqi.unwrap_or(0.0);
                    set_track_marker(&t.track_id, geo.lat, geo.lon, label, tqi);
                    if let Some(vel) = &t.velocity_mps {
                        if vel.len() >= 2 {
                            // Rough local velocity → geo delta (1 m ≈ 9e-6 deg)
                            let dlat = vel[1] * 9e-6;
                            let dlon = vel[0] * 9e-6 / (geo.lat.to_radians().cos()).max(0.01);
                            set_track_velocity_vector(&t.track_id, geo.lat, geo.lon, dlat, dlon);
                        }
                    }
                }
            }
        });
    }

    // Sync detections → map markers (newest event only, JS shim auto-removes after 30s)
    {
        Effect::new(move |_| {
            let _ = theme.get();
            let ds = detections.get();
            if let Some(d) = ds.front() {
                if let Some(geo) = &d.position_geo {
                    let label = d.label.as_deref().unwrap_or("detection");
                    add_detection_marker(&d.event_id, geo.lat, geo.lon, label);
                }
            }
        });
    }

    view! {
        <div class="panel map-column">
            <div class="panel-header">"Map"</div>
            <div class="leaflet-container-wrap">
                <div id="leaflet-map"></div>
                <div class="map-overlay-stack map-overlay-top-left">
                    <section class="map-floating-panel map-picture-card">
                        <div class="map-floating-title">"Tactical Picture"</div>
                        <div class="map-chip-row">
                            <span class="strip-chip info">
                                <span class="label">"Tracks"</span>
                                <span class="value">{move || {
                                    cop.get()
                                        .map(|status| status.active_tracks)
                                        .unwrap_or_else(|| tracks.get().len() as u32)
                                }}</span>
                            </span>
                            <span class="strip-chip ok">
                                <span class="label">"Online"</span>
                                <span class="value">{move || {
                                    cop.get()
                                        .map(|status| status.active_nodes)
                                        .unwrap_or_else(|| nodes.get().len() as u32)
                                }}</span>
                            </span>
                            <span class="strip-chip danger">
                                <span class="label">"Alerts"</span>
                                <span class="value">{move || {
                                    cop.get()
                                        .map(|status| status.open_alerts)
                                        .unwrap_or_else(|| alerts.get().len() as u32)
                                }}</span>
                            </span>
                        </div>
                        <div class="map-floating-caption">
                            {move || {
                                let latest_label = detections.get()
                                    .front()
                                    .and_then(|detection| detection.label.clone())
                                    .unwrap_or_else(|| "No live detections".to_string());
                                format!("Latest cue: {latest_label}")
                            }}
                        </div>
                    </section>
                </div>
                <div class="map-overlay-stack map-overlay-bottom-right">
                    <section class="map-floating-panel">
                        <div class="map-floating-title">"Overlay Legend"</div>
                        <div class="legend">
                            <span class="legend-item">
                                <span class="legend-dot node"></span>
                                "Nodes"
                            </span>
                            <span class="legend-item">
                                <span class="legend-dot track"></span>
                                "Tracks"
                            </span>
                            <span class="legend-item">
                                <span class="legend-dot detection"></span>
                                "Detections"
                            </span>
                        </div>
                    </section>
                </div>
            </div>
        </div>
    }
}
