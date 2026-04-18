pub mod bindings;

use crate::state::AppState;
use bindings::*;
use leptos::prelude::*;

#[component]
pub fn LeafletMapPanel() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");

    // Init Leaflet once after the component first mounts.
    Effect::new(move |init_done: Option<bool>| {
        if init_done.is_none() {
            init(44.987, -93.258, 17);
        }
        true
    });

    // Sync nodes → map markers (geodetic mode only; position_m is local-frame, use position_geo)
    {
        let nodes = state.nodes;
        let config = state.config;
        Effect::new(move |_| {
            let ns = nodes.get();
            let is_geo = config.get()
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
        let tracks = state.tracks;
        Effect::new(move |_| {
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
                            let dlon = vel[0] * 9e-6
                                / (geo.lat.to_radians().cos()).max(0.01);
                            set_track_velocity_vector(
                                &t.track_id, geo.lat, geo.lon, dlat, dlon,
                            );
                        }
                    }
                }
            }
        });
    }

    // Sync detections → map markers (newest event only, JS shim auto-removes after 30s)
    {
        let dets = state.detections;
        Effect::new(move |_| {
            let ds = dets.get();
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
            </div>
            <div class="map-toolbar">
                <div class="legend">
                    <span class="legend-item">
                        <span class="legend-dot" style="background:#58a6ff"></span>
                        "Nodes"
                    </span>
                    <span class="legend-item">
                        <span class="legend-dot" style="background:#3fb950"></span>
                        "Tracks"
                    </span>
                    <span class="legend-item">
                        <span class="legend-dot" style="background:#f78166"></span>
                        "Detections"
                    </span>
                </div>
            </div>
        </div>
    }
}
