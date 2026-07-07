use crate::devices::schema::DeviceKind;
use crate::map::bindings::{
    ensure_layer, remove_bearing_wedge, remove_layer, set_bearing_wedge, set_layer_data,
};
use crate::state::AppState;
use leptos::prelude::*;
use std::collections::HashSet;

const FUTURE_MODALITIES_LAYER_ID: &str = "future-modalities";

fn future_modalities_layer_spec() -> serde_json::Value {
    serde_json::json!({
        "type": "geojson",
        "layer": {
            "type": "circle",
            "paint": {
                "circle-color": [
                    "match", ["get", "kind"],
                    "rf_emitter", "#4cc9f0",
                    "sdr_rf", "#4cc9f0",
                    "radar_24g", "#ffb703",
                    "seismic", "#f28482",
                    "speech_node", "#2ec4b6",
                    "#b8c0ff"
                ],
                "circle-radius": [
                    "case",
                    ["==", ["get", "surface"], "emitter"], 7,
                    6
                ],
                "circle-opacity": 0.78,
                "circle-stroke-color": "#111827",
                "circle-stroke-width": 1.8
            }
        }
    })
}

pub fn mount(state: &AppState) {
    let devices = state.devices;
    let rf_spectra = state.modality.rf_spectra;
    let seismic_traces = state.modality.seismic_traces;
    let speech_lines = state.modality.speech_lines;
    let map_layers = state.map_layers;
    let theme = state.theme;

    Effect::new(move |prev_wedges: Option<HashSet<String>>| {
        let _ = theme.get();
        let visible = map_layers.get().future_modalities;
        let registered_devices = devices.get();
        let rf_frames = rf_spectra.get();
        let seismic_trace_device_ids = seismic_traces
            .get()
            .into_iter()
            .map(|trace| trace.device_id)
            .collect::<HashSet<_>>();
        let speech_line_device_ids = speech_lines
            .get()
            .into_iter()
            .map(|line| line.device_id)
            .collect::<HashSet<_>>();

        let mut features = Vec::new();
        let mut current_wedges = HashSet::new();

        if visible {
            for device in &registered_devices {
                let Some(lat) = device.lat else {
                    continue;
                };
                let Some(lon) = device.lon else {
                    continue;
                };
                let active = match device.kind {
                    DeviceKind::Radar24G | DeviceKind::SdrRf => {
                        rf_frames.iter().any(|frame| frame.device_id == device.id)
                    }
                    DeviceKind::Seismic => seismic_trace_device_ids.contains(&device.id),
                    DeviceKind::SpeechNode => speech_line_device_ids.contains(&device.id),
                };

                features.push(serde_json::json!({
                    "type": "Feature",
                    "properties": {
                        "id": device.id,
                        "label": device.display_label(),
                        "kind": device.kind.as_str(),
                        "surface": "device",
                        "active": active,
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                }));

                if device.kind == DeviceKind::Radar24G {
                    let wedge_id = format!("modality-radar:{}", device.id);
                    let bearing = device.yaw_deg.unwrap_or(0.0).rem_euclid(360.0);
                    set_bearing_wedge(&wedge_id, lat, lon, bearing, 24.0, 240.0);
                    current_wedges.insert(wedge_id);
                }
            }

            for frame in rf_frames {
                for emitter in frame.emitters {
                    let (Some(lat), Some(lon)) = (emitter.lat, emitter.lon) else {
                        continue;
                    };
                    features.push(serde_json::json!({
                        "type": "Feature",
                        "properties": {
                            "id": emitter.emitter_id,
                            "label": emitter.emitter_id,
                            "kind": "rf_emitter",
                            "surface": "emitter",
                            "rssi_dbm": emitter.rssi_dbm,
                            "drone_likelihood": emitter.drone_likelihood,
                            "source_device_id": frame.device_id,
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat],
                        },
                    }));
                }
            }
        }

        if features.is_empty() {
            remove_layer(FUTURE_MODALITIES_LAYER_ID);
        } else {
            let collection = serde_json::json!({
                "type": "FeatureCollection",
                "features": features,
            });
            if let (Ok(spec), Ok(data)) = (
                serde_wasm_bindgen::to_value(&future_modalities_layer_spec()),
                serde_wasm_bindgen::to_value(&collection),
            ) {
                ensure_layer(FUTURE_MODALITIES_LAYER_ID, &spec);
                set_layer_data(FUTURE_MODALITIES_LAYER_ID, &data);
            }
        }

        if let Some(ref previous) = prev_wedges {
            for id in previous.difference(&current_wedges) {
                remove_bearing_wedge(id);
            }
        }

        current_wedges
    });
}
