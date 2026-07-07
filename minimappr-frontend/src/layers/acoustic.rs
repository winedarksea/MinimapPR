use crate::map::bindings::{ensure_layer, remove_layer, set_layer_data};
use crate::state::AppState;
use leptos::prelude::*;

const ACOUSTIC_MAP_LAYER_ID: &str = "acoustic-map";

fn acoustic_layer_spec() -> serde_json::Value {
    serde_json::json!({
        "type": "geojson",
        "layer": {
            "type": "circle",
            "paint": {
                "circle-color": [
                    "interpolate", ["linear"], ["get", "value"],
                    0.0, "#2ec4b6",
                    0.5, "#ffbf4a",
                    1.0, "#ff5a5f"
                ],
                "circle-radius": [
                    "interpolate", ["linear"], ["get", "value"],
                    0.0, 6,
                    1.0, 18
                ],
                "circle-opacity": 0.38,
                "circle-stroke-color": "#111827",
                "circle-stroke-width": 0.8
            }
        }
    })
}

pub fn mount(state: &AppState) {
    let acoustic_maps = state.modality.acoustic_maps;
    let map_layers = state.map_layers;
    let theme = state.theme;

    Effect::new(move |_| {
        let _ = theme.get();
        let visible = map_layers.get().acoustic;
        let maps = acoustic_maps.get();
        if !visible || maps.is_empty() {
            remove_layer(ACOUSTIC_MAP_LAYER_ID);
            return;
        }

        let features = maps
            .iter()
            .flat_map(|map_layer| {
                map_layer.samples.iter().map(|sample| {
                    serde_json::json!({
                        "type": "Feature",
                        "properties": {
                            "id": map_layer.layer_id,
                            "data_type": map_layer.data_type,
                            "time_window": map_layer.time_window,
                            "storey": map_layer.storey,
                            "value": sample.value,
                            "updated_ns": map_layer.updated_ns,
                            "mock": true,
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [sample.lon, sample.lat],
                        },
                    })
                })
            })
            .collect::<Vec<_>>();

        if features.is_empty() {
            remove_layer(ACOUSTIC_MAP_LAYER_ID);
            return;
        }

        let collection = serde_json::json!({
            "type": "FeatureCollection",
            "features": features,
        });
        if let (Ok(spec), Ok(data)) = (
            serde_wasm_bindgen::to_value(&acoustic_layer_spec()),
            serde_wasm_bindgen::to_value(&collection),
        ) {
            ensure_layer(ACOUSTIC_MAP_LAYER_ID, &spec);
            set_layer_data(ACOUSTIC_MAP_LAYER_ID, &data);
        }
    });
}
