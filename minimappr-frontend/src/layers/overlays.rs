use crate::map::bindings::{remove_map_overlay, set_geojson_overlay, set_image_overlay};
use crate::state::AppState;
use gloo_net::http::Request;
use leptos::prelude::*;
use leptos::task::spawn_local;
use std::collections::HashSet;

fn overlay_layer_id(overlay_id: &str) -> String {
    format!("overlay:{overlay_id}")
}

fn overlay_has_four_corners(bounds: &[Vec<f64>]) -> bool {
    bounds.len() == 4 && bounds.iter().all(|corner| corner.len() == 2)
}

fn normalize_geojson_overlay(value: serde_json::Value) -> Option<serde_json::Value> {
    let geometry_type = value
        .get("type")
        .and_then(|item| item.as_str())
        .map(ToString::to_string);
    match geometry_type.as_deref() {
        Some("FeatureCollection") => Some(value),
        Some("Feature") => Some(serde_json::json!({
            "type": "FeatureCollection",
            "features": [value],
        })),
        Some(
            "Point" | "MultiPoint" | "LineString" | "MultiLineString" | "Polygon" | "MultiPolygon",
        ) => Some(serde_json::json!({
            "type": "FeatureCollection",
            "features": [{ "type": "Feature", "properties": {}, "geometry": value }],
        })),
        _ => None,
    }
}

pub fn mount(state: &AppState) {
    let overlays = state.overlays;
    let map_layers = state.map_layers;
    let theme = state.theme;

    Effect::new(move |prev_ids: Option<HashSet<String>>| {
        let _ = theme.get();
        let visible = map_layers.get().overlays;
        overlays.with(|current_overlays| {
            let mut current_ids = HashSet::new();

            if visible {
                for overlay in current_overlays {
                    let layer_id = overlay_layer_id(&overlay.id);
                    if !overlay.enabled {
                        remove_map_overlay(&layer_id);
                        continue;
                    }

                    match overlay.kind.as_str() {
                        "image" | "svg" if overlay_has_four_corners(&overlay.bounds) => {
                            if let Ok(corners) = serde_wasm_bindgen::to_value(&overlay.bounds) {
                                set_image_overlay(
                                    &layer_id,
                                    &overlay.content_url,
                                    &corners,
                                    overlay.opacity,
                                );
                                current_ids.insert(layer_id);
                            }
                        }
                        "geojson" => {
                            current_ids.insert(layer_id.clone());
                            let url = overlay.content_url.clone();
                            let opacity = overlay.opacity;
                            spawn_local(async move {
                                match Request::get(&url).send().await {
                                    Ok(resp) if resp.ok() => {
                                        match resp.json::<serde_json::Value>().await {
                                            Ok(value) => {
                                                if let Some(collection) =
                                                    normalize_geojson_overlay(value)
                                                {
                                                    if let Ok(js_value) =
                                                        serde_wasm_bindgen::to_value(&collection)
                                                    {
                                                        set_geojson_overlay(
                                                            &layer_id, &js_value, opacity,
                                                        );
                                                    }
                                                }
                                            }
                                            Err(error) => {
                                                log::warn!("overlay GeoJSON parse failed: {error}");
                                            }
                                        }
                                    }
                                    Ok(resp) => {
                                        log::warn!("overlay GeoJSON fetch HTTP {}", resp.status());
                                    }
                                    Err(error) => {
                                        log::warn!("overlay GeoJSON fetch failed: {error}");
                                    }
                                }
                            });
                        }
                        _ => {
                            remove_map_overlay(&layer_id);
                        }
                    }
                }
            }

            if let Some(ref prev) = prev_ids {
                for id in prev.difference(&current_ids) {
                    remove_map_overlay(id);
                }
            }

            current_ids
        })
    });
}
