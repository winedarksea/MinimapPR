use crate::map::bindings::{remove_zone, set_zone};
use crate::state::AppState;
use leptos::prelude::*;
use std::collections::HashSet;

pub fn mount(state: &AppState) {
    let zones = state.zones;
    let map_layers = state.map_layers;
    let theme = state.theme;

    Effect::new(move |prev_ids: Option<HashSet<String>>| {
        let _ = theme.get();
        let visible = map_layers.get().zones;
        zones.with(|current_zones| {
            let current_ids: HashSet<String> = if visible {
                current_zones.iter().map(|zone| zone.id.clone()).collect()
            } else {
                HashSet::new()
            };

            if let Some(ref prev) = prev_ids {
                for id in prev.difference(&current_ids) {
                    remove_zone(id);
                }
            }

            if visible {
                for zone in current_zones {
                    if zone.polygon_geo.len() >= 3 {
                        if let Ok(latlngs) = serde_wasm_bindgen::to_value(&zone.polygon_geo) {
                            set_zone(&zone.id, &latlngs, &zone.name);
                        }
                    }
                }
            }

            current_ids
        })
    });
}
