use crate::map::bindings::{remove_node_omni_halo, set_node_omni_halo};
use crate::state::AppState;
use leptos::prelude::*;
use std::collections::HashSet;

pub fn mount(state: &AppState) {
    let nodes = state.nodes;
    let omni_detection_summaries = state.omni_detection_summaries;
    let map_layers = state.map_layers;
    let theme = state.theme;

    Effect::new(move |prev_ids: Option<HashSet<String>>| {
        let _ = theme.get();
        let visible = map_layers.get().omni;
        let node_positions = nodes.with(|ns| {
            ns.iter()
                .filter_map(|node| {
                    node.position_geo
                        .as_ref()
                        .map(|geo| (node.node_id.clone(), geo.lat, geo.lon))
                })
                .collect::<Vec<_>>()
        });
        omni_detection_summaries.with(|summaries| {
            let mut current_ids = HashSet::new();
            if visible {
                for summary in summaries {
                    let Some((_, lat, lon)) = node_positions
                        .iter()
                        .find(|(node_id, _, _)| node_id == &summary.node_id)
                    else {
                        continue;
                    };
                    current_ids.insert(summary.node_id.clone());
                    if let Ok(summary_js) = serde_wasm_bindgen::to_value(summary) {
                        set_node_omni_halo(&summary.node_id, *lat, *lon, &summary_js);
                    }
                }
            }

            if let Some(ref prev) = prev_ids {
                for id in prev.difference(&current_ids) {
                    remove_node_omni_halo(id);
                }
            }

            current_ids
        })
    });
}
