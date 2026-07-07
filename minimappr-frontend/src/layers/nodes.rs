use crate::map::bindings::{remove_node_marker, set_node_marker, set_theme};
use crate::state::AppState;
use leptos::prelude::*;
use std::collections::HashSet;

pub fn mount(state: &AppState) {
    let nodes = state.nodes;
    let map_layers = state.map_layers;
    let theme = state.theme;
    let config = state.config;

    Effect::new(move |prev_ids: Option<HashSet<String>>| {
        let active_theme = theme.get();
        set_theme(&active_theme);
        let is_geo = config
            .get()
            .map(|c| c.coordinate_mode == "geodetic")
            .unwrap_or(false);
        let visible = map_layers.get().nodes;

        nodes.with(|ns| {
            let current_ids: HashSet<String> = if is_geo && visible {
                ns.iter()
                    .filter(|node| node.position_geo.is_some())
                    .map(|node| node.node_id.clone())
                    .collect()
            } else {
                HashSet::new()
            };

            if let Some(ref prev) = prev_ids {
                for id in prev.difference(&current_ids) {
                    remove_node_marker(id);
                }
            }

            if is_geo && visible {
                for node in ns {
                    if let Some(geo) = &node.position_geo {
                        set_node_marker(&node.node_id, geo.lat, geo.lon, &node.health);
                    }
                }
            }

            current_ids
        })
    });
}
