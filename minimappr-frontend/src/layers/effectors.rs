use crate::map::bindings::{remove_effector_marker, set_effector_marker};
use crate::state::AppState;
use leptos::prelude::*;
use std::collections::HashSet;

pub fn mount(state: &AppState) {
    let effectors = state.effectors;
    let map_layers = state.map_layers;
    let theme = state.theme;

    Effect::new(move |prev_ids: Option<HashSet<String>>| {
        let _ = theme.get();
        let visible = map_layers.get().effectors;
        effectors.with(|es| {
            let current_ids: HashSet<String> = if visible {
                es.iter()
                    .filter(|e| e.position_geo.is_some())
                    .map(|e| e.id.clone())
                    .collect()
            } else {
                HashSet::new()
            };

            if let Some(ref prev) = prev_ids {
                for id in prev.difference(&current_ids) {
                    remove_effector_marker(id);
                }
            }

            if visible {
                for e in es {
                    let Some(geo) = &e.position_geo else {
                        continue;
                    };
                    // pan_deg is relative to the registered home yaw; the
                    // wedge needs an absolute compass bearing.
                    let home_yaw = e.orientation.as_ref().map(|o| o.yaw_deg).unwrap_or(0.0);
                    let pan = e.status.as_ref().and_then(|s| s.pan_deg).unwrap_or(0.0);
                    let bearing = (home_yaw + pan).rem_euclid(360.0);
                    let effector_state = e
                        .status
                        .as_ref()
                        .map(|s| s.state.as_str())
                        .unwrap_or("offline");
                    set_effector_marker(&e.id, geo.lat, geo.lon, bearing, effector_state);
                }
            }

            current_ids
        })
    });
}
