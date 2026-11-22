use crate::map::bindings::{remove_track, set_track_marker, set_track_velocity_vector};
use crate::state::AppState;
use leptos::prelude::*;
use std::collections::HashSet;

const APPROX_DEGREES_LATITUDE_PER_METER: f64 = 9e-6;
const MIN_VELOCITY_LONGITUDE_COSINE: f64 = 0.01;

fn velocity_mps_to_geo_delta(lat: f64, velocity_mps: &[f64]) -> Option<(f64, f64)> {
    if velocity_mps.len() < 2 {
        return None;
    }
    let dlat = velocity_mps[1] * APPROX_DEGREES_LATITUDE_PER_METER;
    let dlon = velocity_mps[0] * APPROX_DEGREES_LATITUDE_PER_METER
        / lat.to_radians().cos().max(MIN_VELOCITY_LONGITUDE_COSINE);
    Some((dlat, dlon))
}

pub fn mount(state: &AppState) {
    let tracks = state.tracks;
    let map_layers = state.map_layers;
    let theme = state.theme;
    let bird_entities = state.bird_entities;

    Effect::new(move |prev_ids: Option<HashSet<String>>| {
        let _ = theme.get();
        let visible = map_layers.get().tracks;
        // One bird, one thing on screen: a track bound to a bird entity is drawn
        // by the entity layer instead.
        let entity_track_ids: HashSet<String> = bird_entities
            .with(|entities| entities.iter().filter_map(|e| e.track_id.clone()).collect());
        tracks.with(|ts| {
            let current_ids: HashSet<String> = if visible {
                ts.iter()
                    .filter(|t| !entity_track_ids.contains(&t.track_id))
                    .map(|t| t.track_id.clone())
                    .collect()
            } else {
                HashSet::new()
            };

            if let Some(ref prev) = prev_ids {
                for id in prev.difference(&current_ids) {
                    remove_track(id);
                }
            }

            if visible {
                for t in ts.iter().filter(|t| current_ids.contains(&t.track_id)) {
                    let Some(geo) = &t.position_geo else {
                        continue;
                    };
                    let label = t.label.as_deref().unwrap_or("");
                    let tqi = t.tqi.unwrap_or(0.0);
                    let status = t.status.as_deref().unwrap_or("active");
                    let last_update_ns = t.last_update_ns.unwrap_or(i64::MIN) as f64;
                    let track_kind = t.track_kind.as_deref().unwrap_or("acoustic");
                    set_track_marker(
                        &t.track_id,
                        geo.lat,
                        geo.lon,
                        label,
                        tqi,
                        status,
                        last_update_ns,
                        track_kind,
                    );
                    if let Some((dlat, dlon)) = t
                        .velocity_mps
                        .as_deref()
                        .and_then(|velocity| velocity_mps_to_geo_delta(geo.lat, velocity))
                    {
                        set_track_velocity_vector(
                            &t.track_id,
                            geo.lat,
                            geo.lon,
                            dlat,
                            dlon,
                            status,
                        );
                    }
                }
            }

            current_ids
        })
    });
}
