use crate::map::bindings::{clear_heatmap, set_heatmap_points};
use crate::state::AppState;
use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::JsValue;

fn heatmap_bins_to_js_points(bins: &[crate::api::HeatmapBin]) -> JsValue {
    let points = js_sys::Array::new();
    for bin in bins {
        let triple = js_sys::Array::new();
        triple.push(&JsValue::from_f64(bin.lat));
        triple.push(&JsValue::from_f64(bin.lon));
        triple.push(&JsValue::from_f64(bin.weight as f64));
        points.push(&triple);
    }
    points.into()
}

fn heatmap_response_is_current(
    heatmap_is_enabled: bool,
    current_request_generation: u64,
    response_request_generation: u64,
) -> bool {
    heatmap_is_enabled && current_request_generation == response_request_generation
}

pub fn mount(state: &AppState) {
    let live_heatmap_enabled = state.live_heatmap_enabled;
    let live_heatmap_window = state.live_heatmap_window;
    let live_heatmap_refresh_tick = state.live_heatmap_refresh_tick;
    let live_heatmap_request_generation = state.live_heatmap_request_generation;
    let live_heatmap_bin_count = state.live_heatmap_bin_count;
    let live_heatmap_error = state.live_heatmap_error;

    Effect::new(move |_| {
        // Advancing this token also invalidates a request when the layer is
        // disabled, preventing a late response from repainting the COP map.
        let response_request_generation = live_heatmap_request_generation
            .get_untracked()
            .wrapping_add(1);
        live_heatmap_request_generation.set(response_request_generation);
        let enabled = live_heatmap_enabled.get();
        let window = live_heatmap_window.get();
        if !enabled {
            clear_heatmap();
            live_heatmap_bin_count.set(0);
            live_heatmap_error.set(None);
            return;
        }

        let _refresh_tick = live_heatmap_refresh_tick.get();
        live_heatmap_error.set(None);
        spawn_local(async move {
            match crate::api::fetch_heatmap(&window).await {
                Ok(response) => {
                    if !heatmap_response_is_current(
                        live_heatmap_enabled.get_untracked(),
                        live_heatmap_request_generation.get_untracked(),
                        response_request_generation,
                    ) {
                        return;
                    }
                    if response.bins.is_empty() {
                        clear_heatmap();
                        live_heatmap_bin_count.set(0);
                    } else {
                        let max_weight = response
                            .bins
                            .iter()
                            .map(|bin| bin.weight)
                            .max()
                            .unwrap_or(1)
                            .max(1);
                        let points = heatmap_bins_to_js_points(&response.bins);
                        set_heatmap_points(&points, max_weight as f64);
                        live_heatmap_bin_count.set(response.bins.len());
                    }
                }
                Err(message) => {
                    if !heatmap_response_is_current(
                        live_heatmap_enabled.get_untracked(),
                        live_heatmap_request_generation.get_untracked(),
                        response_request_generation,
                    ) {
                        return;
                    }
                    clear_heatmap();
                    live_heatmap_bin_count.set(0);
                    live_heatmap_error.set(Some(message));
                }
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::heatmap_response_is_current;

    #[test]
    fn disabled_heatmap_rejects_an_in_flight_response() {
        assert!(!heatmap_response_is_current(false, 4, 4));
    }

    #[test]
    fn superseded_heatmap_request_is_rejected() {
        assert!(!heatmap_response_is_current(true, 5, 4));
    }

    #[test]
    fn current_enabled_heatmap_request_is_accepted() {
        assert!(heatmap_response_is_current(true, 4, 4));
    }
}
