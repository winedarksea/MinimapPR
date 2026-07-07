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

pub fn mount(state: &AppState) {
    let live_heatmap_enabled = state.live_heatmap_enabled;
    let live_heatmap_window = state.live_heatmap_window;
    let live_heatmap_refresh_tick = state.live_heatmap_refresh_tick;
    let live_heatmap_bin_count = state.live_heatmap_bin_count;
    let live_heatmap_error = state.live_heatmap_error;

    Effect::new(move |_| {
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
                    clear_heatmap();
                    live_heatmap_bin_count.set(0);
                    live_heatmap_error.set(Some(message));
                }
            }
        });
    });
}
