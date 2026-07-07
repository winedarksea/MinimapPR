use crate::map::bindings::{
    clear_all_cop_uncertainty, clear_cop_highlight, highlight_cop_item, set_cop_uncertainty,
};
use crate::state::{AppState, CopItemKind};
use leptos::prelude::*;
use wasm_bindgen::JsValue;

fn covariance_to_js_value(covariance: &[Vec<f64>]) -> JsValue {
    serde_wasm_bindgen::to_value(covariance).unwrap_or(JsValue::NULL)
}

pub fn mount(state: &AppState) {
    let selected_cop_item = state.selected_cop_item;
    let tracks = state.tracks;
    let detections = state.detections;

    Effect::new(move |_| match selected_cop_item.get() {
        Some(selection) => {
            highlight_cop_item(selection.kind.as_js_kind(), &selection.id);
            clear_all_cop_uncertainty();
            match selection.kind {
                CopItemKind::Track => {
                    tracks.with(|current_tracks| {
                        if let Some(track) = current_tracks
                            .iter()
                            .find(|track| track.track_id == selection.id)
                        {
                            if let (Some(geo), Some(covariance)) =
                                (&track.position_geo, &track.position_covariance_m2)
                            {
                                let covariance_js = covariance_to_js_value(covariance);
                                set_cop_uncertainty(
                                    "track",
                                    &track.track_id,
                                    geo.lat,
                                    geo.lon,
                                    &covariance_js,
                                );
                            }
                        }
                    });
                }
                CopItemKind::Detection => {
                    detections.with(|current_detections| {
                        if let Some(detection) = current_detections
                            .iter()
                            .find(|detection| detection.event_id == selection.id)
                        {
                            if let (Some(geo), Some(covariance)) =
                                (&detection.position_geo, &detection.position_covariance_m2)
                            {
                                let covariance_js = covariance_to_js_value(covariance);
                                set_cop_uncertainty(
                                    "detection",
                                    &detection.event_id,
                                    geo.lat,
                                    geo.lon,
                                    &covariance_js,
                                );
                            }
                        }
                    });
                }
                CopItemKind::Alert
                | CopItemKind::Node
                | CopItemKind::Effector
                | CopItemKind::Zone => {}
            }
        }
        None => {
            clear_cop_highlight();
            clear_all_cop_uncertainty();
        }
    });
}
