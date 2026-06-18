use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "init")]
    pub fn init(lat: f64, lon: f64, zoom: u32);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "setNodeMarker")]
    pub fn set_node_marker(node_id: &str, lat: f64, lon: f64, health_class: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "removeNodeMarker")]
    pub fn remove_node_marker(node_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "setNodeOmniHalo")]
    pub fn set_node_omni_halo(node_id: &str, lat: f64, lon: f64, summary: &JsValue);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "removeNodeOmniHalo")]
    pub fn remove_node_omni_halo(node_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "triggerNodeOmniRipple")]
    pub fn trigger_node_omni_ripple(node_id: &str, lat: f64, lon: f64, label: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "addDetectionMarker")]
    pub fn add_detection_marker(event_id: &str, lat: f64, lon: f64, label: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "addBearingOnlyDetectionMarker")]
    pub fn add_bearing_only_detection_marker(
        event_id: &str,
        lat: f64,
        lon: f64,
        label: &str,
        source_lat: f64,
        source_lon: f64,
        has_source: bool,
    );

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "removeDetectionMarker")]
    pub fn remove_detection_marker(event_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "setTrackMarker")]
    pub fn set_track_marker(
        track_id: &str,
        lat: f64,
        lon: f64,
        label: &str,
        tqi: f64,
        status: &str,
    );

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "setTrackVelocityVector")]
    pub fn set_track_velocity_vector(
        track_id: &str,
        lat: f64,
        lon: f64,
        vel_lat: f64,
        vel_lon: f64,
        status: &str,
    );

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "removeTrack")]
    pub fn remove_track(track_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "pulseTrackMarker")]
    pub fn pulse_track_marker(track_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "setZone")]
    pub fn set_zone(zone_id: &str, latlngs: &JsValue, label: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "removeZone")]
    pub fn remove_zone(zone_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "setGdopCircle")]
    pub fn set_gdop_circle(key: &str, lat: f64, lon: f64, radius_m: f64);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "removeGdopCircle")]
    pub fn remove_gdop_circle(key: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "panTo")]
    pub fn pan_to(lat: f64, lon: f64);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "highlightCopItem")]
    pub fn highlight_cop_item(kind: &str, id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "clearCopHighlight")]
    pub fn clear_cop_highlight();

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "setCopUncertainty")]
    pub fn set_cop_uncertainty(kind: &str, id: &str, lat: f64, lon: f64, covariance: &JsValue);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "clearAllCopUncertainty")]
    pub fn clear_all_cop_uncertainty();

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "setCopSelectionCallback")]
    pub fn set_cop_selection_callback(callback: &js_sys::Function);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "setCopSelectionCallback")]
    pub fn set_cop_selection_callback_value(callback: &JsValue);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "setHeatmapPoints")]
    pub fn set_heatmap_points(points: &JsValue, max_intensity: f64);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "clearHeatmap")]
    pub fn clear_heatmap();

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "fitBoundsLatLons")]
    pub fn fit_bounds_latlons(points: &JsValue);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "invalidateMapSize")]
    pub fn invalidate_map_size();
}
