use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "init")]
    pub fn init(lat: f64, lon: f64, zoom: u32);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "setNodeMarker")]
    pub fn set_node_marker(node_id: &str, lat: f64, lon: f64, health_class: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "removeNodeMarker")]
    pub fn remove_node_marker(node_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "addDetectionMarker")]
    pub fn add_detection_marker(event_id: &str, lat: f64, lon: f64, label: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "setTrackMarker")]
    pub fn set_track_marker(track_id: &str, lat: f64, lon: f64, label: &str, tqi: f64);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "setTrackVelocityVector")]
    pub fn set_track_velocity_vector(track_id: &str, lat: f64, lon: f64, vel_lat: f64, vel_lon: f64);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "removeTrack")]
    pub fn remove_track(track_id: &str);

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

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "setHeatmapPoints")]
    pub fn set_heatmap_points(points: &JsValue, max_intensity: f64);

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "clearHeatmap")]
    pub fn clear_heatmap();

    #[wasm_bindgen(js_namespace = ["globalThis", "leafletInterop"], js_name = "fitBoundsLatLons")]
    pub fn fit_bounds_latlons(points: &JsValue);
}
