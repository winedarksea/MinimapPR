use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "init")]
    pub fn init(lat: f64, lon: f64, zoom: u32);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "disposeCopMap")]
    pub fn dispose_cop_map();

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setTheme")]
    pub fn set_theme(theme: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setNodeMarker")]
    pub fn set_node_marker(node_id: &str, lat: f64, lon: f64, health_class: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "removeNodeMarker")]
    pub fn remove_node_marker(node_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setNodeOmniHalo")]
    pub fn set_node_omni_halo(node_id: &str, lat: f64, lon: f64, summary: &JsValue);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "removeNodeOmniHalo")]
    pub fn remove_node_omni_halo(node_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "triggerNodeOmniRipple")]
    pub fn trigger_node_omni_ripple(node_id: &str, lat: f64, lon: f64, label: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "addDetectionMarker")]
    pub fn add_detection_marker(event_id: &str, lat: f64, lon: f64, label: &str, received_ns: f64);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "addBearingOnlyDetectionMarker")]
    pub fn add_bearing_only_detection_marker(
        event_id: &str,
        lat: f64,
        lon: f64,
        label: &str,
        source_lat: f64,
        source_lon: f64,
        has_source: bool,
        received_ns: f64,
    );

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "removeDetectionMarker")]
    pub fn remove_detection_marker(event_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setDetectionLayerData")]
    pub fn set_detection_layer_data(data: &JsValue);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "clearDetectionLayer")]
    pub fn clear_detection_layer();

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setTrackMarker")]
    pub fn set_track_marker(
        track_id: &str,
        lat: f64,
        lon: f64,
        label: &str,
        tqi: f64,
        status: &str,
        last_update_ns: f64,
    );

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setTrackVelocityVector")]
    pub fn set_track_velocity_vector(
        track_id: &str,
        lat: f64,
        lon: f64,
        vel_lat: f64,
        vel_lon: f64,
        status: &str,
    );

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "removeTrack")]
    pub fn remove_track(track_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "pulseTrackMarker")]
    pub fn pulse_track_marker(track_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setZone")]
    pub fn set_zone(zone_id: &str, latlngs: &JsValue, label: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "removeZone")]
    pub fn remove_zone(zone_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setGdopCircle")]
    pub fn set_gdop_circle(key: &str, lat: f64, lon: f64, radius_m: f64);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "removeGdopCircle")]
    pub fn remove_gdop_circle(key: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setEffectorMarker")]
    pub fn set_effector_marker(
        effector_id: &str,
        lat: f64,
        lon: f64,
        bearing_deg: f64,
        state: &str,
    );

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "removeEffectorMarker")]
    pub fn remove_effector_marker(effector_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "panTo")]
    pub fn pan_to(lat: f64, lon: f64);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "flyTo")]
    pub fn fly_to(lat: f64, lon: f64, zoom: Option<f64>);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "highlightCopItem")]
    pub fn highlight_cop_item(kind: &str, id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "clearCopHighlight")]
    pub fn clear_cop_highlight();

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setCopUncertainty")]
    pub fn set_cop_uncertainty(kind: &str, id: &str, lat: f64, lon: f64, covariance: &JsValue);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "clearAllCopUncertainty")]
    pub fn clear_all_cop_uncertainty();

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setBearingWedge")]
    pub fn set_bearing_wedge(
        event_id: &str,
        lat: f64,
        lon: f64,
        bearing_deg: f64,
        half_angle_deg: f64,
        range_m: f64,
    );

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "removeBearingWedge")]
    pub fn remove_bearing_wedge(event_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "clearBearingWedges")]
    pub fn clear_bearing_wedges();

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setCopSelectionCallback")]
    pub fn set_cop_selection_callback(callback: &js_sys::Function);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setCopSelectionCallback")]
    pub fn set_cop_selection_callback_value(callback: &JsValue);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setContextMenuCallback")]
    pub fn set_context_menu_callback(callback: &js_sys::Function);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setContextMenuCallback")]
    pub fn set_context_menu_callback_value(callback: &JsValue);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setHeatmapPoints")]
    pub fn set_heatmap_points(points: &JsValue, max_intensity: f64);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "clearHeatmap")]
    pub fn clear_heatmap();

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "ensureLayer")]
    pub fn ensure_layer(layer_id: &str, spec: &JsValue);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setLayerData")]
    pub fn set_layer_data(layer_id: &str, data: &JsValue);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "removeLayer")]
    pub fn remove_layer(layer_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setImageOverlay")]
    pub fn set_image_overlay(layer_id: &str, url: &str, corners: &JsValue, opacity: f64);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "setGeoJsonOverlay")]
    pub fn set_geojson_overlay(layer_id: &str, data: &JsValue, opacity: f64);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "removeMapOverlay")]
    pub fn remove_map_overlay(layer_id: &str);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "fitBoundsLatLons")]
    pub fn fit_bounds_latlons(points: &JsValue);

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "invalidateMapSize")]
    pub fn invalidate_map_size();

    #[wasm_bindgen(js_namespace = ["globalThis", "mapInterop"], js_name = "resize")]
    pub fn resize();
}
