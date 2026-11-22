//! wasm-bindgen imports of the Field PWA's JS interop files
//! (`field/field-capture-interop.js`, `field/field-birdnet-interop.js`) and
//! small browser helpers (service worker, diagnostics object).

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = mmpFieldCapture, js_name = start, catch)]
    fn capture_start(options: JsValue, on_chunk: &Closure<dyn FnMut(f64, js_sys::Float32Array)>) -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen(js_namespace = mmpFieldCapture, js_name = stop, catch)]
    fn capture_stop() -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen(js_namespace = mmpFieldBirdnet, js_name = load, catch)]
    fn birdnet_load(options: JsValue) -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen(js_namespace = mmpFieldBirdnet, js_name = classify, catch)]
    fn birdnet_classify(samples: js_sys::Float32Array, top_count: u32) -> Result<js_sys::Promise, JsValue>;
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureStartOptions {
    pub requested_sample_rate_hz: u32,
    pub chunk_sample_count: u32,
    pub worklet_url: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NegotiatedCaptureSettings {
    pub context_sample_rate_hz: f64,
    pub track_sample_rate_hz: Option<f64>,
    pub device_label: String,
    pub echo_cancellation: bool,
    pub noise_suppression: bool,
    pub auto_gain_control: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BirdnetLoadOptions {
    pub model_url: String,
    pub labels_url: String,
    pub ort_wasm_base_url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BirdnetLoadReport {
    pub model_bytes: f64,
    pub model_from_cache: bool,
    /// False when Cache Storage refused the model (quota): online-only.
    pub model_stored_for_offline: bool,
    pub storage_persisted: bool,
    pub label_count: u32,
    pub fetch_milliseconds: f64,
    pub session_create_milliseconds: f64,
    pub intra_op_thread_count: u32,
    pub cross_origin_isolated: bool,
    pub ort_version: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BirdnetClassScore {
    pub class_index: u32,
    pub score: f32,
    pub common_name: String,
    pub scientific_name: String,
    pub taxon_group: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BirdnetClassification {
    pub inference_milliseconds: f64,
    pub top: Vec<BirdnetClassScore>,
}

pub fn describe_error(value: &JsValue) -> String {
    value
        .as_string()
        .or_else(|| js_sys::Reflect::get(value, &JsValue::from_str("message")).ok().and_then(|m| m.as_string()))
        .unwrap_or_else(|| format!("{value:?}"))
}

/// JSON-compatible: maps become plain objects (not JS `Map`), so
/// `serde_json::Value` diagnostics survive `JSON.stringify`.
fn to_js<T: Serialize>(value: &T) -> Result<JsValue, String> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|error| error.to_string())
}

fn from_js<T: for<'de> Deserialize<'de>>(value: JsValue) -> Result<T, String> {
    serde_wasm_bindgen::from_value(value).map_err(|error| error.to_string())
}

async fn await_promise(promise: Result<js_sys::Promise, JsValue>) -> Result<JsValue, String> {
    JsFuture::from(promise.map_err(|error| describe_error(&error))?)
        .await
        .map_err(|error| describe_error(&error))
}

/// Starts mic capture; `on_chunk(first_frame, samples)` runs per worklet chunk.
/// The closure must stay alive for as long as capture runs.
pub async fn start_capture(
    options: &CaptureStartOptions,
    on_chunk: &Closure<dyn FnMut(f64, js_sys::Float32Array)>,
) -> Result<NegotiatedCaptureSettings, String> {
    let settings = await_promise(capture_start(to_js(options)?, on_chunk)).await?;
    from_js(settings)
}

pub async fn stop_capture() -> Result<(), String> {
    await_promise(capture_stop()).await.map(|_| ())
}

pub async fn load_birdnet(options: &BirdnetLoadOptions) -> Result<BirdnetLoadReport, String> {
    from_js(await_promise(birdnet_load(to_js(options)?)).await?)
}

pub async fn classify_with_birdnet(samples_32_khz: &[f32], top_count: u32) -> Result<BirdnetClassification, String> {
    let array = js_sys::Float32Array::from(samples_32_khz);
    from_js(await_promise(birdnet_classify(array, top_count)).await?)
}

/// Registers the offline-shell service worker (no-op where unsupported).
pub fn register_service_worker(script_url: &str) {
    let Some(window) = web_sys::window() else { return };
    let navigator = window.navigator();
    // `navigator.serviceWorker` is undefined in insecure contexts (plain-http
    // LAN address); calling `register` on it would throw and abort the mount.
    let has_service_worker = js_sys::Reflect::has(&navigator, &JsValue::from_str("serviceWorker")).unwrap_or(false);
    if !has_service_worker {
        return;
    }
    let _ = navigator.service_worker().register(script_url);
}

/// `window.mmpFieldDiagnostics[key] = value`: measurements for the browser
/// checks (Playwright reads them) without scraping the DOM.
pub fn publish_diagnostic<T: Serialize>(key: &str, value: &T) {
    let Some(window) = web_sys::window() else { return };
    let diagnostics_key = JsValue::from_str("mmpFieldDiagnostics");
    let diagnostics = js_sys::Reflect::get(&window, &diagnostics_key)
        .ok()
        .filter(|existing| existing.is_object())
        .unwrap_or_else(|| {
            let created: JsValue = js_sys::Object::new().into();
            let _ = js_sys::Reflect::set(&window, &diagnostics_key, &created);
            created
        });
    if let Ok(serialized) = to_js(value) {
        let _ = js_sys::Reflect::set(&diagnostics, &JsValue::from_str(key), &serialized);
    }
}
