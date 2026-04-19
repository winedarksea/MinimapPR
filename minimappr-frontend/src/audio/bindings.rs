use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["globalThis", "audioInterop"], js_name = "renderWaveform", catch)]
    pub async fn render_waveform(canvas_id: &str, url: &str) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_namespace = ["globalThis", "audioInterop"], js_name = "renderSpectrogram", catch)]
    pub async fn render_spectrogram(
        canvas_id: &str,
        url: &str,
        fft_size: u32,
    ) -> Result<JsValue, JsValue>;
}
