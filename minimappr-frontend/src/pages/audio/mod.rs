use crate::audio::bindings as aud;
use gloo_net::http::Request;
use leptos::prelude::*;
use leptos::task::spawn_local;
use leptos_router::hooks::use_params_map;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct LabelHit {
    #[serde(default)]
    label: String,
    #[serde(default)]
    confidence: f64,
    #[serde(default)]
    source: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct Detection {
    #[serde(default)]
    event_id: String,
    #[serde(default)]
    detection_id: Option<String>,
    #[serde(default)]
    timestamp_ns: i64,
    #[serde(default)]
    node_id: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    label_confidence: Option<f64>,
    #[serde(default)]
    label_hits: Vec<LabelHit>,
    #[serde(default)]
    classification_label_hits: Option<Vec<LabelHit>>,
    #[serde(default)]
    snippet_url: Option<String>,
}

fn fmt_ts_ns(ns: i64) -> String {
    let ms = (ns as f64) / 1e6;
    let d = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(ms));
    d.to_iso_string().as_string().unwrap_or_default()
}

#[component]
pub fn AudioAnalysisPage() -> impl IntoView {
    let params = use_params_map();
    let id_sig: Signal<Option<String>> = Signal::derive(move || params.read().get("id"));

    let detection: RwSignal<Option<Detection>> = RwSignal::new(None);
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let info: RwSignal<Option<String>> = RwSignal::new(None);

    Effect::new(move |_| {
        let id = id_sig.get();
        detection.set(None);
        error.set(None);
        info.set(None);
        let Some(id) = id else { return; };
        let det_url = format!("/api/v1/detections/{id}");
        spawn_local(async move {
            match Request::get(&det_url).send().await {
                Ok(r) if r.ok() => match r.json::<Detection>().await {
                    Ok(d) => detection.set(Some(d)),
                    Err(e) => error.set(Some(format!("parse: {e}"))),
                },
                Ok(r) => error.set(Some(format!("HTTP {}", r.status()))),
                Err(e) => error.set(Some(e.to_string())),
            }
        });
    });

    // When detection is loaded, render waveform + spectrogram from the snippet URL.
    Effect::new(move |_| {
        let Some(id) = id_sig.get() else { return; };
        if detection.get().is_none() { return; }
        let audio_url = format!("/api/v1/detections/{id}/audio");
        spawn_local(async move {
            match aud::render_waveform("audio-waveform", &audio_url).await {
                Ok(v) => {
                    let dur = js_sys::Reflect::get(&v, &wasm_bindgen::JsValue::from_str("duration_s"))
                        .ok().and_then(|x| x.as_f64());
                    let sr = js_sys::Reflect::get(&v, &wasm_bindgen::JsValue::from_str("sample_rate"))
                        .ok().and_then(|x| x.as_f64());
                    if let (Some(dur), Some(sr)) = (dur, sr) {
                        info.set(Some(format!("{dur:.2}s · {} Hz", sr as u32)));
                    }
                }
                Err(e) => error.set(Some(format!("waveform: {e:?}"))),
            }
            let _ = aud::render_spectrogram("audio-spectrogram", &audio_url, 512).await;
        });
    });

    view! {
        <div class="audio-page">
            {move || match id_sig.get() {
                None => view! {
                    <div class="page-stub">
                        <h2>"Audio analysis"</h2>
                        <p class="muted">"Open a detection from the COP or Labels view to inspect its audio. Live stream mode will land in a later pass."</p>
                    </div>
                }.into_any(),
                Some(id) => {
                    let audio_url = format!("/api/v1/detections/{id}/audio");
                    view! {
                        <div class="audio-layout">
                            <div class="audio-header">
                                <h2 style="margin:0">"Detection " <code>{id.clone()}</code></h2>
                                <span class="muted">{move || info.get().unwrap_or_default()}</span>
                                <span class="daily-error">{move || error.get().unwrap_or_default()}</span>
                            </div>

                            <div class="audio-canvases">
                                <div class="audio-canvas-row">
                                    <label class="muted">"Waveform"</label>
                                    <canvas id="audio-waveform"></canvas>
                                </div>
                                <div class="audio-canvas-row">
                                    <label class="muted">"Spectrogram"</label>
                                    <canvas id="audio-spectrogram"></canvas>
                                </div>
                                <audio
                                    id="audio-page-player"
                                    controls=true
                                    src=audio_url
                                    style="width:100%;margin-top:8px"
                                />
                            </div>

                            <DetectionMetaPanel detection=detection />
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}

#[component]
fn DetectionMetaPanel(detection: RwSignal<Option<Detection>>) -> impl IntoView {
    view! {
        <aside class="audio-meta">
            {move || match detection.get() {
                None => view! { <div class="muted">"Loading detection…"</div> }.into_any(),
                Some(d) => {
                    let ts = fmt_ts_ns(d.timestamp_ns);
                    let node = d.node_id.clone().unwrap_or_else(|| "—".into());
                    let primary = d.label.clone().unwrap_or_else(|| "—".into());
                    let primary_conf = d.label_confidence.map(|c| format!("{:.1}%", c * 100.0)).unwrap_or_else(|| "".into());
                    let hits: Vec<LabelHit> = if !d.label_hits.is_empty() {
                        d.label_hits.clone()
                    } else {
                        d.classification_label_hits.clone().unwrap_or_default()
                    };
                    view! {
                        <h3>"Metadata"</h3>
                        <dl class="audio-meta-dl">
                            <dt>"Event"</dt><dd><code>{d.event_id.clone()}</code></dd>
                            <dt>"Timestamp"</dt><dd>{ts}</dd>
                            <dt>"Node"</dt><dd>{node}</dd>
                            <dt>"Primary label"</dt><dd>{primary} " " <span class="muted">{primary_conf}</span></dd>
                        </dl>
                        <h3>"Classifier hits"</h3>
                        {if hits.is_empty() {
                            view! { <p class="muted">"No additional hits."</p> }.into_any()
                        } else {
                            view! {
                                <ul class="audio-hits">
                                    {hits.into_iter().map(|h| {
                                        let bar_pct = (h.confidence.clamp(0.0, 1.0) * 100.0) as u32;
                                        view! {
                                            <li>
                                                <div class="audio-hit-row">
                                                    <span class="audio-hit-label">{h.label.clone()}</span>
                                                    <span class="audio-hit-conf">{format!("{:.1}%", h.confidence * 100.0)}</span>
                                                </div>
                                                <div class="tqi-bar"><div class="tqi-bar-fill" style=format!("width:{bar_pct}%")></div></div>
                                                {h.source.clone().map(|s| view! { <span class="muted" style="font-size:11px">{s}</span> })}
                                            </li>
                                        }
                                    }).collect_view()}
                                </ul>
                            }.into_any()
                        }}
                    }.into_any()
                }
            }}
        </aside>
    }
}
