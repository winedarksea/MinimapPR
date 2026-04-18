use crate::audio::bindings as aud;
use crate::audio::detection_actions::{audio_analysis_href, detection_audio_download_url, detection_audio_url};
use gloo_net::http::Request;
use leptos::prelude::*;
use leptos::task::spawn_local;
use leptos_router::components::A;
use serde::Deserialize;
use wasm_bindgen::JsCast;

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
}

fn fmt_ts_ns(ns: i64) -> String {
    let ms = (ns as f64) / 1e6;
    let d = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(ms));
    d.to_iso_string().as_string().unwrap_or_default()
}

#[component]
pub fn DetectionAudioAnalysisView(
    detection_id: String,
    #[prop(default = false)] show_expand_link: bool,
    #[prop(default = "audio-page".to_string())] container_class: String,
    #[prop(default = "audio".to_string())] instance_prefix: String,
) -> impl IntoView {
    let detection: RwSignal<Option<Detection>> = RwSignal::new(None);
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let info: RwSignal<Option<String>> = RwSignal::new(None);

    let waveform_id = format!("{instance_prefix}-waveform");
    let spectrogram_id = format!("{instance_prefix}-spectrogram");
    let player_id = format!("{instance_prefix}-player");
    let audio_url = detection_audio_url(&detection_id);
    let download_url = detection_audio_download_url(&detection_id);
    let is_playing: RwSignal<bool> = RwSignal::new(false);
    let expand_href = audio_analysis_href(&detection_id);

    // Fetch metadata panel details for the selected detection.
    Effect::new({
        let detection_id = detection_id.clone();
        move |_| {
            detection.set(None);
            error.set(None);
            info.set(None);
            let det_url = format!("/api/v1/detections/{detection_id}");
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
        }
    });

    // Render waveform/spectrogram once the metadata is loaded.
    Effect::new({
        let audio_url = audio_url.clone();
        let waveform_id = waveform_id.clone();
        let spectrogram_id = spectrogram_id.clone();
        move |_| {
            if detection.get().is_none() {
                return;
            }
            let waveform_id = waveform_id.clone();
            let spectrogram_id = spectrogram_id.clone();
            let audio_url = audio_url.clone();
            spawn_local(async move {
                match aud::render_waveform(&waveform_id, &audio_url).await {
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
                let _ = aud::render_spectrogram(&spectrogram_id, &audio_url, 512).await;
            });
        }
    });

    view! {
        <div class=container_class>
            <div class="audio-layout">
                <div class="audio-header">
                    <h2 style="margin:0">"Detection " <code>{detection_id.clone()}</code></h2>
                    <span class="muted">{move || info.get().unwrap_or_default()}</span>
                    <span class="daily-error">{move || error.get().unwrap_or_default()}</span>
                    {
                        let player_id_play = player_id.clone();
                        view! {
                            <button
                                class="play-btn"
                                title="Play / Pause audio"
                                on:click=move |_| {
                                    if let Some(el) = web_sys::window()
                                        .and_then(|w| w.document())
                                        .and_then(|d| d.get_element_by_id(&player_id_play))
                                    {
                                        if let Ok(audio) = el.dyn_into::<web_sys::HtmlAudioElement>() {
                                            if audio.paused() {
                                                let _ = audio.play();
                                                is_playing.set(true);
                                            } else {
                                                audio.pause().unwrap_or(());
                                                is_playing.set(false);
                                            }
                                        }
                                    }
                                }
                            >
                                {move || if is_playing.get() { "⏸" } else { "▶" }}
                            </button>
                        }
                    }
                    <a class="btn-sm" href=download_url download=format!("{}.wav", detection_id.clone())>
                        "Download WAV"
                    </a>
                    {if show_expand_link {
                        view! {
                            <A href=expand_href>
                                <span class="btn-sm">"Expand"</span>
                            </A>
                        }.into_any()
                    } else {
                        view! { <></> }.into_any()
                    }}
                </div>

                <div class="audio-canvases">
                    <div class="audio-canvas-row">
                        <label class="muted">"Waveform"</label>
                        <canvas id=waveform_id></canvas>
                    </div>
                    <div class="audio-canvas-row">
                        <label class="muted">"Spectrogram"</label>
                        <canvas id=spectrogram_id></canvas>
                    </div>
                    <audio
                        id=player_id
                        controls=true
                        src=audio_url
                        style="width:100%;margin-top:8px"
                        on:ended=move |_| is_playing.set(false)
                        on:pause=move |_| is_playing.set(false)
                        on:play=move |_| is_playing.set(true)
                    />
                </div>

                <DetectionMetaPanel detection=detection />
            </div>
        </div>
    }
}

#[component]
fn DetectionMetaPanel(detection: RwSignal<Option<Detection>>) -> impl IntoView {
    view! {
        <aside class="audio-meta">
            {move || match detection.get() {
                None => view! { <div class="muted">"Loading detection..."</div> }.into_any(),
                Some(d) => {
                    let ts = fmt_ts_ns(d.timestamp_ns);
                    let node = d.node_id.clone().unwrap_or_else(|| "-".into());
                    let primary = d.label.clone().unwrap_or_else(|| "-".into());
                    let primary_conf = d.label_confidence
                        .map(|c| format!("{:.1}%", c * 100.0))
                        .unwrap_or_default();
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
