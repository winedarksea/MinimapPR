use crate::audio::bindings as aud;
use crate::audio::detection_actions::{
    audio_analysis_href, detection_audio_download_url, detection_audio_url,
};
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
struct AudioQuality {
    #[serde(default)]
    source_window_type: Option<String>,
    #[serde(default)]
    coverage_ratio: Option<f64>,
    #[serde(default)]
    missing_ratio: Option<f64>,
    #[serde(default)]
    max_gap_seconds: Option<f64>,
    #[serde(default)]
    warning: Option<bool>,
    #[serde(default)]
    degraded: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct FeatureSummary {
    #[serde(default)]
    audio_quality: Option<AudioQuality>,
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
    feature_summary: Option<FeatureSummary>,
}

fn fmt_ts_ns(ns: i64) -> String {
    let ms = (ns as f64) / 1e6;
    let d = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(ms));
    d.to_iso_string().as_string().unwrap_or_default()
}

// ── Subcomponents ────────────────────────────────────────────────────────────

/// Header bar: detection title, info/error spans, play/pause, download, expand.
#[component]
fn AudioAnalysisHeader(
    analysis_title: String,
    analysis_id: String,
    info: RwSignal<Option<String>>,
    error: RwSignal<Option<String>>,
    is_playing: RwSignal<bool>,
    player_id: String,
    download_url: String,
    expand_href: String,
    show_expand_link: bool,
) -> impl IntoView {
    let player_id_play = player_id.clone();
    view! {
        <div class="audio-header">
            <h2 style="margin:0">{analysis_title} " " <code>{analysis_id.clone()}</code></h2>
            <span class="muted">{move || info.get().unwrap_or_default()}</span>
            <span class="daily-error">{move || error.get().unwrap_or_default()}</span>
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
            <a class="btn-sm" href=download_url download=format!("{}.wav", analysis_id)>
                "Download WAV"
            </a>
            {if show_expand_link {
                view! {
                    <A href=expand_href>
                        <span class="btn-sm">"Expand"</span>
                    </A>
                }.into_any()
            } else {
                ().into_any()
            }}
        </div>
    }
}

/// Canvas area: waveform, spectrogram, and the native audio player element.
#[component]
fn AudioCanvasPanel(
    waveform_id: String,
    spectrogram_id: String,
    player_id: String,
    audio_url: String,
    is_playing: RwSignal<bool>,
) -> impl IntoView {
    view! {
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
    }
}

// ── Main view ────────────────────────────────────────────────────────────────

#[component]
pub fn DetectionAudioAnalysisView(
    detection_id: String,
    #[prop(default = false)] show_expand_link: bool,
    #[prop(default = "audio-page".to_string())] container_class: String,
    #[prop(default = "audio".to_string())] instance_prefix: String,
    #[prop(optional)] audio_url_override: Option<String>,
    #[prop(optional)] download_url_override: Option<String>,
    #[prop(default = true)] load_detection_metadata: bool,
    #[prop(default = "Detection".to_string())] analysis_title: String,
) -> impl IntoView {
    let detection: RwSignal<Option<Detection>> = RwSignal::new(None);
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let info: RwSignal<Option<String>> = RwSignal::new(None);
    let metadata_loaded: RwSignal<bool> = RwSignal::new(false);

    // Monotonically-increasing generation counter used as a stale-render guard.
    // Incremented at the start of each async cycle.  Async tasks capture the
    // generation value at launch and discard their results if a newer cycle has
    // started by the time they resume.  This prevents a delayed response from a
    // superseded detection (or a reused canvas ID from a previous drawer
    // instance) from overwriting the current view.
    let render_gen: RwSignal<u32> = RwSignal::new(0);

    let waveform_id = format!("{instance_prefix}-waveform");
    let spectrogram_id = format!("{instance_prefix}-spectrogram");
    let player_id = format!("{instance_prefix}-player");
    let audio_url = audio_url_override.unwrap_or_else(|| detection_audio_url(&detection_id));
    let download_url =
        download_url_override.unwrap_or_else(|| detection_audio_download_url(&detection_id));
    let metadata_url = format!("/api/v1/detections/{detection_id}");
    let is_playing: RwSignal<bool> = RwSignal::new(false);
    let expand_href = audio_analysis_href(&detection_id);

    // Fetch metadata panel details when this view is backed by a detection.
    Effect::new({
        let metadata_url = metadata_url.clone();
        move |_| {
            let gen = render_gen.get_untracked().wrapping_add(1);
            render_gen.set(gen);
            detection.set(None);
            metadata_loaded.set(false);
            error.set(None);
            info.set(None);
            if !load_detection_metadata {
                metadata_loaded.set(true);
                return;
            }
            let det_url = metadata_url.clone();
            spawn_local(async move {
                let result = Request::get(&det_url).send().await;
                // Abort if a newer cycle started while we were awaiting.
                if render_gen.get_untracked() != gen {
                    return;
                }
                match result {
                    Ok(r) if r.ok() => match r.json::<Detection>().await {
                        Ok(d) => {
                            if render_gen.get_untracked() == gen {
                                detection.set(Some(d));
                                metadata_loaded.set(true);
                            }
                        }
                        Err(e) => {
                            if render_gen.get_untracked() == gen {
                                error.set(Some(format!("parse: {e}")));
                                metadata_loaded.set(true);
                            }
                        }
                    },
                    Ok(r) => {
                        if render_gen.get_untracked() == gen {
                            error.set(Some(format!("HTTP {}", r.status())));
                            metadata_loaded.set(true);
                        }
                    }
                    Err(e) => {
                        if render_gen.get_untracked() == gen {
                            error.set(Some(e.to_string()));
                            metadata_loaded.set(true);
                        }
                    }
                }
            });
        }
    });

    // Render waveform and spectrogram from a single decode pass once metadata
    // is loaded.  The JS-side renderWaveformAndSpectrogram also stamps a URL
    // token on each canvas element before the async fetch, providing an
    // additional layer of stale detection at the DOM level.
    Effect::new({
        let audio_url = audio_url.clone();
        let waveform_id = waveform_id.clone();
        let spectrogram_id = spectrogram_id.clone();
        move |_| {
            if !metadata_loaded.get() {
                return;
            }
            let gen = render_gen.get_untracked();
            let waveform_id = waveform_id.clone();
            let spectrogram_id = spectrogram_id.clone();
            let audio_url = audio_url.clone();
            spawn_local(async move {
                match aud::render_waveform_and_spectrogram(
                    &waveform_id,
                    &spectrogram_id,
                    &audio_url,
                    512,
                )
                .await
                {
                    Ok(v) => {
                        // Discard if a newer render cycle has taken over.
                        if render_gen.get_untracked() != gen {
                            return;
                        }
                        let dur = js_sys::Reflect::get(
                            &v,
                            &wasm_bindgen::JsValue::from_str("duration_s"),
                        )
                        .ok()
                        .and_then(|x| x.as_f64());
                        let sr = js_sys::Reflect::get(
                            &v,
                            &wasm_bindgen::JsValue::from_str("sample_rate"),
                        )
                        .ok()
                        .and_then(|x| x.as_f64());
                        if let (Some(dur), Some(sr)) = (dur, sr) {
                            info.set(Some(format!("{dur:.2}s · {} Hz", sr as u32)));
                        }
                    }
                    Err(e) => {
                        if render_gen.get_untracked() == gen {
                            error.set(Some(format!("render: {e:?}")));
                        }
                    }
                }
            });
        }
    });

    view! {
        <div class=container_class>
            <div class="audio-layout">
                <AudioAnalysisHeader
                    analysis_title=analysis_title.clone()
                    analysis_id=detection_id.clone()
                    info=info
                    error=error
                    is_playing=is_playing
                    player_id=player_id.clone()
                    download_url=download_url
                    expand_href=expand_href
                    show_expand_link=show_expand_link
                />
                <AudioCanvasPanel
                    waveform_id=waveform_id
                    spectrogram_id=spectrogram_id
                    player_id=player_id
                    audio_url=audio_url
                    is_playing=is_playing
                />
                {if load_detection_metadata {
                    view! { <DetectionMetaPanel detection=detection /> }.into_any()
                } else {
                    ().into_any()
                }}
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
                    let audio_quality = d.feature_summary.as_ref().and_then(|f| f.audio_quality.clone());
                    view! {
                        <h3>"Metadata"</h3>
                        <dl class="audio-meta-dl">
                            <dt>"Event"</dt><dd><code>{d.event_id.clone()}</code></dd>
                            <dt>"Timestamp"</dt><dd>{ts}</dd>
                            <dt>"Node"</dt><dd>{node}</dd>
                            <dt>"Primary label"</dt><dd>{primary} " " <span class="muted">{primary_conf}</span></dd>
                        </dl>
                        {audio_quality.map(|q| {
                            let missing = q.missing_ratio.unwrap_or(0.0);
                            let gap_ms = q.max_gap_seconds.unwrap_or(0.0) * 1000.0;
                            let status = if q.degraded.unwrap_or(false) {
                                "Degraded"
                            } else if q.warning.unwrap_or(false) || missing > 0.0 || gap_ms > 0.0 {
                                "Warning"
                            } else {
                                "Clean"
                            };
                            let source = q.source_window_type.unwrap_or_else(|| "-".into());
                            view! {
                                <h3>"Audio quality"</h3>
                                <dl class="audio-meta-dl">
                                    <dt>"Status"</dt><dd>{status}</dd>
                                    <dt>"Missing"</dt><dd>{format!("{:.1}%", missing * 100.0)}</dd>
                                    <dt>"Max gap"</dt><dd>{format!("{:.0} ms", gap_ms)}</dd>
                                    <dt>"Window"</dt><dd>{source}</dd>
                                </dl>
                            }
                        })}
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
