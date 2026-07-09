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
struct LabelSuggestion {
    #[serde(default)]
    name: String,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    count: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct LabelSuggestionResponse {
    #[serde(default)]
    labels: Vec<LabelSuggestion>,
}

fn suggested_category_for_label(suggestions: &[LabelSuggestion], label: &str) -> Option<String> {
    suggestions
        .iter()
        .find(|suggestion| suggestion.name == label)
        .and_then(|suggestion| suggestion.category.clone())
}

fn training_kind_for_review(review_state: &str, selected_kind: &str) -> Option<String> {
    (review_state == "confirmed" && selected_kind != "none").then(|| selected_kind.to_string())
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
    label_category: Option<String>,
    #[serde(default)]
    label_confidence: Option<f64>,
    #[serde(default)]
    label_hits: Vec<LabelHit>,
    #[serde(default)]
    classification_label_hits: Option<Vec<LabelHit>>,
    #[serde(default)]
    feature_summary: Option<FeatureSummary>,
    #[serde(default)]
    review_state: Option<String>,
    #[serde(default)]
    review_label: Option<String>,
    #[serde(default)]
    review_label_category: Option<String>,
    #[serde(default)]
    review_notes: Option<String>,
    #[serde(default)]
    review_updated_ns: Option<i64>,
    #[serde(default)]
    promote_to_training: bool,
    #[serde(default)]
    training_example_kind: Option<String>,
}

fn fmt_ts_ns(ns: i64) -> String {
    let ms = (ns as f64) / 1e6;
    let d = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(ms));
    d.to_iso_string().as_string().unwrap_or_default()
}

fn form_value(id: &str) -> String {
    web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id(id))
        .and_then(|element| {
            element
                .clone()
                .dyn_into::<web_sys::HtmlInputElement>()
                .ok()
                .map(|input| input.value())
                .or_else(|| {
                    element
                        .dyn_into::<web_sys::HtmlTextAreaElement>()
                        .ok()
                        .map(|input| input.value())
                })
        })
        .unwrap_or_default()
}

fn review_state_class(state: &str) -> &'static str {
    match state {
        "confirmed" => "tone-badge ok",
        "rejected" => "tone-badge danger",
        "unreviewed" => "tone-badge neutral",
        _ => "tone-badge neutral",
    }
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
                    view! {
                        <DetectionMetaPanel detection=detection />
                        <DetectionReviewPanel
                            detection_id=detection_id.clone()
                            detection=detection
                        />
                    }.into_any()
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
                            <dt>"Review"</dt><dd>
                                <span class=review_state_class(d.review_state.as_deref().unwrap_or("unreviewed"))>
                                    {d.review_state.clone().unwrap_or_else(|| "unreviewed".to_string())}
                                </span>
                            </dd>
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

#[component]
fn DetectionReviewPanel(
    detection_id: String,
    detection: RwSignal<Option<Detection>>,
) -> impl IntoView {
    let label_input_id = format!("review-label-{detection_id}");
    let category_input_id = format!("review-category-{detection_id}");
    let notes_input_id = format!("review-notes-{detection_id}");
    let datalist_id = format!("review-label-options-{detection_id}");

    let label_value = RwSignal::new(String::new());
    let category_value = RwSignal::new(String::new());
    let notes_value = RwSignal::new(String::new());
    let training_example_kind = RwSignal::new("none".to_string());
    let suggestions = RwSignal::new(Vec::<LabelSuggestion>::new());
    let pending = RwSignal::new(false);
    let review_message = RwSignal::new(None::<String>);
    let review_error = RwSignal::new(None::<String>);
    let label_input_id_for_submit = label_input_id.clone();
    let category_input_id_for_submit = category_input_id.clone();
    let notes_input_id_for_submit = notes_input_id.clone();

    Effect::new(move |_| {
        if let Some(d) = detection.get() {
            label_value.set(
                d.review_label
                    .clone()
                    .or(d.label.clone())
                    .unwrap_or_default(),
            );
            category_value.set(
                d.review_label_category
                    .clone()
                    .or(d.label_category.clone())
                    .unwrap_or_else(|| "unknown".to_string()),
            );
            notes_value.set(d.review_notes.clone().unwrap_or_default());
            training_example_kind.set(if d.promote_to_training {
                d.training_example_kind
                    .unwrap_or_else(|| "positive".to_string())
            } else {
                "none".to_string()
            });
        }
    });

    Effect::new(move |_| {
        spawn_local(async move {
            match Request::get("/api/v1/labels").send().await {
                Ok(resp) if resp.ok() => match resp.json::<LabelSuggestionResponse>().await {
                    Ok(payload) => suggestions.set(payload.labels),
                    Err(error) => log::warn!("label suggestions parse failed: {error}"),
                },
                Ok(resp) => log::warn!("label suggestions HTTP {}", resp.status()),
                Err(error) => log::warn!("label suggestions fetch failed: {error}"),
            }
        });
    });

    let submit_review = move |review_state: &'static str, include_label: bool| {
        if pending.get_untracked() {
            return;
        }

        let detection_id = detection_id.clone();
        let label = form_value(&label_input_id_for_submit).trim().to_string();
        let category = form_value(&category_input_id_for_submit).trim().to_string();
        let notes = form_value(&notes_input_id_for_submit).trim().to_string();
        let selected_training_kind = training_example_kind.get_untracked();
        let selected_training_kind =
            training_kind_for_review(review_state, &selected_training_kind);
        let promote = selected_training_kind.is_some();
        let detection = detection;

        let mut payload = serde_json::json!({
            "review_state": review_state,
            "review_notes": if notes.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(notes) },
            "promote_to_training": promote,
        });
        if let Some(training_kind) = selected_training_kind {
            payload["training_example_kind"] = serde_json::Value::String(training_kind);
        }
        if include_label && !label.is_empty() {
            payload["review_label"] = serde_json::Value::String(label);
            payload["review_label_category"] = serde_json::Value::String(if category.is_empty() {
                "unknown".to_string()
            } else {
                category
            });
        } else if review_state == "rejected" {
            payload["review_label"] = serde_json::Value::Null;
            payload["review_label_category"] = serde_json::Value::Null;
        }

        pending.set(true);
        review_error.set(None);
        review_message.set(None);

        spawn_local(async move {
            match crate::api::patch_detection_review(&detection_id, payload).await {
                Ok(value) => match serde_json::from_value::<Detection>(value) {
                    Ok(updated) => {
                        detection.set(Some(updated));
                        review_message.set(Some(if promote {
                            "Review saved; training example stored".to_string()
                        } else {
                            "Review saved".to_string()
                        }));
                    }
                    Err(error) => review_error.set(Some(format!("parse: {error}"))),
                },
                Err(error) => review_error.set(Some(error)),
            }
            pending.set(false);
        });
    };

    let save_unconfirmed_review = {
        let submit_review = submit_review.clone();
        move |_| submit_review("unreviewed", true)
    };
    let confirm_review = {
        let submit_review = submit_review.clone();
        move |_| submit_review("confirmed", true)
    };
    let reject_review = move |_| submit_review("rejected", false);

    view! {
        <aside class="audio-meta detection-review-card">
            <div class="detection-review-header">
                <h3>"Review"</h3>
                {move || {
                    detection.get().map(|d| {
                        let state = d.review_state.unwrap_or_else(|| "unreviewed".to_string());
                        let state_class = review_state_class(&state);
                        view! { <span class=state_class>{state}</span> }
                    })
                }}
            </div>
            <div class="detection-review-grid">
                <label>
                    <span>"Label"</span>
                    <input
                        id=label_input_id.clone()
                        list=datalist_id.clone()
                        type="text"
                        prop:value=move || label_value.get()
                        on:input=move |event| {
                            let value = event_target_value(&event);
                            label_value.set(value.clone());
                            if let Some(category) = suggested_category_for_label(
                                &suggestions.get_untracked(),
                                &value,
                            ) {
                                category_value.set(category);
                            }
                        }
                    />
                </label>
                <label>
                    <span>"Category"</span>
                    <input
                        id=category_input_id.clone()
                        type="text"
                        prop:value=move || category_value.get()
                        on:input=move |event| category_value.set(event_target_value(&event))
                    />
                </label>
                <datalist id=datalist_id>
                    {move || suggestions.get().into_iter().map(|suggestion| {
                        let label = suggestion.name;
                        let category = suggestion.category.unwrap_or_else(|| "unknown".to_string());
                        let display = match suggestion.count {
                            Some(count) => format!("{category} · {count}"),
                            None => category,
                        };
                        view! { <option value=label label=display></option> }
                    }).collect_view()}
                </datalist>
                <label class="detection-review-notes">
                    <span>"Notes"</span>
                    <textarea
                        id=notes_input_id.clone()
                        prop:value=move || notes_value.get()
                        on:input=move |event| notes_value.set(event_target_value(&event))
                    ></textarea>
                </label>
                <label class="review-training-toggle">
                    <span>"Training dataset"</span>
                    <select
                        prop:value=move || training_example_kind.get()
                        on:change=move |event| training_example_kind.set(event_target_value(&event))
                    >
                        <option value="none">"Do not include"</option>
                        <option value="positive">"Positive example"</option>
                        <option value="negative">"Negative / background example"</option>
                    </select>
                </label>
            </div>
            <div class="detection-review-actions">
                <button
                    class="btn-sm"
                    disabled=move || pending.get()
                    on:click=save_unconfirmed_review
                >
                    "Save unconfirmed"
                </button>
                <button
                    class="btn-sm btn-primary"
                    disabled=move || pending.get()
                    on:click=confirm_review
                >
                    "Confirm"
                </button>
                <button
                    class="btn-sm btn-danger"
                    disabled=move || pending.get()
                    on:click=reject_review
                >
                    "Reject"
                </button>
            </div>
            {move || review_message.get().map(|message| view! {
                <span class="review-status-ok">{message}</span>
            })}
            {move || review_error.get().map(|message| view! {
                <span class="daily-error">{message}</span>
            })}
        </aside>
    }
}

#[cfg(test)]
mod tests {
    use super::{suggested_category_for_label, training_kind_for_review, LabelSuggestion};

    #[test]
    fn suggestion_selection_autofills_category_without_restricting_free_text() {
        let suggestions = vec![LabelSuggestion {
            name: "ambient".to_string(),
            category: Some("background".to_string()),
            count: None,
        }];

        assert_eq!(
            suggested_category_for_label(&suggestions, "ambient"),
            Some("background".to_string())
        );
        assert_eq!(
            suggested_category_for_label(&suggestions, "custom label"),
            None
        );
    }

    #[test]
    fn only_confirmed_reviews_can_create_training_examples() {
        assert_eq!(
            training_kind_for_review("confirmed", "positive"),
            Some("positive".to_string())
        );
        assert_eq!(
            training_kind_for_review("confirmed", "negative"),
            Some("negative".to_string())
        );
        assert_eq!(training_kind_for_review("unreviewed", "positive"), None);
        assert_eq!(training_kind_for_review("confirmed", "none"), None);
    }
}
