use crate::audio::detection_analysis::DetectionAudioAnalysisView;
use gloo_net::http::Request;
use leptos::prelude::*;
use leptos::task::spawn_local;
use leptos_router::components::A;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct Transcript {
    id: String,
    #[serde(default)]
    node_id: Option<String>,
    #[serde(default)]
    sensor_id: Option<String>,
    start_ns: i64,
    end_ns: i64,
    text: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    trigger_confidence: Option<f64>,
    #[serde(default)]
    detection_id: Option<String>,
    created_ns: i64,
}

fn transcript_detail_url(transcript_id: &str) -> String {
    format!(
        "/api/v1/transcripts/{}",
        js_sys::encode_uri_component(transcript_id)
    )
}

fn transcript_audio_url(transcript_id: &str) -> String {
    format!("{}/audio", transcript_detail_url(transcript_id))
}

fn fmt_ts_ns(ns: i64) -> String {
    let ms = (ns as f64) / 1e6;
    js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(ms))
        .to_iso_string()
        .as_string()
        .unwrap_or_default()
}

#[component]
pub fn TranscriptAudioAnalysisView(transcript_id: String) -> impl IntoView {
    let transcript: RwSignal<Option<Transcript>> = RwSignal::new(None);
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let detail_url = transcript_detail_url(&transcript_id);

    Effect::new(move |_| {
        transcript.set(None);
        error.set(None);
        let detail_url = detail_url.clone();
        spawn_local(async move {
            match Request::get(&detail_url).send().await {
                Ok(response) if response.ok() => match response.json::<Transcript>().await {
                    Ok(record) => transcript.set(Some(record)),
                    Err(fetch_error) => {
                        error.set(Some(format!("Could not read transcript: {fetch_error}")))
                    }
                },
                Ok(response) if response.status() == 404 => {
                    error.set(Some("Transcript not found or expired.".to_string()));
                }
                Ok(response) => error.set(Some(format!(
                    "Could not load transcript (HTTP {}).",
                    response.status()
                ))),
                Err(fetch_error) => {
                    error.set(Some(format!("Could not load transcript: {fetch_error}")))
                }
            }
        });
    });

    let audio_url = transcript_audio_url(&transcript_id);
    view! {
        <div style="display:flex;flex-direction:column;gap:8px;height:100%">
            <div style="display:flex;align-items:center;gap:8px;flex-shrink:0">
                <A href="/cop"><span class="btn-sm">"← COP"</span></A>
                <A href="/audio/analysis"><span class="btn-sm">"← Audio analysis"</span></A>
            </div>
            {move || match (transcript.get(), error.get()) {
                (_, Some(message)) => view! { <div class="daily-error">{message}</div> }.into_any(),
                (None, None) => view! { <div class="muted">"Loading transcript..."</div> }.into_any(),
                (Some(record), None) => {
                    let detection_href = record.detection_id.as_ref()
                        .map(|id| format!("/audio/d/{}", js_sys::encode_uri_component(id)));
                    view! {
                        <section class="transcript-review-card">
                            <h2>"Transcript"</h2>
                            <dl class="audio-meta-dl transcript-review-meta">
                                <dt>"ID"</dt><dd><code>{record.id.clone()}</code></dd>
                                <dt>"Node"</dt><dd>{record.node_id.clone().unwrap_or_else(|| "—".to_string())}</dd>
                                <dt>"Sensor"</dt><dd>{record.sensor_id.clone().unwrap_or_else(|| "—".to_string())}</dd>
                                <dt>"Captured"</dt><dd>{format!("{} – {}", fmt_ts_ns(record.start_ns), fmt_ts_ns(record.end_ns))}</dd>
                                <dt>"Created"</dt><dd>{fmt_ts_ns(record.created_ns)}</dd>
                                <dt>"Model"</dt><dd>{record.model.clone().unwrap_or_else(|| "—".to_string())}</dd>
                                <dt>"Trigger confidence"</dt><dd>{record.trigger_confidence.map(|value| format!("{:.0}%", value * 100.0)).unwrap_or_else(|| "—".to_string())}</dd>
                                {detection_href.map(|href| view! {
                                    <dt>"Detection"</dt><dd><A href=href>"Open originating detection"</A></dd>
                                })}
                            </dl>
                            <p class="transcript-review-text">{record.text}</p>
                        </section>
                        <DetectionAudioAnalysisView
                            detection_id=transcript_id.clone()
                            show_expand_link=false
                            container_class="audio-page".to_string()
                            instance_prefix="transcript-page".to_string()
                            audio_url_override=audio_url.clone()
                            download_url_override=audio_url.clone()
                            load_detection_metadata=false
                            analysis_title="Transcript audio".to_string()
                        />
                    }.into_any()
                }
            }}
        </div>
    }
}
