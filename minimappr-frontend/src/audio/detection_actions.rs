use crate::state::AppState;
use leptos::prelude::*;
use leptos_router::components::A;
use wasm_bindgen::JsCast;

pub fn detection_audio_url(event_id: &str) -> String {
    format!(
        "/api/v1/detections/{}/audio",
        js_sys::encode_uri_component(event_id)
    )
}

fn play_detection_audio(url: &str) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(doc) = window.document() else { return };
    let Some(el) = doc.get_element_by_id("audio-player") else {
        return;
    };
    if let Ok(audio) = el.dyn_into::<web_sys::HtmlAudioElement>() {
        audio.set_src(url);
        let _ = audio.play();
    }
}

pub fn detection_audio_download_url(event_id: &str) -> String {
    format!(
        "/api/v1/detections/{}/audio?download=true",
        js_sys::encode_uri_component(event_id)
    )
}

pub fn audio_analysis_href(event_id: &str) -> String {
    format!("/audio/d/{}", js_sys::encode_uri_component(event_id))
}

pub fn download_detection(event_id: &str) {
    let url = detection_audio_download_url(event_id);
    if let Some(window) = web_sys::window() {
        let _ = window.open_with_url(&url);
    }
}

#[component]
pub fn DetectionAudioActions(event_id: String) -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let drawer_open = state.audio_drawer_open;
    let drawer_detection_id = state.audio_drawer_detection_id;
    let drawer_track_id = state.audio_drawer_track_id;

    let play_eid = event_id.clone();
    let download_eid = event_id.clone();
    let analysis_href = audio_analysis_href(&event_id);
    let inspect_eid = event_id.clone();

    view! {
        <button
            class="play-btn"
            title="Play detection audio"
            on:click=move |_| {
                play_detection_audio(&detection_audio_url(&play_eid));
            }
        >
            "▶"
        </button>
        <button class="btn-sm" on:click=move |_| download_detection(&download_eid)>
            "Download"
        </button>
        <button class="btn-sm" on:click=move |_| {
            drawer_detection_id.set(Some(inspect_eid.clone()));
            drawer_track_id.set(None);
            drawer_open.set(true);
        }>
            "Analyze"
        </button>
        <A href=analysis_href>
            <span class="btn-sm">"⤢"</span>
        </A>
    }
}
