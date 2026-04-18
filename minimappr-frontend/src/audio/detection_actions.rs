use leptos::prelude::*;
use leptos_router::components::A;
use crate::state::AppState;

pub fn detection_audio_url(event_id: &str) -> String {
    format!(
        "/api/v1/detections/{}/audio",
        js_sys::encode_uri_component(event_id)
    )
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

    let inspect_eid = event_id.clone();
    let download_eid = event_id.clone();
    let analysis_href = audio_analysis_href(&event_id);

    view! {
        <button
            class="play-btn"
            title="Open audio analysis drawer"
            on:click=move |_| {
                drawer_detection_id.set(Some(inspect_eid.clone()));
                drawer_open.set(true);
            }
        >
            "▶"
        </button>
        <button class="btn-sm" on:click=move |_| download_detection(&download_eid)>
            "Download"
        </button>
        <A href=analysis_href>
            <span class="btn-sm">"Analyze"</span>
        </A>
    }
}
