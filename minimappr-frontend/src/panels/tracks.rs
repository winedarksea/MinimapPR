use crate::state::AppState;
use leptos::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{HtmlAudioElement, Window};

fn audio_element() -> Option<HtmlAudioElement> {
    web_sys::window()?
        .document()?
        .get_element_by_id("audio-player")?
        .dyn_into::<HtmlAudioElement>()
        .ok()
}

fn play_track_audio(track_id: &str) {
    let url = format!(
        "/api/v1/tracks/{}/audio",
        js_sys::encode_uri_component(track_id),
    );
    if let Some(audio) = audio_element() {
        audio.set_src(&url);
        let _ = audio.play();
    }
}

fn trigger_track_download(track_id: &str) {
    let url = format!(
        "/api/v1/tracks/{}/audio?download=true",
        js_sys::encode_uri_component(track_id),
    );
    let window: Option<Window> = web_sys::window();
    if let Some(win) = window {
        let _ = win.open_with_url(&url);
    }
}

#[component]
pub fn TracksPane() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let tracks = state.tracks;
    let detections = state.detections;

    view! {
        <div class="tab-pane">
            {move || {
                let ts = tracks.get();
                if ts.is_empty() {
                    return view! { <div class="empty-state">"No active tracks"</div> }.into_any();
                }
                view! {
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>"ID"</th>
                                <th>"Label"</th>
                                <th>"Conf"</th>
                                <th>"TQI"</th>
                                <th>"Sensors"</th>
                                <th>"Pos (m)"</th>
                                <th>"Audio"</th>
                            </tr>
                        </thead>
                        <tbody>
                            {ts.into_iter().map(|t| {
                                let id   = t.track_id[..8.min(t.track_id.len())].to_string();
                                let label= t.label.clone().unwrap_or_else(|| "—".to_string());
                                let conf = t.confidence.map(|c| format!("{:.0}%", c * 100.0)).unwrap_or_else(|| "—".to_string());
                                let tqi_val = t.tqi.unwrap_or(0.0);
                                let tqi_w = format!("{}px", (tqi_val * 60.0) as u32);
                                let sensors = t.sensor_count.map(|s| s.to_string()).unwrap_or_else(|| "—".to_string());
                                let pos = t.position_m.as_ref().map(|p| {
                                    match p.as_slice() {
                                        [x, y, z] => format!("{x:.1},{y:.1},{z:.1}"),
                                        [x, y]    => format!("{x:.1},{y:.1}"),
                                        _          => "—".to_string(),
                                    }
                                }).unwrap_or_else(|| "—".to_string());
                                let track_id = t.track_id.clone();
                                let has_audio = detections
                                    .get()
                                    .iter()
                                    .any(|d| d.track_id.as_deref() == Some(track_id.as_str()) && d.snippet_path.is_some());

                                view! {
                                    <tr>
                                        <td><code>{id}</code></td>
                                        <td>{label}</td>
                                        <td><span class="conf-pill">{conf}</span></td>
                                        <td>
                                            <span class="tqi-bar" style:width=tqi_w></span>
                                        </td>
                                        <td>{sensors}</td>
                                        <td style="font-size:0.7rem">{pos}</td>
                                        <td>
                                            {if has_audio {
                                                let play_id = track_id.clone();
                                                let download_id = track_id.clone();
                                                view! {
                                                    <button class="play-btn" on:click=move |_| play_track_audio(&play_id)>
                                                        "▶"
                                                    </button>
                                                    <button class="btn-sm" on:click=move |_| trigger_track_download(&download_id)>
                                                        "Download"
                                                    </button>
                                                }.into_any()
                                            } else {
                                                view! { <span style="color:var(--text-muted)">"-"</span> }.into_any()
                                            }}
                                        </td>
                                    </tr>
                                }
                            }).collect_view()}
                        </tbody>
                    </table>
                }.into_any()
            }}
        </div>
    }
}
