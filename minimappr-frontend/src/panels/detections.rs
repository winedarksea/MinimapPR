use crate::state::AppState;
use leptos::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlAudioElement;

fn audio_element() -> Option<HtmlAudioElement> {
    web_sys::window()?
        .document()?
        .get_element_by_id("audio-player")?
        .dyn_into::<HtmlAudioElement>()
        .ok()
}

fn play_detection(event_id: &str) {
    let url = format!(
        "/api/v1/detections/{}/audio",
        js_sys::encode_uri_component(event_id)
    );
    if let Some(audio) = audio_element() {
        audio.set_src(&url);
        let _ = audio.play();
    }
}

#[component]
pub fn DetectionsPane() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let detections = state.detections;

    view! {
        <div class="tab-pane">
            {move || {
                let dets: Vec<_> = detections.get().into_iter().collect();
                if dets.is_empty() {
                    return view! { <div class="empty-state">"No recent detections"</div> }.into_any();
                }
                view! {
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>"Label"</th>
                                <th>"Node"</th>
                                <th>"Conf"</th>
                                <th>"Audio"</th>
                            </tr>
                        </thead>
                        <tbody>
                            {dets.into_iter().map(|d| {
                                let label = d.label.clone().unwrap_or_else(|| "—".to_string());
                                let node  = d.node_id.clone().unwrap_or_else(|| "—".to_string());
                                let conf  = d.label_confidence.or(d.confidence).map(|c| format!("{:.0}%", c * 100.0)).unwrap_or_else(|| "—".to_string());
                                let has_audio = d.has_audio.unwrap_or(false);
                                let eid = d.event_id.clone();

                                view! {
                                    <tr>
                                        <td>{label}</td>
                                        <td style="font-size:0.7rem">{node}</td>
                                        <td><span class="conf-pill">{conf}</span></td>
                                        <td>
                                            {if has_audio {
                                                view! {
                                                    <button class="play-btn" on:click=move |_| play_detection(&eid)>
                                                        "▶"
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
