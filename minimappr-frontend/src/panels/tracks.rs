use crate::state::AppState;
use crate::ui::{classify_age_from_ns, short_id, track_status_chip_class, track_status_label};
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
                                <th>"St"</th>
                                <th>"Label"</th>
                                <th>"Conf"</th>
                                <th title="Track Quality Index">"TQI"</th>
                                <th>"Age"</th>
                                <th>"Audio"</th>
                            </tr>
                        </thead>
                        <tbody>
                            {ts.into_iter().map(|t| {
                                let id_short = short_id(&t.track_id, 8);
                                let label = t.label.clone().unwrap_or_else(|| "—".to_string());
                                let conf = t.confidence
                                    .map(|c| format!("{:.0}%", c * 100.0))
                                    .unwrap_or_else(|| "—".to_string());
                                let tqi_val = t.tqi.unwrap_or(0.0);
                                let tqi_w = format!("{}px", (tqi_val * 60.0) as u32);
                                let tqi_pct = format!("{:.0}%", tqi_val * 100.0);
                                let (age_text, age_class) = classify_age_from_ns(t.last_update_ns, 30.0, 120.0);
                                let status_str = t.status.as_deref();
                                let st_class = track_status_chip_class(status_str);
                                let st_label = track_status_label(status_str);

                                // Geo tooltip: show on ID hover if available
                                let geo_title = t.position_geo.as_ref().map(|g| {
                                    match g.alt_m {
                                        Some(alt) => format!(
                                            "{:.5}°N  {:.5}°E  alt {:.1}m\n(click for details)",
                                            g.lat, g.lon, alt
                                        ),
                                        None => format!("{:.5}°N  {:.5}°E", g.lat, g.lon),
                                    }
                                }).unwrap_or_else(|| t.track_id.clone());

                                let track_id = t.track_id.clone();

                                view! {
                                    <tr class=move || if age_class == "age-lost" { "track-row-stale" } else { "" }>
                                        <td>
                                            <code class="track-id-code" title=geo_title>
                                                {id_short}
                                            </code>
                                        </td>
                                        <td>
                                            <span class=st_class>{st_label}</span>
                                        </td>
                                        <td>{label}</td>
                                        <td><span class="conf-pill">{conf}</span></td>
                                        <td>
                                            <span class="tqi-bar" style:width=tqi_w title=tqi_pct></span>
                                        </td>
                                        <td>
                                            <span class=age_class>{age_text}</span>
                                        </td>
                                        <td>
                                            {
                                                let play_id = track_id.clone();
                                                let download_id = track_id.clone();
                                                view! {
                                                    <button class="play-btn" title="Play" on:click=move |_| play_track_audio(&play_id)>
                                                        "▶"
                                                    </button>
                                                    <button class="btn-sm" title="Download" on:click=move |_| trigger_track_download(&download_id)>
                                                        "↓"
                                                    </button>
                                                }.into_any()
                                            }
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
