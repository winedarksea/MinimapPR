use crate::map::bindings::pan_to;
use crate::panels::contributors::CompactContributorChips;
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
    let selected_track = state.selected_track;

    view! {
        <div class="tab-pane">
            {move || {
                let ts = tracks.get();
                if ts.is_empty() {
                    return view! { <div class="empty-state">"No active tracks"</div> }.into_any();
                }
                view! {
                    <ul class="compact-list">
                        {ts.into_iter().map(|t| {
                            let id_short = short_id(&t.track_id, 8);
                            let label_text = t.label.clone().unwrap_or_else(|| format!("Track {id_short}"));
                            let conf = t.confidence
                                .map(|c| format!("{:.0}%", c * 100.0))
                                .unwrap_or_else(|| "—".to_string());
                            let tqi_val = t.tqi.unwrap_or(0.0);
                            let tqi_w = format!("{}px", (tqi_val * 40.0) as u32);
                            let tqi_pct = format!("{:.0}%", tqi_val * 100.0);
                            let (age_text, age_class) = classify_age_from_ns(t.last_update_ns, 30.0, 120.0);
                            let status_str = t.status.as_deref();
                            let st_class = track_status_chip_class(status_str);
                            let st_label = track_status_label(status_str);

                            let geo_title = t.position_geo.as_ref().map(|g| {
                                match g.alt_m {
                                    Some(alt) => format!(
                                        "{:.5}°N  {:.5}°E  alt {:.1}m",
                                        g.lat, g.lon, alt
                                    ),
                                    None => format!("{:.5}°N  {:.5}°E", g.lat, g.lon),
                                }
                            }).unwrap_or_else(|| t.track_id.clone());

                            let geo_display = t.position_geo.as_ref().map(|g| match g.alt_m {
                                Some(alt) => format!("{:.5}°N, {:.5}°E · {:.1} m", g.lat, g.lon, alt),
                                None => format!("{:.5}°N, {:.5}°E", g.lat, g.lon),
                            });

                            let track_id = t.track_id.clone();
                            let hover_id = track_id.clone();
                            let leave_id = track_id.clone();
                            let row_tid = track_id.clone();
                            let contributors = t.contributors.clone();
                            let contributor_count = t.contributor_count.max(contributors.len() as u32);
                            let geo_for_pan = t.position_geo.clone();

                            let row_class = move || {
                                let sel = selected_track.get();
                                let is_sel = sel.as_deref() == Some(&row_tid);
                                let stale = age_class == "age-lost";
                                match (stale, is_sel) {
                                    (true, true)   => "compact-row track-row-stale track-row-selected",
                                    (true, false)  => "compact-row track-row-stale",
                                    (false, true)  => "compact-row track-row-selected",
                                    (false, false) => "compact-row",
                                }
                            };

                            view! {
                                <li>
                                    <details
                                        class=row_class
                                        on:mouseenter=move |_| selected_track.set(Some(hover_id.clone()))
                                        on:mouseleave={
                                            let leave_id = leave_id.clone();
                                            move |_| {
                                                if selected_track.get_untracked().as_deref() == Some(&leave_id) {
                                                    selected_track.set(None);
                                                }
                                            }
                                        }
                                    >
                                        <summary>
                                            <span class=st_class title=geo_title.clone()>{st_label}</span>
                                            <span class="row-label">{label_text}</span>
                                            <span class="row-summary-meta">
                                                <span class="conf-pill">{conf}</span>
                                                <span class="tqi-bar" style:width=tqi_w title=tqi_pct></span>
                                                <span class=age_class>{age_text.clone()}</span>
                                            </span>
                                            <span class="row-chevron" aria-hidden="true">"▾"</span>
                                        </summary>
                                        <dl class="compact-detail">
                                            <dt>"ID"</dt>
                                            <dd>
                                                <code class="track-id-code" title=geo_title>{id_short}</code>
                                            </dd>

                                            <dt>"Geo"</dt>
                                            <dd>
                                                <div class="compact-detail-actions">
                                                    <span>{geo_display.clone().unwrap_or_else(|| "—".to_string())}</span>
                                                    {match geo_for_pan {
                                                        Some(geo) => {
                                                            let lat = geo.lat;
                                                            let lon = geo.lon;
                                                            view! {
                                                                <button
                                                                    class="btn-sm"
                                                                    title="Center map on track"
                                                                    on:click=move |_| pan_to(lat, lon)
                                                                >
                                                                    "Center on map"
                                                                </button>
                                                            }.into_any()
                                                        }
                                                        None => ().into_any(),
                                                    }}
                                                </div>
                                            </dd>

                                            <dt>"Nodes"</dt>
                                            <dd>
                                                <div class="compact-detail-actions">
                                                    <CompactContributorChips contributors=contributors fallback_node_id=None />
                                                    {if contributor_count > 0 {
                                                        view! {
                                                            <span class="track-contributor-count">
                                                                {format!("{contributor_count} nodes")}
                                                            </span>
                                                        }.into_any()
                                                    } else {
                                                        ().into_any()
                                                    }}
                                                </div>
                                            </dd>

                                            <dt>"Audio"</dt>
                                            <dd>
                                                <div class="compact-detail-actions">
                                                    {
                                                        let play_id = track_id.clone();
                                                        let download_id = track_id.clone();
                                                        view! {
                                                            <button class="play-btn" title="Play" on:click=move |_| play_track_audio(&play_id)>
                                                                "▶"
                                                            </button>
                                                            <button class="btn-sm" title="Download" on:click=move |_| trigger_track_download(&download_id)>
                                                                "↓ Download"
                                                            </button>
                                                        }
                                                    }
                                                </div>
                                            </dd>
                                        </dl>
                                    </details>
                                </li>
                            }
                        }).collect_view()}
                    </ul>
                }.into_any()
            }}
        </div>
    }
}
