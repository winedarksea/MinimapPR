use crate::state::AppState;
use crate::audio::detection_actions::DetectionAudioActions;
use leptos::prelude::*;

fn age_from_ns(ns: Option<i64>) -> (String, &'static str) {
    let Some(ns) = ns else {
        return ("—".to_string(), "age-unknown");
    };
    let age_secs = (js_sys::Date::now() * 1_000_000.0 - ns as f64) / 1_000_000_000.0;
    if age_secs < 0.0 {
        return ("0s".to_string(), "age-fresh");
    }
    let class = if age_secs < 30.0 {
        "age-fresh"
    } else if age_secs < 300.0 {
        "age-stale"
    } else {
        "age-lost"
    };
    (format_age_secs(age_secs as u64), class)
}

fn format_age_secs(s: u64) -> String {
    if s < 60 {
        format!("{}s", s)
    } else if s < 3600 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else {
        format!("{}h {:02}m", s / 3600, (s % 3600) / 60)
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
                                <th>"Age"</th>
                                <th>"Label"</th>
                                <th>"Node"</th>
                                <th>"Conf"</th>
                                <th title="Fused track ID">"Track"</th>
                                <th>"Audio"</th>
                            </tr>
                        </thead>
                        <tbody>
                            {dets.into_iter().map(|d| {
                                let label = d.label.clone().unwrap_or_else(|| "—".to_string());
                                let node  = d.node_id.clone().unwrap_or_else(|| "—".to_string());
                                let conf  = d.label_confidence.or(d.confidence)
                                    .map(|c| format!("{:.0}%", c * 100.0))
                                    .unwrap_or_else(|| "—".to_string());
                                let has_audio = d.has_audio.unwrap_or(false) || d.snippet_path.is_some();
                                let eid = d.event_id.clone();
                                let (age_text, age_class) = age_from_ns(d.received_ns);

                                // Short track ID chip if this detection was fused
                                let track_chip = d.track_id.as_ref().map(|tid| {
                                    tid[..8.min(tid.len())].to_string()
                                });

                                view! {
                                    <tr>
                                        <td>
                                            <span class=age_class>{age_text}</span>
                                        </td>
                                        <td>{label}</td>
                                        <td>
                                            <code class="node-chip">{node}</code>
                                        </td>
                                        <td><span class="conf-pill">{conf}</span></td>
                                        <td>
                                            {match track_chip {
                                                Some(tid) => view! {
                                                    <code class="track-link-chip">{tid}</code>
                                                }.into_any(),
                                                None => view! {
                                                    <span class="age-unknown">"-"</span>
                                                }.into_any(),
                                            }}
                                        </td>
                                        <td>
                                            {if has_audio {
                                                view! {
                                                    <DetectionAudioActions event_id=eid.clone() />
                                                }.into_any()
                                            } else {
                                                view! {
                                                    <span class="age-unknown">"-"</span>
                                                }.into_any()
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
