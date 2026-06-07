use crate::audio::detection_actions::DetectionAudioActions;
use crate::map::bindings::pan_to;
use crate::panels::contributors::CompactContributorChips;
use crate::state::{AppState, CopItemKind, CopSelection};
use crate::ui::{classify_age_from_ns, cop_sidebar_element_id, is_cop_item_selected, short_id};
use leptos::prelude::*;

#[component]
pub fn DetectionsPane() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let detections = state.detections;
    let selected_cop_item = state.selected_cop_item;

    view! {
        <div class="tab-pane">
            {move || {
                let dets: Vec<_> = detections.get().into_iter().collect();
                if dets.is_empty() {
                    return view! { <div class="empty-state">"No recent detections"</div> }.into_any();
                }
                view! {
                    <ul class="compact-list">
                        {dets.into_iter().map(|d| {
                            let label = d.label.clone().unwrap_or_else(|| "—".to_string());
                            let node  = d.node_id.clone().unwrap_or_else(|| "—".to_string());
                            let contributors = d.contributors.clone();
                            let conf  = d.label_confidence.or(d.confidence)
                                .map(|c| format!("{:.0}%", c * 100.0))
                                .unwrap_or_else(|| "—".to_string());
                            let has_audio = d.has_audio.unwrap_or(false) || d.snippet_path.is_some();
                            let eid = d.event_id.clone();
                            let hover_id = eid.clone();
                            let leave_id = eid.clone();
                            let click_id = eid.clone();
                            let row_id = eid.clone();
                            let row_element_id = cop_sidebar_element_id(CopItemKind::Detection, &eid);
                            let (age_text, age_class) = classify_age_from_ns(d.received_ns, 30.0, 300.0);

                            let track_chip = d.track_id.as_ref().map(|tid| short_id(tid, 8));
                            let dot_class = format!("row-status-dot {age_class}");
                            let geo_display = d.position_geo.as_ref().map(|g| match g.alt_m {
                                Some(alt) => format!("{:.5}°N, {:.5}°E · {:.1} m", g.lat, g.lon, alt),
                                None => format!("{:.5}°N, {:.5}°E", g.lat, g.lon),
                            });
                            let geo_for_pan = d.position_geo.clone();
                            let summary_geo = d.position_geo.as_ref().map(|g| (g.lat, g.lon));
                            let row_class = move || {
                                if is_cop_item_selected(
                                    &selected_cop_item.get(),
                                    CopItemKind::Detection,
                                    &row_id,
                                ) {
                                    "compact-row cop-row-selected"
                                } else {
                                    "compact-row"
                                }
                            };

                            view! {
                                <li>
                                    <details
                                        id=row_element_id
                                        class=row_class
                                        on:mouseenter=move |_| {
                                            selected_cop_item.set(Some(CopSelection::hovered(
                                                CopItemKind::Detection,
                                                hover_id.clone(),
                                            )));
                                        }
                                        on:mouseleave={
                                            let leave_id = leave_id.clone();
                                            move |_| {
                                                let should_clear = selected_cop_item
                                                    .get_untracked()
                                                    .as_ref()
                                                    .is_some_and(|selected| {
                                                        selected.kind == CopItemKind::Detection
                                                            && selected.id == leave_id
                                                            && !selected.pinned
                                                    });
                                                if should_clear {
                                                    selected_cop_item.set(None);
                                                }
                                            }
                                        }
                                    >
                                        <summary
                                            on:click=move |_| {
                                                selected_cop_item.set(Some(CopSelection::pinned(
                                                    CopItemKind::Detection,
                                                    click_id.clone(),
                                                )));
                                                if let Some((lat, lon)) = summary_geo {
                                                    pan_to(lat, lon);
                                                }
                                            }
                                        >
                                            <span class=dot_class title=age_text.clone()></span>
                                            <span class="row-label">{label}</span>
                                            <span class="row-summary-meta">
                                                <span class="conf-pill">{conf}</span>
                                                <span class=age_class>{age_text.clone()}</span>
                                            </span>
                                            <span class="row-chevron" aria-hidden="true">"▾"</span>
                                        </summary>
                                        <dl class="compact-detail">
                                            <dt>"Node"</dt>
                                            <dd>
                                                <CompactContributorChips contributors=contributors fallback_node_id=Some(node) />
                                            </dd>

                                            <dt>"Track"</dt>
                                            <dd>
                                                {match track_chip {
                                                    Some(tid) => view! {
                                                        <code class="track-link-chip">{tid}</code>
                                                    }.into_any(),
                                                    None => view! { <span class="age-unknown">"—"</span> }.into_any(),
                                                }}
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
                                                                    title="Center map on detection"
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

                                            <dt>"Audio"</dt>
                                            <dd>
                                                {if has_audio {
                                                    view! {
                                                        <div class="compact-detail-actions">
                                                            <DetectionAudioActions event_id=eid.clone() />
                                                        </div>
                                                    }.into_any()
                                                } else {
                                                    view! { <span class="age-unknown">"—"</span> }.into_any()
                                                }}
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
