//! Field Mode "Heard nearby" tray: unlocalized bird entities, listed (never
//! pinned on the map). When a later bout gives a bearing or fix, the entity
//! leaves the tray and appears on the map as the same entity.

use crate::state::entities::BirdEntity;
use crate::state::AppState;
use crate::ui::format_age_seconds;
use leptos::prelude::*;

const NANOSECONDS_PER_SECOND: f64 = 1_000_000_000.0;
const NANOSECONDS_PER_MILLISECOND: f64 = 1_000_000.0;
/// Ages beyond this (or negative) mean event time is not wall time (replayed
/// bundle): show the clock time instead of a misleading age.
const MAX_PLAUSIBLE_AGE_SECONDS: f64 = 86_400.0;
const PERCENT: f64 = 100.0;

fn heard_when(timestamp_ns: i64) -> String {
    let age_seconds = (js_sys::Date::now() * NANOSECONDS_PER_MILLISECOND - timestamp_ns as f64) / NANOSECONDS_PER_SECOND;
    if (0.0..MAX_PLAUSIBLE_AGE_SECONDS).contains(&age_seconds) {
        return format!("{} ago", format_age_seconds(age_seconds as u64));
    }
    let date = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(timestamp_ns as f64 / NANOSECONDS_PER_MILLISECOND));
    format!("{:02}:{:02}:{:02}", date.get_hours(), date.get_minutes(), date.get_seconds())
}

fn tray_row(entity: BirdEntity) -> impl IntoView {
    let probability = format!("{:.0}%", entity.species_top_probability * PERCENT);
    let node = entity
        .reporting_node_id
        .clone()
        .map(|node_id| format!("near {node_id}"))
        .unwrap_or_else(|| "nearby".to_string());
    let heard_count = entity.detection_count + entity.absorbed_detection_count;
    let first_heard = heard_when(entity.first_heard_ns);
    let last_heard = heard_when(entity.last_heard_ns);
    view! {
        <li class="heard-nearby-row" data-entity-id=entity.id.clone()>
            <span class="material-symbols-rounded heard-nearby-icon" aria-hidden="true">"hearing"</span>
            <div class="heard-nearby-text">
                <div class="heard-nearby-species">
                    <span>{entity.species_top_label.clone()}</span>
                    <span class="heard-nearby-probability">{probability}</span>
                </div>
                <div class="heard-nearby-meta">
                    {format!("{node} · {heard_count}× · last {last_heard} · first {first_heard}")}
                </div>
            </div>
        </li>
    }
}

#[component]
pub fn HeardNearbyTray() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let open = RwSignal::new(true);
    let heard_nearby = Memo::new(move |_| {
        state
            .bird_entities
            .get()
            .into_iter()
            .filter(BirdEntity::is_heard_nearby)
            .collect::<Vec<_>>()
    });
    let spatial_count = Signal::derive(move || {
        state.bird_entities.with(|entities| entities.iter().filter(|e| !e.is_heard_nearby()).count())
    });

    view! {
        <Show when=move || state.bird_entities_enabled.get()>
            <aside class="heard-nearby-tray" aria-label="Heard nearby">
                <button
                    type="button"
                    class="heard-nearby-header"
                    aria-expanded=move || open.get().to_string()
                    on:click=move |_| open.update(|value| *value = !*value)
                >
                    <span class="material-symbols-rounded" aria-hidden="true">"hearing"</span>
                    <span class="heard-nearby-title">"Heard nearby"</span>
                    <span class="heard-nearby-badge">{move || heard_nearby.get().len()}</span>
                    <span class="heard-nearby-spatial">{move || format!("{} on map", spatial_count.get())}</span>
                    <span class="material-symbols-rounded" aria-hidden="true">
                        {move || if open.get() { "expand_less" } else { "expand_more" }}
                    </span>
                </button>
                <Show when=move || open.get()>
                    {move || {
                        let rows = heard_nearby.get();
                        if rows.is_empty() {
                            view! { <p class="heard-nearby-empty">"No unlocalized birds right now."</p> }.into_any()
                        } else {
                            view! {
                                <ul class="heard-nearby-list">
                                    {rows.into_iter().map(tray_row).collect_view()}
                                </ul>
                            }
                            .into_any()
                        }
                    }}
                </Show>
            </aside>
        </Show>
    }
}
