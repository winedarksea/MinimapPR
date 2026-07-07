use crate::api::{aim_effector_at_position, upsert_zone};
use crate::map::bindings::pan_to;
use crate::state::{AppState, Effector, SiteOriginSnapshot, ZoneDraft, ZoneSpec};
use leptos::prelude::*;
use serde_json::json;
use wasm_bindgen_futures::{spawn_local, JsFuture};

const EARTH_RADIUS_M: f64 = 6_371_000.0;
const DEFAULT_ZONE_RADIUS_M: f64 = 25.0;

#[component]
pub fn MapContextMenu() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let menu = state.map_context_menu;
    let effectors = state.effectors;
    let config = state.config;
    let zone_draft = state.zone_draft;
    let action_status = RwSignal::new(None::<String>);

    let close_menu = move |_| {
        action_status.set(None);
        menu.set(None);
    };

    view! {
        {move || {
            let Some(menu_state) = menu.get() else {
                return ().into_any();
            };
            let target_m = config
                .get()
                .and_then(|snapshot| snapshot.site_origin)
                .map(|origin| geo_to_local_m(&origin, menu_state.lat, menu_state.lon));
            let positioned_effectors = nearby_effectors(&effectors.get(), menu_state.lat, menu_state.lon);
            let position_style = format!(
                "left: {:.0}px; top: {:.0}px;",
                menu_state.screen_x.clamp(12.0, 9_999.0),
                menu_state.screen_y.clamp(12.0, 9_999.0),
            );
            view! {
                <div class="map-context-menu" style=position_style role="menu">
                    <div class="map-context-menu-head">
                        <span class="tone-badge neutral">"Map point"</span>
                        <button class="map-context-close" type="button" on:click=close_menu aria-label="Close context menu">
                            <span class="material-symbols-rounded" aria-hidden="true">"close"</span>
                        </button>
                    </div>
                    <code>{format!("{:.6}, {:.6}", menu_state.lat, menu_state.lon)}</code>
                    <div class="map-context-actions">
                        <button
                            type="button"
                            on:click=move |_| {
                                let text = format!("{:.6}, {:.6}", menu_state.lat, menu_state.lon);
                                action_status.set(Some("Copying coordinates...".to_string()));
                                spawn_local(async move {
                                    match copy_text_to_clipboard(text).await {
                                        Ok(()) => action_status.set(Some("Coordinates copied".to_string())),
                                        Err(reason) => action_status.set(Some(reason)),
                                    }
                                });
                            }
                        >
                            "Copy coords"
                        </button>
                        <button
                            type="button"
                            on:click=move |_| {
                                pan_to(menu_state.lat, menu_state.lon);
                                menu.set(None);
                            }
                        >
                            "Center map here"
                        </button>
                        <button
                            type="button"
                            title="Create a starter square zone centered on this point"
                            on:click=move |_| {
                                zone_draft.set(Some(new_zone_draft(menu_state.lat, menu_state.lon)));
                                action_status.set(None);
                                menu.set(None);
                            }
                        >
                            "Draw zone from here"
                        </button>
                    </div>

                    <div class="map-context-section">
                        <span class="map-context-label">"Slew camera"</span>
                        {if positioned_effectors.is_empty() {
                            view! { <div class="map-context-empty">"No registered cameras"</div> }.into_any()
                        } else {
                            view! {
                                <div class="map-context-effector-list">
                                    {positioned_effectors.into_iter().take(4).map(|effector| {
                                        let effector_id = effector.id.clone();
                                        let armed = effector.status.as_ref().map(|status| status.armed).unwrap_or(false);
                                        let disabled_reason = if target_m.is_none() {
                                            Some("Site origin required for lat/lon targeting")
                                        } else if !armed {
                                            Some("Camera is disarmed")
                                        } else {
                                            None
                                        };
                                        let button_label = effector.id.clone();
                                        view! {
                                            <button
                                                type="button"
                                                disabled=disabled_reason.is_some()
                                                title=disabled_reason.unwrap_or("Slew this camera to the map point")
                                                on:click=move |_| {
                                                    let Some(target_m) = target_m else {
                                                        action_status.set(Some("Site origin required for lat/lon targeting".to_string()));
                                                        return;
                                                    };
                                                    let effector_id = effector_id.clone();
                                                    action_status.set(Some(format!("Slewing {effector_id}...")));
                                                    spawn_local(async move {
                                                        match aim_effector_at_position(&effector_id, target_m).await {
                                                            Ok(()) => {
                                                                action_status.set(Some(format!("Slew command sent to {effector_id}")));
                                                                menu.set(None);
                                                            }
                                                            Err(reason) => action_status.set(Some(reason)),
                                                        }
                                                    });
                                                }
                                            >
                                                <span>{button_label}</span>
                                                <small>{if armed { "armed" } else { "disarmed" }}</small>
                                            </button>
                                        }
                                    }).collect_view()}
                                </div>
                            }.into_any()
                        }}
                    </div>
                    {move || action_status.get().map(|status| view! {
                        <span class="map-context-status">{status}</span>
                    })}
                </div>
            }.into_any()
        }}
    }
}

#[component]
pub fn ZoneDraftDialog() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let zone_draft = state.zone_draft;
    let zones = state.zones;
    let selected_cop_item = state.selected_cop_item;
    let error_message = RwSignal::new(None::<String>);

    view! {
        {move || {
            let Some(draft) = zone_draft.get() else {
                return ().into_any();
            };
            let id_signal = RwSignal::new(draft.id.clone());
            let name_signal = RwSignal::new(draft.name.clone());
            let type_signal = RwSignal::new(draft.zone_type.clone());
            let radius_signal = RwSignal::new(draft.radius_m.to_string());
            let center_lat = draft.center_lat;
            let center_lon = draft.center_lon;

            view! {
                <div class="zone-draft-scrim" role="presentation">
                    <section class="zone-draft-dialog" role="dialog" aria-modal="true" aria-label="Create zone">
                        <header class="zone-draft-header">
                            <div>
                                <span class="tone-badge neutral">"New zone"</span>
                                <h3>"Create map zone"</h3>
                            </div>
                            <button
                                class="map-context-close"
                                type="button"
                                aria-label="Cancel zone creation"
                                on:click=move |_| {
                                    error_message.set(None);
                                    zone_draft.set(None);
                                }
                            >
                                <span class="material-symbols-rounded" aria-hidden="true">"close"</span>
                            </button>
                        </header>

                        <label class="zone-draft-field">
                            <span>"ID"</span>
                            <input
                                type="text"
                                prop:value=move || id_signal.get()
                                on:input=move |event| id_signal.set(event_target_value(&event))
                            />
                        </label>
                        <label class="zone-draft-field">
                            <span>"Name"</span>
                            <input
                                type="text"
                                prop:value=move || name_signal.get()
                                on:input=move |event| name_signal.set(event_target_value(&event))
                            />
                        </label>
                        <label class="zone-draft-field">
                            <span>"Type"</span>
                            <select
                                prop:value=move || type_signal.get()
                                on:change=move |event| type_signal.set(event_target_value(&event))
                            >
                                <option value="interest_zone">"Interest"</option>
                                <option value="alert_zone">"Alert"</option>
                                <option value="exclusion_zone">"Exclusion"</option>
                                <option value="coverage_zone">"Coverage"</option>
                            </select>
                        </label>
                        <label class="zone-draft-field">
                            <span>"Half-width meters"</span>
                            <input
                                type="number"
                                min="1"
                                max="10000"
                                step="1"
                                prop:value=move || radius_signal.get()
                                on:input=move |event| radius_signal.set(event_target_value(&event))
                            />
                        </label>
                        <code>{format!("{center_lat:.6}, {center_lon:.6}")}</code>
                        {move || error_message.get().map(|message| view! {
                            <span class="daily-error">{message}</span>
                        })}
                        <footer class="zone-draft-actions">
                            <button
                                class="btn-sm"
                                type="button"
                                on:click=move |_| {
                                    error_message.set(None);
                                    zone_draft.set(None);
                                }
                            >
                                "Cancel"
                            </button>
                            <button
                                class="btn-sm active"
                                type="button"
                                on:click=move |_| {
                                    let id = id_signal.get().trim().to_string();
                                    let name = name_signal.get().trim().to_string();
                                    let zone_type = type_signal.get();
                                    let radius_m = match radius_signal.get().parse::<f64>() {
                                        Ok(value) if value.is_finite() && value > 0.0 => value,
                                        _ => {
                                            error_message.set(Some("Half-width must be a positive number".to_string()));
                                            return;
                                        }
                                    };
                                    if id.is_empty() || name.is_empty() {
                                        error_message.set(Some("ID and name are required".to_string()));
                                        return;
                                    }
                                    let zone = ZoneSpec {
                                        id: id.clone(),
                                        name,
                                        zone_type,
                                        polygon_geo: square_polygon_around(center_lat, center_lon, radius_m),
                                        properties: json!({
                                            "created_from": "map_context_menu",
                                            "center_geo": [center_lat, center_lon],
                                            "half_width_m": radius_m,
                                        }),
                                        created_ns: None,
                                    };
                                    spawn_local(async move {
                                        match upsert_zone(zone.clone()).await {
                                            Ok(()) => {
                                                zones.update(|current| {
                                                    if let Some(existing) = current.iter_mut().find(|item| item.id == zone.id) {
                                                        *existing = zone.clone();
                                                    } else {
                                                        current.push(zone.clone());
                                                    }
                                                });
                                                selected_cop_item.set(Some(crate::state::CopSelection::pinned(
                                                    crate::state::CopItemKind::Zone,
                                                    zone.id.clone(),
                                                )));
                                                zone_draft.set(None);
                                            }
                                            Err(reason) => error_message.set(Some(reason)),
                                        }
                                    });
                                }
                            >
                                "Save zone"
                            </button>
                        </footer>
                    </section>
                </div>
            }.into_any()
        }}
    }
}

fn nearby_effectors(effectors: &[Effector], lat: f64, lon: f64) -> Vec<Effector> {
    let mut positioned = effectors
        .iter()
        .filter(|effector| effector.position_geo.is_some())
        .cloned()
        .collect::<Vec<_>>();
    positioned.sort_by(|left, right| {
        let left_distance = left
            .position_geo
            .as_ref()
            .map(|geo| distance_m_between_points(lat, lon, geo.lat, geo.lon))
            .unwrap_or(f64::INFINITY);
        let right_distance = right
            .position_geo
            .as_ref()
            .map(|geo| distance_m_between_points(lat, lon, geo.lat, geo.lon))
            .unwrap_or(f64::INFINITY);
        left_distance.total_cmp(&right_distance)
    });
    positioned
}

fn geo_to_local_m(origin: &SiteOriginSnapshot, lat: f64, lon: f64) -> [f64; 3] {
    let origin_lat_rad = origin.lat.to_radians();
    let north_m = (lat - origin.lat).to_radians() * EARTH_RADIUS_M;
    let east_m = (lon - origin.lon).to_radians() * EARTH_RADIUS_M * origin_lat_rad.cos();
    [east_m, north_m, 0.0]
}

fn distance_m_between_points(
    source_lat: f64,
    source_lon: f64,
    target_lat: f64,
    target_lon: f64,
) -> f64 {
    let delta_lat = (target_lat - source_lat).to_radians();
    let delta_lon = (target_lon - source_lon).to_radians();
    let source_lat = source_lat.to_radians();
    let target_lat = target_lat.to_radians();
    let a = (delta_lat / 2.0).sin().powi(2)
        + source_lat.cos() * target_lat.cos() * (delta_lon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_M * a.sqrt().atan2((1.0 - a).sqrt())
}

async fn copy_text_to_clipboard(text: String) -> Result<(), String> {
    let window = web_sys::window().ok_or_else(|| "Clipboard unavailable".to_string())?;
    let clipboard = window.navigator().clipboard();
    JsFuture::from(clipboard.write_text(&text))
        .await
        .map(|_| ())
        .map_err(|_| "Clipboard write failed".to_string())
}

fn new_zone_draft(lat: f64, lon: f64) -> ZoneDraft {
    let timestamp_ms = js_sys::Date::now() as i64;
    ZoneDraft {
        id: format!("zone-{timestamp_ms}"),
        name: "New map zone".to_string(),
        zone_type: "interest_zone".to_string(),
        center_lat: lat,
        center_lon: lon,
        radius_m: DEFAULT_ZONE_RADIUS_M,
    }
}

fn square_polygon_around(lat: f64, lon: f64, half_width_m: f64) -> Vec<Vec<f64>> {
    let delta_lat = (half_width_m / EARTH_RADIUS_M).to_degrees();
    let lon_scale = lat.to_radians().cos().abs().max(0.01);
    let delta_lon = (half_width_m / (EARTH_RADIUS_M * lon_scale)).to_degrees();
    vec![
        vec![lat - delta_lat, lon - delta_lon],
        vec![lat - delta_lat, lon + delta_lon],
        vec![lat + delta_lat, lon + delta_lon],
        vec![lat + delta_lat, lon - delta_lon],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geo_to_local_m_uses_east_north_axes() {
        let origin = SiteOriginSnapshot {
            lat: 45.0,
            lon: -93.0,
            alt_m: Some(260.0),
            reconcile_delay_seconds: None,
            mode: None,
            source: None,
        };

        let local = geo_to_local_m(&origin, 45.001, -92.999);

        assert!(local[0] > 78.0 && local[0] < 79.5);
        assert!(local[1] > 111.0 && local[1] < 112.0);
        assert_eq!(local[2], 0.0);
    }

    #[test]
    fn square_polygon_uses_four_lat_lon_corners() {
        let polygon = square_polygon_around(45.0, -93.0, 25.0);

        assert_eq!(polygon.len(), 4);
        assert!(polygon.iter().all(|point| point.len() == 2));
        assert!(polygon[0][0] < 45.0);
        assert!(polygon[2][0] > 45.0);
        assert!(polygon[0][1] < -93.0);
        assert!(polygon[1][1] > -93.0);
    }
}
