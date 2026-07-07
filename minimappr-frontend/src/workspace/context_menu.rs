use crate::api::aim_effector_at_position;
use crate::map::bindings::pan_to;
use crate::state::{AppState, Effector, SiteOriginSnapshot};
use leptos::prelude::*;
use wasm_bindgen_futures::{spawn_local, JsFuture};

const EARTH_RADIUS_M: f64 = 6_371_000.0;

#[component]
pub fn MapContextMenu() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let menu = state.map_context_menu;
    let effectors = state.effectors;
    let config = state.config;
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
                        <button type="button" disabled=true title="Zone draw/edit mode is backend pending in this slice">
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
}
