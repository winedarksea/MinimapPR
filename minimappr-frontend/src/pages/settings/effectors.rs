use crate::api::{
    arm_effector, delete_effector, disarm_effector, get_effector_safety, patch_effector_safety,
    register_effector, EffectorSafetyConfig,
};
use crate::state::{AppState, Effector};
use leptos::prelude::*;
use leptos::task::spawn_local;

#[derive(Clone, Debug, Default)]
struct RegisterForm {
    id: String,
    host: String,
    port: String,
    username: String,
    password: String,
    rtsp_url: String,
    lat: String,
    lon: String,
    alt_m: String,
    yaw_deg: String,
    pitch_deg: String,
}

fn parse_or(text: &str, default: f64) -> f64 {
    text.trim().parse::<f64>().unwrap_or(default)
}

fn parse_zone_ids(text: &str) -> Vec<String> {
    text.split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn join_zone_ids(zone_ids: &[String]) -> String {
    zone_ids.join(", ")
}

fn parse_optional_nonnegative_float(text: &str) -> Result<Option<f64>, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let value = trimmed
        .parse::<f64>()
        .map_err(|_| "Min slew interval must be a number".to_string())?;
    if value < 0.0 {
        return Err("Min slew interval cannot be negative".to_string());
    }
    Ok(Some(value))
}

fn update_effector_armed(effectors: RwSignal<Vec<Effector>>, effector_id: &str, armed: bool) {
    effectors.update(|list| {
        if let Some(effector) = list.iter_mut().find(|item| item.id == effector_id) {
            if let Some(status) = effector.status.as_mut() {
                status.armed = armed;
                status.state = "idle".to_string();
            }
        }
    });
}

fn build_register_payload(form: &RegisterForm) -> Result<serde_json::Value, String> {
    if form.id.trim().is_empty() {
        return Err("Camera ID is required".to_string());
    }
    if form.host.trim().is_empty() {
        return Err("Host/IP is required".to_string());
    }
    let lat: f64 = form
        .lat
        .trim()
        .parse()
        .map_err(|_| "Latitude must be a number".to_string())?;
    let lon: f64 = form
        .lon
        .trim()
        .parse()
        .map_err(|_| "Longitude must be a number".to_string())?;

    let mut transport = serde_json::json!({
        "host": form.host.trim(),
        "port": parse_or(&form.port, 80.0) as u32,
        "username": form.username.trim(),
        "password": form.password,
    });
    if !form.rtsp_url.trim().is_empty() {
        transport["rtsp_url"] = serde_json::Value::String(form.rtsp_url.trim().to_string());
    }

    Ok(serde_json::json!({
        "id": form.id.trim(),
        "effector_type": "camera_ptz",
        "position_geo": { "lat": lat, "lon": lon, "alt_m": parse_or(&form.alt_m, 0.0) },
        "orientation": {
            "yaw_deg": parse_or(&form.yaw_deg, 0.0),
            "pitch_deg": parse_or(&form.pitch_deg, 0.0),
        },
        "capabilities": ["ptz", "snapshot"],
        "transport": transport,
    }))
}

#[component]
pub fn EffectorsView() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let effectors = state.effectors;

    let form: RwSignal<RegisterForm> = RwSignal::new(RegisterForm::default());
    let submitting = RwSignal::new(false);
    let submit_error: RwSignal<Option<String>> = RwSignal::new(None);
    let show_form = RwSignal::new(false);

    let on_register = move |_| {
        submit_error.set(None);
        let payload = match build_register_payload(&form.get_untracked()) {
            Ok(p) => p,
            Err(e) => {
                submit_error.set(Some(e));
                return;
            }
        };
        submitting.set(true);
        spawn_local(async move {
            match register_effector(payload).await {
                Ok(effector) => {
                    effectors.update(|list| {
                        if !list.iter().any(|e| e.id == effector.id) {
                            list.push(effector);
                        }
                    });
                    form.set(RegisterForm::default());
                    show_form.set(false);
                }
                Err(e) => submit_error.set(Some(e)),
            }
            submitting.set(false);
        });
    };

    view! {
        <div class="page-stub">
            <h2 style="margin:0">"Effector Cameras"</h2>
            <p class="muted" style="font-size:0.85rem">
                "Register a PTZ camera to aim it at tracks from the COP view. Optional — with no camera "
                "registered, this is the only effector UI you'll see."
            </p>

            {move || {
                let list = effectors.get();
                if list.is_empty() {
                    view! { <div class="empty-state"><p>"No cameras registered yet."</p></div> }.into_any()
                } else {
                    view! {
                        <div class="effector-card-grid">
                            {list.into_iter().map(|effector| {
                                view! {
                                    <EffectorSafetyCard effector_id=effector.id.clone() effectors=effectors />
                                }
                            }).collect_view()}
                        </div>
                    }.into_any()
                }
            }}

            {move || if show_form.get() {
                view! {
                    <div class="diag-card" style="max-width:32rem">
                        <div class="pipeline-section-label">"Add camera"</div>
                        <div class="compact-form-grid">
                            <label>"Camera ID"
                                <input type="text" placeholder="cam-backyard"
                                    prop:value=move || form.get().id
                                    on:input=move |ev| form.update(|f| f.id = event_target_value(&ev))
                                />
                            </label>
                            <label>"Host / IP"
                                <input type="text" placeholder="192.168.1.50"
                                    prop:value=move || form.get().host
                                    on:input=move |ev| form.update(|f| f.host = event_target_value(&ev))
                                />
                            </label>
                            <label>"Port"
                                <input type="number" placeholder="80"
                                    prop:value=move || form.get().port
                                    on:input=move |ev| form.update(|f| f.port = event_target_value(&ev))
                                />
                            </label>
                            <label>"Username"
                                <input type="text"
                                    prop:value=move || form.get().username
                                    on:input=move |ev| form.update(|f| f.username = event_target_value(&ev))
                                />
                            </label>
                            <label>"Password"
                                <input type="password"
                                    prop:value=move || form.get().password
                                    on:input=move |ev| form.update(|f| f.password = event_target_value(&ev))
                                />
                            </label>
                            <label>"RTSP URL (optional fallback)"
                                <input type="text" placeholder="rtsp://192.168.1.50/stream1"
                                    prop:value=move || form.get().rtsp_url
                                    on:input=move |ev| form.update(|f| f.rtsp_url = event_target_value(&ev))
                                />
                            </label>
                            <label>"Latitude"
                                <input type="number" step="0.000001"
                                    prop:value=move || form.get().lat
                                    on:input=move |ev| form.update(|f| f.lat = event_target_value(&ev))
                                />
                            </label>
                            <label>"Longitude"
                                <input type="number" step="0.000001"
                                    prop:value=move || form.get().lon
                                    on:input=move |ev| form.update(|f| f.lon = event_target_value(&ev))
                                />
                            </label>
                            <label>"Altitude (m)"
                                <input type="number" step="0.1"
                                    prop:value=move || form.get().alt_m
                                    on:input=move |ev| form.update(|f| f.alt_m = event_target_value(&ev))
                                />
                            </label>
                            <label>"Home yaw (deg, 0=north)"
                                <input type="number" step="1"
                                    prop:value=move || form.get().yaw_deg
                                    on:input=move |ev| form.update(|f| f.yaw_deg = event_target_value(&ev))
                                />
                            </label>
                            <label>"Home pitch (deg, 0=level)"
                                <input type="number" step="1"
                                    prop:value=move || form.get().pitch_deg
                                    on:input=move |ev| form.update(|f| f.pitch_deg = event_target_value(&ev))
                                />
                            </label>
                        </div>
                        <div class="pipeline-save-row">
                            <button class="btn-primary" disabled=move || submitting.get() on:click=on_register>
                                {move || if submitting.get() { "Registering…" } else { "Register camera" }}
                            </button>
                            <button class="btn-sm" on:click=move |_| show_form.set(false)>"Cancel"</button>
                            {move || submit_error.get().map(|e| view! { <span class="daily-error">{e}</span> })}
                        </div>
                    </div>
                }.into_any()
            } else {
                view! {
                    <button class="btn-primary" on:click=move |_| show_form.set(true)>
                        "+ Add camera"
                    </button>
                }.into_any()
            }}
        </div>
    }
}

#[component]
fn EffectorSafetyCard(effector_id: String, effectors: RwSignal<Vec<Effector>>) -> impl IntoView {
    let safety = RwSignal::new(EffectorSafetyConfig::default());
    let min_interval_text = RwSignal::new(String::new());
    let no_go_zone_text = RwSignal::new(String::new());
    let loading = RwSignal::new(true);
    let saving = RwSignal::new(false);
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let notice: RwSignal<Option<String>> = RwSignal::new(None);

    let load_safety = {
        let effector_id = effector_id.clone();
        move || {
            loading.set(true);
            error.set(None);
            notice.set(None);
            let id = effector_id.clone();
            spawn_local(async move {
                match get_effector_safety(&id).await {
                    Ok(next) => {
                        min_interval_text.set(
                            next.min_slew_interval_seconds
                                .map(|value| value.to_string())
                                .unwrap_or_default(),
                        );
                        no_go_zone_text.set(join_zone_ids(&next.no_go_zone_ids));
                        safety.set(next);
                    }
                    Err(message) => error.set(Some(message)),
                }
                loading.set(false);
            });
        }
    };
    load_safety();

    let save_safety = {
        let effector_id = effector_id.clone();
        move |_| {
            error.set(None);
            notice.set(None);
            let min_slew_interval_seconds =
                match parse_optional_nonnegative_float(&min_interval_text.get_untracked()) {
                    Ok(value) => value,
                    Err(message) => {
                        error.set(Some(message));
                        return;
                    }
                };
            let mut next = safety.get_untracked();
            next.min_slew_interval_seconds = min_slew_interval_seconds;
            next.no_go_zone_ids = parse_zone_ids(&no_go_zone_text.get_untracked());
            saving.set(true);
            let id = effector_id.clone();
            spawn_local(async move {
                match patch_effector_safety(&id, next).await {
                    Ok(saved) => {
                        min_interval_text.set(
                            saved
                                .min_slew_interval_seconds
                                .map(|value| value.to_string())
                                .unwrap_or_default(),
                        );
                        no_go_zone_text.set(join_zone_ids(&saved.no_go_zone_ids));
                        safety.set(saved);
                        notice.set(Some("Safety saved".to_string()));
                    }
                    Err(message) => error.set(Some(message)),
                }
                saving.set(false);
            });
        }
    };

    let on_arm = {
        let effector_id = effector_id.clone();
        move |_| {
            error.set(None);
            notice.set(None);
            let id = effector_id.clone();
            saving.set(true);
            spawn_local(async move {
                match arm_effector(&id).await {
                    Ok(_) => {
                        update_effector_armed(effectors, &id, true);
                        notice.set(Some("Camera armed".to_string()));
                    }
                    Err(message) => error.set(Some(message)),
                }
                saving.set(false);
            });
        }
    };

    let on_disarm = {
        let effector_id = effector_id.clone();
        move |_| {
            error.set(None);
            notice.set(None);
            let id = effector_id.clone();
            saving.set(true);
            spawn_local(async move {
                match disarm_effector(&id).await {
                    Ok(_) => {
                        update_effector_armed(effectors, &id, false);
                        notice.set(Some("Camera disarmed".to_string()));
                    }
                    Err(message) => error.set(Some(message)),
                }
                saving.set(false);
            });
        }
    };

    let on_delete = {
        let effector_id = effector_id.clone();
        move |_| {
            let id = effector_id.clone();
            spawn_local(async move {
                if delete_effector(&id).await.is_ok() {
                    effectors.update(|list| list.retain(|effector| effector.id != id));
                }
            });
        }
    };

    let status_effector_id = effector_id.clone();
    let status_label = Signal::derive(move || {
        effectors.with(|list| {
            list.iter()
                .find(|item| item.id == status_effector_id)
                .and_then(|effector| effector.status.as_ref())
                .map(|status| status.state.clone())
                .unwrap_or_else(|| "offline".to_string())
        })
    });
    let armed_effector_id = effector_id.clone();
    let armed = Signal::derive(move || {
        effectors.with(|list| {
            list.iter()
                .find(|item| item.id == armed_effector_id)
                .and_then(|effector| effector.status.as_ref())
                .map(|status| status.armed)
                .unwrap_or(false)
        })
    });
    let health_class = move || match status_label.get().as_str() {
        "idle" | "slewing" | "streaming" => "health-chip online",
        "error" => "health-chip offline",
        _ => "health-chip degraded",
    };

    view! {
        <section class="diag-card effector-safety-card">
            <div class="effector-card-header">
                <div>
                    <div class="pipeline-section-label">{effector_id.clone()}</div>
                    <div class="effector-card-status">
                        <span class=health_class>{move || status_label.get()}</span>
                        <span class=move || if armed.get() { "tone-badge danger" } else { "tone-badge neutral" }>
                            {move || if armed.get() { "armed" } else { "disarmed" }}
                        </span>
                    </div>
                </div>
                <button class="btn-sm btn-sm--danger" title="Remove camera" on:click=on_delete>
                    "Remove"
                </button>
            </div>

            <div class="effector-command-row">
                <button class="btn-primary" disabled=move || saving.get() || armed.get() on:click=on_arm>
                    "Arm"
                </button>
                <button class="btn-sm btn-sm--danger" disabled=move || saving.get() || !armed.get() on:click=on_disarm>
                    "Disarm"
                </button>
                <button class="btn-sm" disabled=move || loading.get() || saving.get() on:click=move |_| load_safety()>
                    "Reload safety"
                </button>
            </div>

            <div class="compact-form-grid effector-safety-grid">
                <label class="rule-enabled-toggle effector-checkbox-row">
                    <input
                        type="checkbox"
                        prop:checked=move || safety.get().require_arm_for_slew
                        on:change=move |event| {
                            let checked = event_target_checked(&event);
                            safety.update(|item| item.require_arm_for_slew = checked);
                        }
                    />
                    <span>"Require arm for slew"</span>
                </label>
                <label>"Min slew interval override"
                    <input
                        type="number"
                        min="0"
                        step="0.1"
                        class="mic-input"
                        placeholder="global"
                        prop:value=move || min_interval_text.get()
                        on:input=move |event| min_interval_text.set(event_target_value(&event))
                    />
                </label>
                <label>"No-go zone IDs"
                    <input
                        type="text"
                        class="mic-input effector-zone-input"
                        placeholder="zone-a, zone-b"
                        prop:value=move || no_go_zone_text.get()
                        on:input=move |event| no_go_zone_text.set(event_target_value(&event))
                    />
                </label>
            </div>

            <div class="pipeline-save-row">
                <button class="btn-primary" disabled=move || loading.get() || saving.get() on:click=save_safety>
                    {move || if saving.get() { "Saving..." } else { "Save safety" }}
                </button>
                {move || error.get().map(|message| view! { <span class="daily-error">{message}</span> })}
                {move || notice.get().map(|message| view! { <span class="tone-badge ok">{message}</span> })}
            </div>
        </section>
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_form() -> RegisterForm {
        RegisterForm {
            id: "cam-1".into(),
            host: "192.168.1.50".into(),
            port: "80".into(),
            username: "admin".into(),
            password: "secret".into(),
            rtsp_url: String::new(),
            lat: "44.9".into(),
            lon: "-93.2".into(),
            alt_m: "10".into(),
            yaw_deg: "90".into(),
            pitch_deg: "0".into(),
        }
    }

    #[test]
    fn build_register_payload_requires_id() {
        let mut form = sample_form();
        form.id = "  ".into();
        assert!(build_register_payload(&form).is_err());
    }

    #[test]
    fn build_register_payload_requires_valid_lat_lon() {
        let mut form = sample_form();
        form.lat = "not-a-number".into();
        assert!(build_register_payload(&form).is_err());
    }

    #[test]
    fn build_register_payload_shapes_expected_json() {
        let payload = build_register_payload(&sample_form()).unwrap();
        assert_eq!(payload["id"], "cam-1");
        assert_eq!(payload["effector_type"], "camera_ptz");
        assert_eq!(payload["position_geo"]["lat"], 44.9);
        assert_eq!(payload["orientation"]["yaw_deg"], 90.0);
        assert_eq!(payload["transport"]["host"], "192.168.1.50");
        assert!(payload["transport"].get("rtsp_url").is_none());
    }

    #[test]
    fn build_register_payload_includes_rtsp_url_when_present() {
        let mut form = sample_form();
        form.rtsp_url = "rtsp://192.168.1.50/stream1".into();
        let payload = build_register_payload(&form).unwrap();
        assert_eq!(
            payload["transport"]["rtsp_url"],
            "rtsp://192.168.1.50/stream1"
        );
    }
}
