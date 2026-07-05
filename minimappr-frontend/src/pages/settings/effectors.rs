use crate::api::{delete_effector, register_effector};
use crate::state::AppState;
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
                        <ul class="compact-list">
                            {list.into_iter().map(|effector| {
                                let delete_id = effector.id.clone();
                                let state_label = effector.status.as_ref()
                                    .map(|s| s.state.clone())
                                    .unwrap_or_else(|| "offline".to_string());
                                let chip_class = match state_label.as_str() {
                                    "idle" | "slewing" | "streaming" => "health-chip online",
                                    "error" => "health-chip offline",
                                    _ => "health-chip degraded",
                                };
                                view! {
                                    <li class="compact-row">
                                        <span class=chip_class>{state_label}</span>
                                        <span class="row-label">{effector.id.clone()}</span>
                                        <button
                                            class="btn-sm"
                                            title="Remove camera"
                                            on:click=move |_| {
                                                let id = delete_id.clone();
                                                spawn_local(async move {
                                                    if delete_effector(&id).await.is_ok() {
                                                        effectors.update(|list| list.retain(|e| e.id != id));
                                                    }
                                                });
                                            }
                                        >
                                            "Remove"
                                        </button>
                                    </li>
                                }
                            }).collect_view()}
                        </ul>
                    }.into_any()
                }
            }}

            {move || if show_form.get() {
                view! {
                    <div class="diag-card" style="max-width:32rem">
                        <div class="pipeline-section-label">"Add camera"</div>
                        <div class="settings-form-grid">
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
        assert_eq!(payload["transport"]["rtsp_url"], "rtsp://192.168.1.50/stream1");
    }
}
