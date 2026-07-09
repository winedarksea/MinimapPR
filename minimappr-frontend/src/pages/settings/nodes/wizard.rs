//! Add Node wizard: three branches — Audio/Sensor (ingest + live first-report
//! watch), Camera/PTZ (ONVIF connect form → operator registration), and a
//! disabled Scan & Pair placeholder.

use crate::api::register_node;
use crate::state::NodeStatus;
use leptos::prelude::*;
use leptos::task::spawn_local;
use std::collections::HashSet;

#[derive(Clone, Debug, Default)]
struct CameraForm {
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

fn build_camera_payload(form: &CameraForm) -> Result<serde_json::Value, String> {
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
        "node_type": "point",
        "position_geo": { "lat": lat, "lon": lon, "alt_m": parse_or(&form.alt_m, 0.0) },
        "orientation": {
            "yaw_deg": parse_or(&form.yaw_deg, 0.0),
            "pitch_deg": parse_or(&form.pitch_deg, 0.0),
            "roll_deg": 0.0,
        },
        "capabilities": ["ptz_camera"],
        "transport": transport,
    }))
}

#[component]
pub fn AddNodeWizard(nodes: RwSignal<Vec<NodeStatus>>, on_close: Callback<()>) -> impl IntoView {
    let active_branch = RwSignal::new("audio".to_string());

    view! {
        <div class="node-wizard">
            <div class="node-wizard-cards">
                <WizardCard
                    branch="audio" active_branch=active_branch
                    icon="sensors" title="Audio / Sensor"
                    blurb="Self-registering node (mic array, environment). Point firmware at ingest." />
                <WizardCard
                    branch="camera" active_branch=active_branch
                    icon="videocam" title="Camera / PTZ"
                    blurb="Operator-registered ONVIF PTZ camera. Enter connection details." />
                <div class="node-wizard-card is-disabled" aria-disabled="true">
                    <span class="material-symbols-rounded node-wizard-card-icon" aria-hidden="true">"radar"</span>
                    <div class="node-wizard-card-title">"Scan & Pair"</div>
                    <div class="node-wizard-card-blurb">"Auto-discover nearby nodes."</div>
                    <span class="tone-badge neutral">"Coming soon"</span>
                </div>
            </div>

            <div class="node-wizard-body">
                {move || match active_branch.get().as_str() {
                    "camera" => view! { <CameraBranch nodes=nodes on_close=on_close /> }.into_any(),
                    _ => view! { <AudioBranch nodes=nodes /> }.into_any(),
                }}
            </div>
        </div>
    }
}

#[component]
fn WizardCard(
    branch: &'static str,
    active_branch: RwSignal<String>,
    icon: &'static str,
    title: &'static str,
    blurb: &'static str,
) -> impl IntoView {
    let is_active = move || active_branch.get() == branch;
    view! {
        <button
            class="node-wizard-card"
            class:is-active=is_active
            on:click=move |_| active_branch.set(branch.to_string())
        >
            <span class="material-symbols-rounded node-wizard-card-icon" aria-hidden="true">{icon}</span>
            <div class="node-wizard-card-title">{title}</div>
            <div class="node-wizard-card-blurb">{blurb}</div>
        </button>
    }
}

#[component]
fn AudioBranch(nodes: RwSignal<Vec<NodeStatus>>) -> impl IntoView {
    // Snapshot the node ids already known when this branch mounts, so we can
    // surface any node that reports in *after* the operator started waiting.
    let baseline: HashSet<String> = nodes
        .get_untracked()
        .into_iter()
        .map(|node| node.node_id)
        .collect();

    let new_arrivals = move || {
        nodes
            .get()
            .into_iter()
            .filter(|node| !baseline.contains(&node.node_id))
            .collect::<Vec<_>>()
    };

    view! {
        <div class="node-branch">
            <ol class="node-ingest-steps">
                <li>"Flash / configure the node's firmware with this server's address."</li>
                <li>
                    "It will POST audio frames to the binary ingest endpoint:"
                    <code class="node-ingest-endpoint">"POST /api/v1/ingest/binary"</code>
                </li>
                <li>"The node self-registers on its first frame and appears below."</li>
            </ol>

            <div class="node-waiting">
                {move || {
                    let arrivals = new_arrivals();
                    if arrivals.is_empty() {
                        view! {
                            <div class="node-waiting-idle">
                                <span class="node-waiting-spinner" aria-hidden="true"></span>
                                <span>"Waiting for first report…"</span>
                            </div>
                        }.into_any()
                    } else {
                        view! {
                            <div>
                                <div class="node-waiting-ok">
                                    <span class="material-symbols-rounded" aria-hidden="true">"check_circle"</span>
                                    {format!("{} new node(s) reporting", arrivals.len())}
                                </div>
                                <ul class="node-waiting-list">
                                    {arrivals.into_iter().map(|node| {
                                        let last_seen = node
                                            .last_seen_seconds_ago
                                            .map(|seconds| format!("{seconds:.1}s ago"))
                                            .unwrap_or_else(|| "just now".to_string());
                                        let rms = node
                                            .rms_history
                                            .as_ref()
                                            .and_then(|history| history.last().copied())
                                            .map(|value| format!("RMS {value:.3}"))
                                            .unwrap_or_default();
                                        let href = format!("/settings/nodes/{}", node.node_id);
                                        view! {
                                            <li>
                                                <a href=href>{node.node_id.clone()}</a>
                                                <span class="muted">{last_seen}</span>
                                                {(!rms.is_empty()).then(|| view! { <span class="muted">{rms}</span> })}
                                            </li>
                                        }
                                    }).collect_view()}
                                </ul>
                            </div>
                        }.into_any()
                    }
                }}
            </div>
        </div>
    }
}

#[component]
fn CameraBranch(nodes: RwSignal<Vec<NodeStatus>>, on_close: Callback<()>) -> impl IntoView {
    let form = RwSignal::new(CameraForm::default());
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let submitting = RwSignal::new(false);

    let submit = move |_| {
        error.set(None);
        let payload = match build_camera_payload(&form.get_untracked()) {
            Ok(payload) => payload,
            Err(message) => {
                error.set(Some(message));
                return;
            }
        };
        submitting.set(true);
        spawn_local(async move {
            match register_node(payload).await {
                Ok(node) => {
                    nodes.update(|items| {
                        if !items
                            .iter()
                            .any(|existing| existing.node_id == node.node_id)
                        {
                            items.push(node);
                        }
                    });
                    form.set(CameraForm::default());
                    on_close.run(());
                }
                Err(message) => error.set(Some(message)),
            }
            submitting.set(false);
        });
    };

    view! {
        <div class="node-branch">
            <div class="compact-form-grid">
                <label>"Camera ID"<input type="text" prop:value=move || form.get().id on:input=move |ev| form.update(|f| f.id = event_target_value(&ev)) /></label>
                <label>"Host / IP"<input type="text" prop:value=move || form.get().host on:input=move |ev| form.update(|f| f.host = event_target_value(&ev)) /></label>
                <label>"Port"<input type="number" placeholder="80" prop:value=move || form.get().port on:input=move |ev| form.update(|f| f.port = event_target_value(&ev)) /></label>
                <label>"Username"<input type="text" prop:value=move || form.get().username on:input=move |ev| form.update(|f| f.username = event_target_value(&ev)) /></label>
                <label>"Password"<input type="password" prop:value=move || form.get().password on:input=move |ev| form.update(|f| f.password = event_target_value(&ev)) /></label>
                <label>"RTSP URL"<input type="text" prop:value=move || form.get().rtsp_url on:input=move |ev| form.update(|f| f.rtsp_url = event_target_value(&ev)) /></label>
                <label>"Latitude"<input type="number" step="0.000001" prop:value=move || form.get().lat on:input=move |ev| form.update(|f| f.lat = event_target_value(&ev)) /></label>
                <label>"Longitude"<input type="number" step="0.000001" prop:value=move || form.get().lon on:input=move |ev| form.update(|f| f.lon = event_target_value(&ev)) /></label>
                <label>"Altitude"<input type="number" step="0.1" prop:value=move || form.get().alt_m on:input=move |ev| form.update(|f| f.alt_m = event_target_value(&ev)) /></label>
                <label>"Home yaw"<input type="number" step="1" prop:value=move || form.get().yaw_deg on:input=move |ev| form.update(|f| f.yaw_deg = event_target_value(&ev)) /></label>
                <label>"Home pitch"<input type="number" step="1" prop:value=move || form.get().pitch_deg on:input=move |ev| form.update(|f| f.pitch_deg = event_target_value(&ev)) /></label>
            </div>
            <div class="node-form-actions">
                <button class="btn-primary" disabled=move || submitting.get() on:click=submit>
                    {move || if submitting.get() { "Registering…" } else { "Register Camera" }}
                </button>
                {move || error.get().map(|message| view! { <span class="daily-error">{message}</span> })}
            </div>
        </div>
    }
}
