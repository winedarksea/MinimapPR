use crate::api::{delete_node, register_node};
use crate::state::{AppState, NodeStatus};
use gloo_net::http::Request;
use leptos::prelude::*;
use leptos::task::spawn_local;

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

fn capability_list(node: &NodeStatus) -> Vec<String> {
    node.capabilities.clone().unwrap_or_default()
}

fn is_pinned(node: &NodeStatus) -> bool {
    node.overrides
        .as_object()
        .map(|overrides| {
            overrides.contains_key("position_m")
                || overrides.contains_key("position_geo")
                || overrides.contains_key("orientation")
                || overrides.contains_key("capabilities")
                || overrides.contains_key("mobility")
        })
        .unwrap_or(false)
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

async fn patch_node(node_id: &str, body: serde_json::Value) -> Result<NodeStatus, String> {
    let encoded = js_sys::encode_uri_component(node_id)
        .as_string()
        .unwrap_or_default();
    let url = format!("/api/v1/nodes/{encoded}");
    let resp = Request::patch(&url)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&body).unwrap_or_default())
        .map_err(|error| error.to_string())?
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if resp.ok() {
        resp.json::<NodeStatus>().await.map_err(|error| error.to_string())
    } else {
        Err(resp.text().await.unwrap_or_default())
    }
}

#[component]
pub fn NodesView() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let nodes = state.nodes;
    let show_wizard = RwSignal::new(false);

    view! {
        <div class="page-stub">
            <div style="display:flex;align-items:center;justify-content:space-between;gap:1rem">
                <h2 style="margin:0">"Nodes"</h2>
                <button class="btn-primary" on:click=move |_| show_wizard.update(|value| *value = !*value)>
                    {move || if show_wizard.get() { "Close" } else { "Add Node" }}
                </button>
            </div>

            {move || show_wizard.get().then(|| view! { <AddNodeWizard nodes=nodes /> })}

            <div class="diag-card" style="overflow:auto">
                <table class="compact-table">
                    <thead>
                        <tr>
                            <th>"ID"</th>
                            <th>"Type"</th>
                            <th>"Capabilities"</th>
                            <th>"Health"</th>
                            <th>"Last seen"</th>
                            <th>"Pinned"</th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody>
                        {move || nodes.get().into_iter().map(|node| view! {
                            <NodeRow node=node nodes=nodes />
                        }).collect_view()}
                    </tbody>
                </table>
            </div>
        </div>
    }
}

#[component]
fn NodeRow(node: NodeStatus, nodes: RwSignal<Vec<NodeStatus>>) -> impl IntoView {
    let node_id = node.node_id.clone();
    let detail_href = format!("/settings/nodes/{node_id}");
    let node_type = node.node_type.clone().unwrap_or_else(|| "unknown".to_string());
    let health = node.health.clone();
    let last_seen = node
        .last_seen_seconds_ago
        .map(|seconds| format!("{seconds:.1}s"))
        .unwrap_or_else(|| "never".to_string());
    let pinned = if is_pinned(&node) { "yes" } else { "no" };
    let on_delete = move |_| {
        let id = node_id.clone();
        let nodes_signal = nodes;
        spawn_local(async move {
            if delete_node(&id).await.is_ok() {
                nodes_signal.update(|items| items.retain(|node| node.node_id != id));
            }
        });
    };
    view! {
        <tr>
            <td><a href=detail_href>{node.node_id.clone()}</a></td>
            <td>{node_type}</td>
            <td>
                {capability_list(&node).into_iter().map(|capability| view! {
                    <span class="tone-badge tone-blue" style="margin-right:0.25rem">{capability}</span>
                }).collect_view()}
            </td>
            <td><span class=format!("health-chip {}", health)>{health.clone()}</span></td>
            <td>{last_seen}</td>
            <td>{pinned}</td>
            <td>
                <button class="btn-sm" on:click=on_delete>"Delete"</button>
            </td>
        </tr>
    }
}

#[component]
fn AddNodeWizard(nodes: RwSignal<Vec<NodeStatus>>) -> impl IntoView {
    let active_branch = RwSignal::new("audio".to_string());
    let camera_form = RwSignal::new(CameraForm::default());
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let submitting = RwSignal::new(false);

    let submit_camera = move |_| {
        error.set(None);
        let payload = match build_camera_payload(&camera_form.get_untracked()) {
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
                        if !items.iter().any(|existing| existing.node_id == node.node_id) {
                            items.push(node);
                        }
                    });
                    camera_form.set(CameraForm::default());
                }
                Err(message) => error.set(Some(message)),
            }
            submitting.set(false);
        });
    };

    view! {
        <div class="diag-card">
            <div class="pipeline-section-label">"Add Node"</div>
            <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1rem">
                <button class="btn-sm" on:click=move |_| active_branch.set("audio".to_string())>"Audio / Sensor"</button>
                <button class="btn-sm" on:click=move |_| active_branch.set("camera".to_string())>"Camera / PTZ"</button>
                <button class="btn-sm" disabled=true>"Scan & Pair"</button>
            </div>
            {move || match active_branch.get().as_str() {
                "camera" => view! {
                    <div>
                        <div class="compact-form-grid">
                            <label>"Camera ID"<input type="text" prop:value=move || camera_form.get().id on:input=move |ev| camera_form.update(|f| f.id = event_target_value(&ev)) /></label>
                            <label>"Host / IP"<input type="text" prop:value=move || camera_form.get().host on:input=move |ev| camera_form.update(|f| f.host = event_target_value(&ev)) /></label>
                            <label>"Port"<input type="number" placeholder="80" prop:value=move || camera_form.get().port on:input=move |ev| camera_form.update(|f| f.port = event_target_value(&ev)) /></label>
                            <label>"Username"<input type="text" prop:value=move || camera_form.get().username on:input=move |ev| camera_form.update(|f| f.username = event_target_value(&ev)) /></label>
                            <label>"Password"<input type="password" prop:value=move || camera_form.get().password on:input=move |ev| camera_form.update(|f| f.password = event_target_value(&ev)) /></label>
                            <label>"RTSP URL"<input type="text" prop:value=move || camera_form.get().rtsp_url on:input=move |ev| camera_form.update(|f| f.rtsp_url = event_target_value(&ev)) /></label>
                            <label>"Latitude"<input type="number" step="0.000001" prop:value=move || camera_form.get().lat on:input=move |ev| camera_form.update(|f| f.lat = event_target_value(&ev)) /></label>
                            <label>"Longitude"<input type="number" step="0.000001" prop:value=move || camera_form.get().lon on:input=move |ev| camera_form.update(|f| f.lon = event_target_value(&ev)) /></label>
                            <label>"Altitude"<input type="number" step="0.1" prop:value=move || camera_form.get().alt_m on:input=move |ev| camera_form.update(|f| f.alt_m = event_target_value(&ev)) /></label>
                            <label>"Home yaw"<input type="number" step="1" prop:value=move || camera_form.get().yaw_deg on:input=move |ev| camera_form.update(|f| f.yaw_deg = event_target_value(&ev)) /></label>
                            <label>"Home pitch"<input type="number" step="1" prop:value=move || camera_form.get().pitch_deg on:input=move |ev| camera_form.update(|f| f.pitch_deg = event_target_value(&ev)) /></label>
                        </div>
                        <div class="pipeline-save-row">
                            <button class="btn-primary" disabled=move || submitting.get() on:click=submit_camera>
                                {move || if submitting.get() { "Registering..." } else { "Register Camera" }}
                            </button>
                            {move || error.get().map(|message| view! { <span class="daily-error">{message}</span> })}
                        </div>
                    </div>
                }.into_any(),
                _ => view! {
                    <div class="empty-state">
                        <p>"Point firmware at this server's ingest endpoint and wait for its first report."</p>
                        <code>"/api/v1/ingest/binary"</code>
                        <p>{move || format!("{} node(s) reporting", nodes.get().len())}</p>
                    </div>
                }.into_any(),
            }}
        </div>
    }
}

#[component]
pub fn NodeDetailView() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let params = leptos_router::hooks::use_params_map();
    let node_id = move || params.with(|params| params.get("id").unwrap_or_default());
    let nodes = state.nodes;
    let error: RwSignal<Option<String>> = RwSignal::new(None);

    let pin_position = move |_| {
        let id = node_id();
        let node = nodes.with(|items| items.iter().find(|node| node.node_id == id).cloned());
        if let Some(node) = node {
            if let Some(position_m) = node.position_m.clone() {
                spawn_local(async move {
                    let body = serde_json::json!({"overrides": {"position_m": position_m}});
                    match patch_node(&id, body).await {
                        Ok(updated) => nodes.update(|items| {
                            if let Some(existing) = items.iter_mut().find(|item| item.node_id == updated.node_id) {
                                *existing = updated;
                            }
                        }),
                        Err(message) => error.set(Some(message)),
                    }
                });
            }
        }
    };
    let unpin_position = move |_| {
        let id = node_id();
        spawn_local(async move {
            let body = serde_json::json!({"clear_overrides": ["position_m", "position_geo"]});
            match patch_node(&id, body).await {
                Ok(updated) => nodes.update(|items| {
                    if let Some(existing) = items.iter_mut().find(|item| item.node_id == updated.node_id) {
                        *existing = updated;
                    }
                }),
                Err(message) => error.set(Some(message)),
            }
        });
    };

    view! {
        <div class="page-stub">
            {move || {
                let id = node_id();
                let node = nodes.with(|items| items.iter().find(|node| node.node_id == id).cloned());
                match node {
                    Some(node) => view! {
                        <div>
                            <h2 style="margin:0 0 1rem">{node.node_id.clone()}</h2>
                            <div class="diag-card">
                                <div class="pipeline-section-label">"Info"</div>
                                <p>"Type: " {node.node_type.clone().unwrap_or_else(|| "unknown".to_string())}</p>
                                <p>"Health: " {node.health.clone()}</p>
                                <div>{capability_list(&node).into_iter().map(|capability| view! {
                                    <span class="tone-badge tone-blue" style="margin-right:0.25rem">{capability}</span>
                                }).collect_view()}</div>
                            </div>
                            <div class="diag-card">
                                <div class="pipeline-section-label">"Location & Orientation"</div>
                                <p>"Effective position: " {format!("{:?}", node.position_m.clone().unwrap_or_default())}</p>
                                <p>"Reported position: " {format!("{:?}", node.reported_position_m.clone().unwrap_or_default())}</p>
                                <div class="pipeline-save-row">
                                    <button class="btn-sm" on:click=pin_position>"Pin current position"</button>
                                    <button class="btn-sm" on:click=unpin_position>"Unpin position"</button>
                                    {move || error.get().map(|message| view! { <span class="daily-error">{message}</span> })}
                                </div>
                            </div>
                            {node.has_capability("ptz_camera").then(|| view! {
                                <div class="diag-card">
                                    <div class="pipeline-section-label">"Camera / PTZ"</div>
                                    <p>"State: " {node.ptz_status.as_ref().map(|status| status.state.clone()).unwrap_or_else(|| "unknown".to_string())}</p>
                                    <img style="max-width:360px;width:100%" src=format!("/api/v1/nodes/{}/effector/snapshot.jpg", node.node_id) />
                                </div>
                            })}
                            <div class="diag-card">
                                <div class="pipeline-section-label">"Permissions"</div>
                                <pre>{serde_json::to_string_pretty(&node.permissions).unwrap_or_else(|_| "{}".to_string())}</pre>
                            </div>
                        </div>
                    }.into_any(),
                    None => view! { <div class="empty-state"><p>"Node not found."</p></div> }.into_any(),
                }
            }}
        </div>
    }
}
