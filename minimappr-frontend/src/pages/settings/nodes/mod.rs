//! Unified Nodes settings surface: a node table + Add Node wizard (`NodesView`)
//! and a capability-driven per-node detail page (`NodeDetailView`).

mod audio_pipeline;
mod detail;
mod wizard;

pub use detail::NodeDetailView;

use crate::api::delete_node;
use crate::state::{AppState, NodeStatus};
use leptos::prelude::*;
use leptos::task::spawn_local;
use wizard::AddNodeWizard;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Map a capability value to a `cap-chip` tone modifier (see page-nodes.css).
pub(crate) fn capability_tone(capability: &str) -> &'static str {
    match capability {
        "audio" | "array_localization" => "cap-audio",
        "ptz_camera" => "cap-camera",
        "radar_24g" | "sdr" => "cap-rf",
        "seismic" => "cap-seismic",
        "speech" => "cap-speech",
        "relay" => "cap-relay",
        "gps" | "gps_optional" => "cap-gps",
        "temperature" | "humidity" | "environment" => "cap-env",
        _ => "cap-other",
    }
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

/// PATCH a node's operator-owned fields / overrides. Shared by the detail sections.
pub(crate) async fn patch_node(node_id: &str, body: serde_json::Value) -> Result<(), String> {
    let encoded = js_sys::encode_uri_component(node_id)
        .as_string()
        .unwrap_or_default();
    let resp = gloo_net::http::Request::patch(&format!("/api/v1/nodes/{encoded}"))
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&body).unwrap_or_default())
        .map_err(|error| error.to_string())?
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if resp.ok() {
        Ok(())
    } else {
        Err(resp.text().await.unwrap_or_default())
    }
}

// ---------------------------------------------------------------------------
// NodesView
// ---------------------------------------------------------------------------

#[component]
pub fn NodesView() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let nodes = state.nodes;
    let show_wizard = RwSignal::new(false);

    view! {
        <div class="nodes-page">
            <div class="nodes-toolbar">
                <div>
                    <h2>"Nodes"</h2>
                    <span class="muted">{move || format!("{} registered", nodes.get().len())}</span>
                </div>
                <button class="btn-primary" on:click=move |_| show_wizard.update(|value| *value = !*value)>
                    {move || if show_wizard.get() { "Close" } else { "＋ Add Node" }}
                </button>
            </div>

            {move || show_wizard.get().then(|| view! {
                <AddNodeWizard nodes=nodes on_close=Callback::new(move |_| show_wizard.set(false)) />
            })}

            {move || {
                if nodes.get().is_empty() {
                    view! {
                        <div class="node-empty">
                            <span class="material-symbols-rounded" aria-hidden="true">"sensors"</span>
                            <p>"No nodes yet."</p>
                            <p class="muted">"Add a camera, or point a sensor's firmware at this server's ingest endpoint."</p>
                        </div>
                    }.into_any()
                } else {
                    view! {
                        <div class="node-table-wrap">
                            <table class="node-table">
                                <thead>
                                    <tr>
                                        <th>"ID"</th>
                                        <th>"Type"</th>
                                        <th>"Capabilities"</th>
                                        <th>"Health"</th>
                                        <th></th>
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
                    }.into_any()
                }
            }}
        </div>
    }
}

#[component]
fn NodeRow(node: NodeStatus, nodes: RwSignal<Vec<NodeStatus>>) -> impl IntoView {
    let node_id = node.node_id.clone();
    let detail_href = format!("/settings/nodes/{node_id}");
    let node_type = node
        .node_type
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let health = node.health.clone();
    let pinned = is_pinned(&node);
    let capabilities = capability_list(&node);
    let delete_id = node_id.clone();

    let confirming = RwSignal::new(false);

    view! {
        <tr>
            <td>
                <a class="node-id-link" href=detail_href>{node.node_id.clone()}</a>
                {pinned.then(|| view! { <span class="pin-indicator is-pinned" title="Has operator overrides">"📌"</span> })}
            </td>
            <td><span class="node-type-tag">{node_type}</span></td>
            <td>
                <div class="node-cap-chips">
                    {capabilities.into_iter().map(|capability| {
                        let tone = capability_tone(&capability);
                        view! { <span class=format!("cap-chip {tone}")>{capability}</span> }
                    }).collect_view()}
                </div>
            </td>
            <td><span class=format!("health-chip {}", health)>{health.clone()}</span></td>
            <td>
                <a class="btn-sm" href=format!("/settings/nodes/{}", node.node_id)>"Configure"</a>
            </td>
            <td>
                {move || if confirming.get() {
                    let id = delete_id.clone();
                    let nodes_for_delete = nodes;
                    let on_delete = move |_| {
                        let id = id.clone();
                        spawn_local(async move {
                            if delete_node(&id).await.is_ok() {
                                nodes_for_delete.update(|items| items.retain(|node| node.node_id != id));
                            }
                        });
                    };
                    view! {
                        <span class="node-del-confirm">
                            <button class="btn-sm btn-danger" on:click=on_delete>"Confirm"</button>
                            <button class="btn-sm" on:click=move |_| confirming.set(false)>"Cancel"</button>
                        </span>
                    }.into_any()
                } else {
                    view! {
                        <button class="btn-sm" on:click=move |_| confirming.set(true)>"Delete"</button>
                    }.into_any()
                }}
            </td>
        </tr>
    }
}
