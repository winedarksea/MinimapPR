//! Node detail view — capability-driven sections for a single node. Fetches the
//! enriched detail (`GET /api/v1/nodes/{id}`) directly so deep links resolve even
//! before the live poll populates the shared `nodes` signal.

use super::audio_pipeline::AudioPipelineSection;
use super::{capability_tone, patch_node};
use crate::api::{arm_ptz_node, disarm_ptz_node};
use crate::panels::ptz_view::PtzLiveView;
use gloo_net::http::Request;
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::Deserialize;

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
struct Orientation {
    #[serde(default)]
    yaw_deg: f64,
    #[serde(default)]
    pitch_deg: f64,
    #[serde(default)]
    roll_deg: f64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
struct PtzStatus {
    #[serde(default)]
    state: String,
    #[serde(default)]
    armed: bool,
    pan_deg: Option<f64>,
    tilt_deg: Option<f64>,
    zoom: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
struct NodeDetail {
    #[serde(alias = "node_id")]
    id: String,
    #[serde(default)]
    node_type: Option<String>,
    #[serde(default, alias = "health_status")]
    health: Option<String>,
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    position_m: Option<Vec<f64>>,
    #[serde(default)]
    reported_position_m: Option<Vec<f64>>,
    #[serde(default)]
    orientation: Option<Orientation>,
    #[serde(default)]
    reported_orientation: Option<Orientation>,
    #[serde(default)]
    overrides: serde_json::Value,
    #[serde(default)]
    permissions: serde_json::Value,
    #[serde(default)]
    firmware_version: Option<String>,
    #[serde(default)]
    bit_failure_codes: Option<Vec<String>>,
    #[serde(default)]
    ptz_status: Option<PtzStatus>,
    #[serde(default)]
    last_seen_ns: Option<i64>,
}

impl NodeDetail {
    fn has_capability(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|item| item == capability)
    }
    fn override_pinned(&self, key: &str) -> bool {
        self.overrides
            .as_object()
            .map(|map| map.contains_key(key))
            .unwrap_or(false)
    }
}

async fn fetch_node(node_id: &str) -> Result<NodeDetail, String> {
    let encoded = js_sys::encode_uri_component(node_id)
        .as_string()
        .unwrap_or_default();
    let resp = Request::get(&format!("/api/v1/nodes/{encoded}"))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.json::<NodeDetail>().await.map_err(|e| e.to_string())
}

#[component]
pub fn NodeDetailView() -> impl IntoView {
    let params = leptos_router::hooks::use_params_map();
    let node_id = move || params.with(|params| params.get("id").unwrap_or_default());

    let detail: RwSignal<Option<NodeDetail>> = RwSignal::new(None);
    let load_error: RwSignal<Option<String>> = RwSignal::new(None);

    let reload = move || {
        let id = node_id();
        spawn_local(async move {
            match fetch_node(&id).await {
                Ok(node) => {
                    detail.set(Some(node));
                    load_error.set(None);
                }
                Err(message) => load_error.set(Some(message)),
            }
        });
    };

    Effect::new(move |_| {
        // Re-run whenever the route param changes.
        let _ = node_id();
        reload();
    });

    view! {
        <div class="nodes-page node-detail">
            <a class="node-detail-back" href="/settings/nodes">"← All nodes"</a>
            {move || match detail.get() {
                None => view! {
                    <div class="node-empty">
                        {move || load_error.get()
                            .map(|message| view! { <p class="daily-error">{message}</p> }.into_any())
                            .unwrap_or_else(|| view! { <p class="muted">"Loading…"</p> }.into_any())}
                    </div>
                }.into_any(),
                Some(node) => view! { <DetailBody node=node reload=Callback::new(move |_| reload()) /> }.into_any(),
            }}
        </div>
    }
}

#[component]
fn DetailBody(node: NodeDetail, reload: Callback<()>) -> impl IntoView {
    let health = node.health.clone().unwrap_or_else(|| "unknown".to_string());
    let node_type = node.node_type.clone().unwrap_or_else(|| "unknown".to_string());
    let capabilities = node.capabilities.clone();
    let firmware = node.firmware_version.clone();
    let bit_codes = node.bit_failure_codes.clone().unwrap_or_default();
    let node_id = node.id.clone();

    let is_audio = node.has_capability("audio") || node.capabilities.is_empty();
    let is_ptz = node.has_capability("ptz_camera");

    view! {
        <header class="node-detail-header">
            <h2>{node.id.clone()}</h2>
            <span class=format!("health-chip {health}")>{health.clone()}</span>
            <span class="node-type-tag">{node_type}</span>
        </header>

        <div class="node-detail-grid">
            // Info
            <section class="node-section-card">
                <h3 class="node-section-title">"Info"</h3>
                <div class="node-cap-chips">
                    {capabilities.iter().cloned().map(|capability| {
                        let tone = capability_tone(&capability);
                        view! { <span class=format!("cap-chip {tone}")>{capability}</span> }
                    }).collect_view()}
                </div>
                <dl class="node-kv">
                    {firmware.map(|version| view! { <div><dt>"Firmware"</dt><dd>{version}</dd></div> })}
                    {(!bit_codes.is_empty()).then(|| {
                        let codes = bit_codes.join(", ");
                        view! { <div><dt>"BIT failures"</dt><dd class="node-bit-fail">{codes}</dd></div> }
                    })}
                </dl>
            </section>

            // Location & orientation
            <LocationSection node=node.clone() reload=reload />

            // Audio pipeline
            {is_audio.then(|| view! {
                <section class="node-section-card node-section-wide">
                    <h3 class="node-section-title">"Audio Pipeline"</h3>
                    <AudioPipelineSection node_id=node_id.clone() />
                </section>
            })}

            // Camera / PTZ
            {is_ptz.then(|| view! {
                <PtzSection node=node.clone() reload=reload />
            })}

            // Safety (PTZ / actuating nodes)
            {is_ptz.then(|| view! {
                <SafetySection node_id=node_id.clone() />
            })}

            // Permissions (placeholder)
            <section class="node-section-card">
                <h3 class="node-section-title">"Permissions"</h3>
                {permission_rows(&node.permissions)}
            </section>
        </div>
    }
}

fn permission_rows(permissions: &serde_json::Value) -> impl IntoView {
    match permissions.as_object() {
        Some(map) if !map.is_empty() => view! {
            <dl class="node-kv">
                {map.iter().map(|(key, value)| {
                    let key = key.clone();
                    let value = value.to_string();
                    view! { <div><dt>{key}</dt><dd>{value}</dd></div> }
                }).collect_view()}
            </dl>
        }.into_any(),
        _ => view! { <p class="muted">"No permissions configured for this node."</p> }.into_any(),
    }
}

// ---------------------------------------------------------------------------
// Location & orientation — per-field (position / orientation) pinning.
// ---------------------------------------------------------------------------

#[component]
fn LocationSection(node: NodeDetail, reload: Callback<()>) -> impl IntoView {
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let node_id = node.id.clone();

    let reported = node.reported_position_m.clone().unwrap_or_default();
    let effective = node.position_m.clone().unwrap_or_default();
    let position_pinned = node.override_pinned("position_m") || node.override_pinned("position_geo");

    // Editable position fields, seeded from the effective value.
    let pos_x = RwSignal::new(effective.first().copied().unwrap_or(0.0).to_string());
    let pos_y = RwSignal::new(effective.get(1).copied().unwrap_or(0.0).to_string());
    let pos_z = RwSignal::new(effective.get(2).copied().unwrap_or(0.0).to_string());

    let reported_orient = node.reported_orientation.clone().unwrap_or_default();
    let effective_orient = node.orientation.clone().unwrap_or_default();
    let orient_pinned = node.override_pinned("orientation");
    let yaw = RwSignal::new(effective_orient.yaw_deg.to_string());
    let pitch = RwSignal::new(effective_orient.pitch_deg.to_string());
    let roll = RwSignal::new(effective_orient.roll_deg.to_string());

    let apply = {
        let node_id = node_id.clone();
        move |body: serde_json::Value| {
            let id = node_id.clone();
            error.set(None);
            spawn_local(async move {
                match patch_node(&id, body).await {
                    Ok(_) => reload.run(()),
                    Err(message) => error.set(Some(message)),
                }
            });
        }
    };

    let pin_position = {
        let apply = apply.clone();
        move |_| {
            let x = pos_x.get_untracked().trim().parse::<f64>().unwrap_or(0.0);
            let y = pos_y.get_untracked().trim().parse::<f64>().unwrap_or(0.0);
            let z = pos_z.get_untracked().trim().parse::<f64>().unwrap_or(0.0);
            apply(serde_json::json!({ "overrides": { "position_m": [x, y, z] } }));
        }
    };
    let unpin_position = {
        let apply = apply.clone();
        move |_| apply(serde_json::json!({ "clear_overrides": ["position_m", "position_geo"] }))
    };
    let pin_orientation = {
        let apply = apply.clone();
        move |_| {
            let y = yaw.get_untracked().trim().parse::<f64>().unwrap_or(0.0);
            let p = pitch.get_untracked().trim().parse::<f64>().unwrap_or(0.0);
            let r = roll.get_untracked().trim().parse::<f64>().unwrap_or(0.0);
            apply(serde_json::json!({
                "overrides": { "orientation": { "yaw_deg": y, "pitch_deg": p, "roll_deg": r } }
            }));
        }
    };
    let unpin_orientation = {
        let apply = apply.clone();
        move |_| apply(serde_json::json!({ "clear_overrides": ["orientation"] }))
    };

    let fmt_vec = |values: &[f64]| {
        if values.is_empty() {
            "—".to_string()
        } else {
            values
                .iter()
                .map(|value| format!("{value:.2}"))
                .collect::<Vec<_>>()
                .join(", ")
        }
    };

    view! {
        <section class="node-section-card">
            <h3 class="node-section-title">"Location & Orientation"</h3>

            <div class="node-field-block">
                <div class="node-field-head">
                    <span>"Position (x, y, z m)"</span>
                    <span class=if position_pinned { "pin-indicator is-pinned" } else { "pin-indicator" }>
                        {if position_pinned { "📌 pinned" } else { "reported" }}
                    </span>
                </div>
                <p class="muted node-reported-line">{format!("Reported: {}", fmt_vec(&reported))}</p>
                <div class="node-xyz">
                    <input type="number" step="0.1" prop:value=move || pos_x.get() on:input=move |ev| pos_x.set(event_target_value(&ev)) />
                    <input type="number" step="0.1" prop:value=move || pos_y.get() on:input=move |ev| pos_y.set(event_target_value(&ev)) />
                    <input type="number" step="0.1" prop:value=move || pos_z.get() on:input=move |ev| pos_z.set(event_target_value(&ev)) />
                </div>
                <div class="node-field-actions">
                    <button class="btn-sm btn-primary" on:click=pin_position>"Pin position"</button>
                    <button class="btn-sm" disabled=!position_pinned on:click=unpin_position>"Unpin"</button>
                </div>
            </div>

            <div class="node-field-block">
                <div class="node-field-head">
                    <span>"Orientation (yaw, pitch, roll °)"</span>
                    <span class=if orient_pinned { "pin-indicator is-pinned" } else { "pin-indicator" }>
                        {if orient_pinned { "📌 pinned" } else { "reported" }}
                    </span>
                </div>
                <p class="muted node-reported-line">
                    {format!("Reported: {:.1}, {:.1}, {:.1}", reported_orient.yaw_deg, reported_orient.pitch_deg, reported_orient.roll_deg)}
                </p>
                <div class="node-xyz">
                    <input type="number" step="1" prop:value=move || yaw.get() on:input=move |ev| yaw.set(event_target_value(&ev)) />
                    <input type="number" step="1" prop:value=move || pitch.get() on:input=move |ev| pitch.set(event_target_value(&ev)) />
                    <input type="number" step="1" prop:value=move || roll.get() on:input=move |ev| roll.set(event_target_value(&ev)) />
                </div>
                <div class="node-field-actions">
                    <button class="btn-sm btn-primary" on:click=pin_orientation>"Pin orientation"</button>
                    <button class="btn-sm" disabled=!orient_pinned on:click=unpin_orientation>"Unpin"</button>
                </div>
            </div>

            {move || error.get().map(|message| view! { <p class="daily-error">{message}</p> })}
        </section>
    }
}

// ---------------------------------------------------------------------------
// Camera / PTZ — arm/disarm + live snapshot view.
// ---------------------------------------------------------------------------

#[component]
fn PtzSection(node: NodeDetail, reload: Callback<()>) -> impl IntoView {
    let node_id = node.id.clone();
    let status = node.ptz_status.clone().unwrap_or_default();
    let armed = status.armed;
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let busy = RwSignal::new(false);

    let toggle_arm = {
        let node_id = node_id.clone();
        move |_| {
            let id = node_id.clone();
            busy.set(true);
            error.set(None);
            spawn_local(async move {
                let result = if armed {
                    disarm_ptz_node(&id).await
                } else {
                    arm_ptz_node(&id).await
                };
                match result {
                    Ok(()) => reload.run(()),
                    Err(message) => error.set(Some(message)),
                }
                busy.set(false);
            });
        }
    };

    let state_label = status.state.clone();
    let pan_tilt = match (status.pan_deg, status.tilt_deg) {
        (Some(pan), Some(tilt)) => format!("pan {pan:.0}° · tilt {tilt:.0}°"),
        _ => "position unknown".to_string(),
    };

    view! {
        <section class="node-section-card node-section-wide">
            <h3 class="node-section-title">"Camera / PTZ"</h3>
            <div class="node-ptz-bar">
                <span class=if armed { "tone-badge ok" } else { "tone-badge neutral" }>
                    {if armed { "Armed" } else { "Disarmed" }}
                </span>
                <span class="muted">{state_label}</span>
                <span class="muted">{pan_tilt}</span>
                <button class="btn-sm btn-primary" disabled=move || busy.get() on:click=toggle_arm>
                    {if armed { "Disarm" } else { "Arm" }}
                </button>
                {move || error.get().map(|message| view! { <span class="daily-error">{message}</span> })}
            </div>
            <div class="node-ptz-view">
                <PtzLiveView node_id=node_id.clone() track_id=None on_close=Callback::new(|_| ()) />
            </div>
        </section>
    }
}

// ---------------------------------------------------------------------------
// Safety form.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
struct SafetyConfig {
    #[serde(default)]
    require_arm_for_action: bool,
    #[serde(default)]
    min_action_interval_seconds: Option<f64>,
    #[serde(default)]
    no_go_zone_ids: Vec<String>,
}

#[component]
fn SafetySection(node_id: String) -> impl IntoView {
    let require_arm = RwSignal::new(true);
    let min_interval = RwSignal::new("0".to_string());
    // Preserve fields the form doesn't edit so a save doesn't wipe them.
    let no_go_zone_ids: RwSignal<Vec<String>> = RwSignal::new(Vec::new());
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let saved = RwSignal::new(false);

    {
        let node_id = node_id.clone();
        Effect::new(move |_| {
            let node_id = node_id.clone();
            spawn_local(async move {
                let encoded = js_sys::encode_uri_component(&node_id)
                    .as_string()
                    .unwrap_or_default();
                if let Ok(resp) = Request::get(&format!("/api/v1/nodes/{encoded}/safety")).send().await {
                    if let Ok(cfg) = resp.json::<SafetyConfig>().await {
                        require_arm.set(cfg.require_arm_for_action);
                        min_interval.set(cfg.min_action_interval_seconds.unwrap_or(0.0).to_string());
                        no_go_zone_ids.set(cfg.no_go_zone_ids);
                    }
                }
            });
        });
    }

    let save = {
        let node_id = node_id.clone();
        move |_| {
            let id = node_id.clone();
            error.set(None);
            saved.set(false);
            let body = serde_json::json!({
                "require_arm_for_action": require_arm.get_untracked(),
                "min_action_interval_seconds": min_interval.get_untracked().trim().parse::<f64>().unwrap_or(0.0),
                "no_go_zone_ids": no_go_zone_ids.get_untracked(),
            });
            spawn_local(async move {
                let encoded = js_sys::encode_uri_component(&id)
                    .as_string()
                    .unwrap_or_default();
                let result = Request::patch(&format!("/api/v1/nodes/{encoded}/safety"))
                    .header("Content-Type", "application/json")
                    .body(serde_json::to_string(&body).unwrap_or_default())
                    .map_err(|e| e.to_string());
                match result {
                    Ok(req) => match req.send().await {
                        Ok(resp) if resp.ok() => saved.set(true),
                        Ok(resp) => error.set(Some(format!("HTTP {}", resp.status()))),
                        Err(e) => error.set(Some(e.to_string())),
                    },
                    Err(e) => error.set(Some(e)),
                }
            });
        }
    };

    view! {
        <section class="node-section-card">
            <h3 class="node-section-title">"Safety"</h3>
            <label class="node-checkbox">
                <input type="checkbox" prop:checked=move || require_arm.get()
                    on:change=move |ev| require_arm.set(event_target_checked(&ev)) />
                "Require arm before actions"
            </label>
            <label class="node-inline-field">
                "Min action interval (s)"
                <input type="number" step="0.5" min="0" prop:value=move || min_interval.get()
                    on:input=move |ev| min_interval.set(event_target_value(&ev)) />
            </label>
            <div class="node-field-actions">
                <button class="btn-sm btn-primary" on:click=save>"Save safety"</button>
                {move || saved.get().then(|| view! { <span class="muted">"Saved"</span> })}
                {move || error.get().map(|message| view! { <span class="daily-error">{message}</span> })}
            </div>
        </section>
    }
}
