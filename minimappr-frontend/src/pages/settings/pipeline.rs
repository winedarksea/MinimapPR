use futures::StreamExt;
use gloo_net::http::Request;
use gloo_timers::future::IntervalStream;
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// API response types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct PipelineStage {
    #[serde(default)] name: String,
    #[serde(default)] count_in: u64,
    #[serde(default)] count_out: u64,
    #[serde(default)] drops: u64,
    #[serde(default)] queue_depth: u64,
    #[serde(default)] queue_max: u64,
    #[serde(default)] lag_s: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct MicData {
    #[serde(default)] index: u32,
    #[serde(default)] label: String,
    #[serde(default)] gain_db: f64,
    #[serde(default)] hp_hz: f64,
    #[serde(default)] lp_hz: f64,
    #[serde(default)] smoothing: String,
    #[serde(default)] rms_recent: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Default, PartialEq)]
struct NodeAudioOverride {
    #[serde(default)] mic_gains_db: Option<Vec<f64>>,
    #[serde(default)] hp_hz: Option<f64>,
    #[serde(default)] lp_hz: Option<f64>,
    #[serde(default)] smoothing: Option<String>,
    #[serde(default)] stages: Option<Vec<serde_json::Value>>,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct PipelineNode {
    #[serde(default)] node_id: String,
    #[serde(default)] node_type: String,
    #[serde(default)] mics: Vec<MicData>,
    #[serde(default)] stages: Vec<PipelineStage>,
    #[serde(default)] audio_override: Option<NodeAudioOverride>,
    #[serde(default)] frame_gaps: u64,
    #[serde(default)] zero_padded_degraded: u64,
    #[serde(default)] last_frame_ns: Option<i64>,
    #[serde(default)] sample_rate_hz: Option<u32>,
    #[serde(default)] audio_status: String,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct PipelineNodesResp {
    #[serde(default)] active_pipeline: String,
    #[serde(default)] nodes: Vec<PipelineNode>,
    #[serde(default)] pipeline_seconds_behind_realtime: Option<f64>,
}

// ---------------------------------------------------------------------------
// Local edit state
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct NodeEdit {
    mic_gains_db: Vec<f64>,
    hp_hz: f64,
    lp_hz: f64,
    smoothing: String,
    stage_mode: bool,
    stage_json: String,
    dirty: bool,
}

impl NodeEdit {
    fn from_node(node: &PipelineNode) -> Self {
        let first = node.mics.first();
        let NodeAudioOverride {
            mic_gains_db: override_gains,
            hp_hz: override_hp_hz,
            lp_hz: override_lp_hz,
            smoothing: override_smoothing,
            stages: override_stages,
        } = node.audio_override.clone().unwrap_or_default();
        let mut mic_gains_db = if node.mics.is_empty() {
            vec![0.0]
        } else {
            node.mics.iter().map(|mic| mic.gain_db).collect()
        };
        if let Some(override_gains) = override_gains {
            if override_gains.len() > mic_gains_db.len() {
                mic_gains_db.resize(override_gains.len(), 0.0);
            }
            for (index, gain_db) in override_gains.into_iter().enumerate() {
                mic_gains_db[index] = gain_db;
            }
        }
        let stage_mode = override_stages.as_ref().map(|stages| !stages.is_empty()).unwrap_or(false);
        NodeEdit {
            mic_gains_db,
            hp_hz: override_hp_hz.unwrap_or_else(|| first.map(|m| m.hp_hz).unwrap_or(0.0)),
            lp_hz: override_lp_hz.unwrap_or_else(|| first.map(|m| m.lp_hz).unwrap_or(0.0)),
            smoothing: override_smoothing.unwrap_or_else(|| {
                first.map(|m| m.smoothing.clone()).unwrap_or_else(|| "off".into())
            }),
            stage_mode,
            stage_json: override_stages
                .map(|stages| format_stage_json(&stages))
                .unwrap_or_else(|| "[]".into()),
            dirty: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fmt_lag(lag_s: Option<f64>) -> String {
    match lag_s {
        Some(s) if s >= 60.0 => format!("{:.1}min", s / 60.0),
        Some(s) => format!("{s:.2}s"),
        None => "—".into(),
    }
}

fn stage_chip_class(lag_s: Option<f64>, drops: u64) -> &'static str {
    if drops > 0 { return "pipeline-stage-chip warn"; }
    match lag_s {
        Some(s) if s >= 2.0 => "pipeline-stage-chip danger",
        Some(s) if s >= 0.5 => "pipeline-stage-chip warn",
        _ => "pipeline-stage-chip ok",
    }
}

fn fmt_count(n: u64) -> String {
    if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1_000_000.0) }
    else if n >= 1_000 { format!("{:.1}k", n as f64 / 1_000.0) }
    else { n.to_string() }
}

fn sparkline_path(rms: &[f64], width: f64, height: f64) -> String {
    if rms.len() < 2 { return String::new(); }
    let max = rms.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(1e-9);
    let n = rms.len();
    let pts: Vec<String> = rms.iter().enumerate().map(|(i, &v)| {
        let x = i as f64 / (n - 1).max(1) as f64 * width;
        let y = height - (v / max) * height;
        format!("{x:.1},{y:.1}")
    }).collect();
    format!("M {}", pts.join(" L "))
}

fn audio_status_chip(status: &str) -> &'static str {
    match status {
        "recent" => "health-chip online",
        "stale"  => "health-chip degraded",
        _        => "health-chip offline",
    }
}

fn format_stage_json(stages: &[serde_json::Value]) -> String {
    serde_json::to_string_pretty(stages).unwrap_or_else(|_| "[]".into())
}

fn default_stage_json() -> String {
    format_stage_json(&[serde_json::json!({"type": "gain", "db": 0.0})])
}

fn parse_stage_json(stage_json: &str) -> Result<Vec<serde_json::Value>, String> {
    let trimmed = stage_json.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    let parsed: serde_json::Value = serde_json::from_str(trimmed)
        .map_err(|error| format!("Stage JSON parse error: {error}"))?;
    match parsed {
        serde_json::Value::Array(stages) => Ok(stages),
        _ => Err("Stage JSON must be a JSON array".into()),
    }
}

fn build_audio_override_body(edit: &NodeEdit) -> Result<serde_json::Value, String> {
    if edit.stage_mode {
        return Ok(serde_json::json!({
            "stages": parse_stage_json(&edit.stage_json)?,
        }));
    }

    Ok(serde_json::json!({
        "mic_gains_db": edit.mic_gains_db,
        "hp_hz": edit.hp_hz,
        "lp_hz": edit.lp_hz,
        "smoothing": edit.smoothing,
        "stages": [],
    }))
}

fn age_text(last_frame_ns: Option<i64>) -> Option<String> {
    let ns = last_frame_ns?;
    let now_ns = (js_sys::Date::now() * 1_000_000.0) as i64;
    let age_s = ((now_ns - ns) as f64 / 1_000_000_000.0).max(0.0);
    Some(if age_s < 60.0 { format!("{age_s:.0}s ago") }
         else if age_s < 3600.0 { format!("{:.0}m ago", age_s / 60.0) }
         else { format!("{:.1}h ago", age_s / 3600.0) })
}

// ---------------------------------------------------------------------------
// PipelineView
// ---------------------------------------------------------------------------

#[component]
pub fn PipelineView() -> impl IntoView {
    let data: RwSignal<Option<PipelineNodesResp>> = RwSignal::new(None);
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let edits: RwSignal<HashMap<String, NodeEdit>> = RwSignal::new(HashMap::new());

    let fetch = move || {
        spawn_local(async move {
            match Request::get("/api/v1/pipeline/nodes").send().await {
                Ok(resp) if resp.ok() => match resp.json::<PipelineNodesResp>().await {
                    Ok(r) => {
                        edits.update(|map| {
                            for node in &r.nodes {
                                let e = map.entry(node.node_id.clone())
                                    .or_insert_with(|| NodeEdit::from_node(node));
                                if !e.dirty { *e = NodeEdit::from_node(node); }
                            }
                        });
                        data.set(Some(r));
                        error.set(None);
                    }
                    Err(e) => error.set(Some(format!("parse: {e}"))),
                },
                Ok(resp) => error.set(Some(format!("HTTP {}", resp.status()))),
                Err(e)   => error.set(Some(e.to_string())),
            }
        });
    };

    Effect::new(move |_| fetch());
    Effect::new(move |_| {
        spawn_local(async move {
            let mut iv = IntervalStream::new(1_500);
            while iv.next().await.is_some() { fetch(); }
        });
    });

    view! {
        <div class="page-stub">
            <div class="pipeline-header">
                <h2 style="margin:0">"Node Audio Pipeline"</h2>
                {move || error.get().map(|e| view! { <span class="daily-error">{e}</span> })}
                {move || data.get().map(|d| {
                    let chip_cls = if d.active_pipeline == "rust" { "tone-badge tone-blue" } else { "tone-badge tone-green" };
                    let label = if d.active_pipeline == "rust" { "Rust sidecar" } else { "Python" };
                    let lag = d.pipeline_seconds_behind_realtime
                        .map(|s| format!(" · {s:.2}s behind realtime")).unwrap_or_default();
                    view! {
                        <div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap">
                            <span class=chip_cls>{label}</span>
                            <span class="muted" style="font-size:0.85rem">
                                {format!("{} node(s){}", d.nodes.len(), lag)}
                            </span>
                        </div>
                    }
                })}
            </div>

            {move || match data.get() {
                None => view! { <div class="muted">"Loading…"</div> }.into_any(),
                Some(resp) => {
                    let is_rust = resp.active_pipeline == "rust";
                    if resp.nodes.is_empty() {
                        return view! {
                            <div class="empty-state">
                                <p>"No nodes have sent audio frames yet."</p>
                                <p class="muted">"Nodes appear here after their first ingest frame is received."</p>
                            </div>
                        }.into_any();
                    }
                    resp.nodes.into_iter().map(|node| {
                        view! { <NodeCard node=node edits=edits is_rust=is_rust /> }
                    }).collect_view().into_any()
                }
            }}
        </div>
    }
}

// ---------------------------------------------------------------------------
// NodeCard
// ---------------------------------------------------------------------------

#[component]
fn NodeCard(
    node: PipelineNode,
    edits: RwSignal<HashMap<String, NodeEdit>>,
    is_rust: bool,
) -> impl IntoView {
    let node_id = node.node_id.clone();
    let stages = node.stages.clone();
    let mics = node.mics.clone();
    let frame_gaps = node.frame_gaps;
    let zero_padded = node.zero_padded_degraded;
    let audio_status = node.audio_status.clone();
    let sample_rate = node.sample_rate_hz;
    let last_ns = node.last_frame_ns;
    let node_type_str = if node.node_type.is_empty() { "unknown".to_string() } else { node.node_type.clone() };
    let age = age_text(last_ns);
    let rms_all: Vec<f64> = mics.first().map(|m| m.rms_recent.clone()).unwrap_or_default();
    let spark_all = sparkline_path(&rms_all, 120.0, 24.0);
    let has_rms_all = !rms_all.is_empty();
    let has_gaps = frame_gaps > 0;
    let has_zero_padded = zero_padded > 0;

    let save_error: RwSignal<Option<String>> = RwSignal::new(None);
    let saving: RwSignal<bool> = RwSignal::new(false);
    let save_ok: RwSignal<bool> = RwSignal::new(false);

    // Save handler
    let nid_save = node_id.clone();
    let on_save = move |_| {
        let nid = nid_save.clone();
        let edit = match edits.get_untracked().get(&nid).cloned() {
            Some(e) => e,
            None => return,
        };
        save_error.set(None);
        save_ok.set(false);

        let body = match build_audio_override_body(&edit) {
            Ok(body) => body,
            Err(error) => {
                save_error.set(Some(error));
                return;
            }
        };

        saving.set(true);
        let encoded = js_sys::encode_uri_component(&nid).as_string().unwrap_or_else(|| nid.clone());
        let url = format!("/api/v1/pipeline/nodes/{encoded}/audio");

        spawn_local(async move {
            let result = async {
                Request::patch(&url)
                    .header("Content-Type", "application/json")
                    .body(serde_json::to_string(&body).unwrap_or_default())
                    .map_err(|e: gloo_net::Error| e.to_string())?
                    .send()
                    .await
                    .map_err(|e| e.to_string())
            }.await;

            match result {
                Ok(resp) if resp.ok() => {
                    edits.update(|map| {
                        if let Some(e) = map.get_mut(&nid) { e.dirty = false; }
                    });
                    saving.set(false);
                    save_ok.set(true);
                }
                Ok(resp) => {
                    save_error.set(Some(resp.text().await.unwrap_or_default()));
                    saving.set(false);
                }
                Err(e) => {
                    save_error.set(Some(e));
                    saving.set(false);
                }
            }
        });
    };

    // Pre-build mic rows to avoid borrow issues inside view!
    let mic_rows: Vec<_> = mics.iter().enumerate().map(|(i, mic)| {
        let label = mic.label.clone();
        let spark = sparkline_path(&mic.rms_recent, 60.0, 16.0);
        let has_rms = !mic.rms_recent.is_empty();

        // Each closure needs its own node_id clone.
        let nid_val  = node_id.clone();
        let nid_chg  = node_id.clone();

        let gain_val  = move || edits.get().get(&nid_val).and_then(|e| e.mic_gains_db.get(i).copied()).unwrap_or(0.0);
        let gain_chg  = move |ev: web_sys::Event| {
            let val: f64 = event_target_value(&ev).parse().unwrap_or(0.0);
            edits.update(|map| {
                if let Some(e) = map.get_mut(&nid_chg) {
                    if i < e.mic_gains_db.len() { e.mic_gains_db[i] = val; }
                    e.dirty = true;
                }
            });
        };

        (label, spark, has_rms, gain_val, gain_chg)
    }).collect();

    // HP / LP / smoothing — each closure needs its own clone.
    let nid_hp_val = node_id.clone();
    let nid_hp_chg = node_id.clone();
    let nid_lp_val = node_id.clone();
    let nid_lp_chg = node_id.clone();
    let nid_sm_chg = node_id.clone();

    let hp_val  = move || edits.get().get(&nid_hp_val).map(|e| e.hp_hz).unwrap_or(0.0);
    let hp_chg  = move |ev: web_sys::Event| {
        let val: f64 = event_target_value(&ev).parse().unwrap_or(0.0);
        edits.update(|map| { if let Some(e) = map.get_mut(&nid_hp_chg) { e.hp_hz = val; e.dirty = true; } });
    };
    let lp_val  = move || edits.get().get(&nid_lp_val).map(|e| e.lp_hz).unwrap_or(0.0);
    let lp_chg  = move |ev: web_sys::Event| {
        let val: f64 = event_target_value(&ev).parse().unwrap_or(0.0);
        edits.update(|map| { if let Some(e) = map.get_mut(&nid_lp_chg) { e.lp_hz = val; e.dirty = true; } });
    };
    let sm_chg  = move |ev: web_sys::Event| {
        let val = event_target_value(&ev);
        edits.update(|map| { if let Some(e) = map.get_mut(&nid_sm_chg) { e.smoothing = val; e.dirty = true; } });
    };

    let nid_mode_chg = node_id.clone();
    let mode_chg = move |ev: web_sys::Event| {
        let value = event_target_value(&ev);
        edits.update(|map| {
            if let Some(edit) = map.get_mut(&nid_mode_chg) {
                edit.stage_mode = value == "stages";
                if edit.stage_mode && (edit.stage_json.trim().is_empty() || edit.stage_json.trim() == "[]") {
                    edit.stage_json = default_stage_json();
                }
                edit.dirty = true;
            }
        });
    };

    // Smoothing option selected checks — one clone per option.
    let nid_sm_off = node_id.clone();
    let nid_sm_50  = node_id.clone();
    let nid_sm_200 = node_id.clone();
    let nid_mode_legacy = node_id.clone();
    let nid_mode_stages = node_id.clone();
    let nid_mode_panel = node_id.clone();
    let nid_stage_json_render = node_id.clone();
    let nid_stage_json_input = node_id.clone();

    let nid_dirty = node_id.clone();

    view! {
        <div class="diag-card pipeline-node-card">

            // Header
            <div class="pipeline-node-header">
                <div>
                    <h3 style="margin:0">{node_id.clone()}</h3>
                    <span class="muted" style="font-size:0.8rem">{node_type_str}</span>
                </div>
                <div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap">
                    <span class=audio_status_chip(&audio_status)>{audio_status.clone()}</span>
                    {sample_rate.map(|sr| view! {
                        <span class="muted" style="font-size:0.8rem">{format!("{sr} Hz")}</span>
                    })}
                    {age.map(|t| view! {
                        <span class="muted" style="font-size:0.8rem">{t}</span>
                    })}
                </div>
            </div>

            // Stage chips
            <div class="pipeline-stage-row">
                {stages.into_iter().map(|s| {
                    let chip_cls = stage_chip_class(s.lag_s, s.drops);
                    let lag_str = fmt_lag(s.lag_s);
                    let drops = s.drops;
                    let q_str = if s.queue_max > 0 { format!(" {}/{}", s.queue_depth, s.queue_max) } else { String::new() };
                    let title = format!("in:{} out:{} drops:{}", fmt_count(s.count_in), fmt_count(s.count_out), fmt_count(drops));
                    view! {
                        <div class=chip_cls title=title>
                            <span class="pipeline-stage-name">{s.name}</span>
                            <span class="pipeline-stage-lag">{lag_str}</span>
                            {(!q_str.is_empty()).then(|| view! {
                                <span class="pipeline-stage-queue muted">{q_str}</span>
                            })}
                            {(drops > 0).then(|| view! {
                                <span class="pipeline-stage-drops">{format!("▼{}", fmt_count(drops))}</span>
                            })}
                        </div>
                    }
                }).collect_view()}
            </div>

            // RMS sparkline (aggregate)
            {has_rms_all.then(|| view! {
                <svg width="120" height="24" class="rms-sparkline" style="margin:0.3rem 0;display:block">
                    <path d=spark_all stroke="var(--md-sys-color-primary)" stroke-width="1.5" fill="none" />
                </svg>
            })}

            // Mic settings
            <div style="margin-top:0.75rem">
                <div class="pipeline-section-label">
                    "Mic Settings"
                    {is_rust.then(|| view! {
                        <span class="muted" style="font-size:0.75rem;margin-left:0.5rem">
                            "(live — forwarded to Rust sidecar)"
                        </span>
                    })}
                </div>

                // Per-mic gain + sparkline
                <div class="mic-table">
                    <div class="mic-table-row mic-table-header">
                        <span>"Mic"</span>
                        <span>"Gain (dB)"</span>
                        <span>"RMS"</span>
                    </div>
                    {mic_rows.into_iter().map(|(label, spark, has_rms, gain_val, gain_chg)| view! {
                        <div class="mic-table-row">
                            <span class="mic-label">{label}</span>
                            <input
                                type="number" step="0.5" min="-60" max="60"
                                class="mic-input"
                                prop:value=gain_val
                                on:change=gain_chg
                            />
                            {if has_rms {
                                view! {
                                    <svg width="60" height="16" class="rms-sparkline">
                                        <path d=spark stroke="var(--md-sys-color-primary)" stroke-width="1.2" fill="none" />
                                    </svg>
                                }.into_any()
                            } else {
                                view! { <span class="muted">"—"</span> }.into_any()
                            }}
                        </div>
                    }).collect_view()}
                </div>

                // Node-level: HP / LP / Smoothing
                <div class="mic-table mic-shared-row" style="margin-top:0.5rem">
                    <div class="mic-table-row" style="grid-template-columns:max-content 1fr 1fr 1fr">
                        <span class="mic-label muted" style="font-size:0.8rem">"all mics"</span>
                        <input
                            type="number" step="10" min="0" max="20000"
                            class="mic-input"
                            prop:value=hp_val on:change=hp_chg
                        />
                        <input
                            type="number" step="100" min="0" max="24000"
                            class="mic-input"
                            prop:value=lp_val on:change=lp_chg
                        />
                        <select class="mic-input" on:change=sm_chg>
                            <option value="off"
                                selected=move || edits.get().get(&nid_sm_off).map(|e| e.smoothing == "off").unwrap_or(true)>
                                "off"
                            </option>
                            <option value="ema_50ms"
                                selected=move || edits.get().get(&nid_sm_50).map(|e| e.smoothing == "ema_50ms").unwrap_or(false)>
                                "ema_50ms"
                            </option>
                            <option value="ema_200ms"
                                selected=move || edits.get().get(&nid_sm_200).map(|e| e.smoothing == "ema_200ms").unwrap_or(false)>
                                "ema_200ms"
                            </option>
                        </select>
                    </div>
                    <div class="mic-table-row mic-table-header" style="grid-template-columns:max-content 1fr 1fr 1fr">
                        <span></span>
                        <span>"HP (Hz)"</span>
                        <span>"LP (Hz)"</span>
                        <span>"Smoothing"</span>
                    </div>
                </div>

                <div style="margin-top:0.75rem">
                    <div class="pipeline-section-label">"DSP Override Mode"</div>
                    <div class="pipeline-mode-row">
                        <select class="mic-input" on:change=mode_chg>
                            <option value="legacy"
                                selected=move || edits.get().get(&nid_mode_legacy).map(|e| !e.stage_mode).unwrap_or(true)>
                                "Legacy filters"
                            </option>
                            <option value="stages"
                                selected=move || edits.get().get(&nid_mode_stages).map(|e| e.stage_mode).unwrap_or(false)>
                                "Stage chain"
                            </option>
                        </select>
                        <span class="muted" style="font-size:0.8rem">
                            "Legacy mode sends gain/HP/LP/smoothing and explicitly clears any stored stage chain on save."
                        </span>
                    </div>

                    {move || {
                        if edits.get().get(&nid_mode_panel).map(|e| e.stage_mode).unwrap_or(false) {
                            let nid_stage_json_render = nid_stage_json_render.clone();
                            let nid_stage_json_input = nid_stage_json_input.clone();
                            view! {
                                <div class="pipeline-stage-editor">
                                    <textarea
                                        class="mic-input pipeline-stage-json"
                                        prop:value=move || {
                                            edits
                                                .get()
                                                .get(&nid_stage_json_render)
                                                .map(|edit| edit.stage_json.clone())
                                                .unwrap_or_else(|| "[]".into())
                                        }
                                        on:input=move |ev: web_sys::Event| {
                                            let value = event_target_value(&ev);
                                            edits.update(|map| {
                                                if let Some(edit) = map.get_mut(&nid_stage_json_input) {
                                                    edit.stage_json = value;
                                                    edit.dirty = true;
                                                }
                                            });
                                        }
                                        placeholder="[{\"type\":\"gain\",\"db\":0.0}]"
                                    />
                                    <p class="muted" style="font-size:0.8rem">
                                        "Accepted stages: gain, highpass, lowpass, bandpass, dc_block, passthrough. Save sends this JSON array directly to the canonical backend validator."
                                    </p>
                                </div>
                            }.into_any()
                        } else {
                            view! {
                                <p class="muted" style="font-size:0.8rem;margin-top:0.4rem">
                                    "Stage mode is off. Save writes only the legacy per-node filter fields."
                                </p>
                            }.into_any()
                        }
                    }}
                </div>

                // Save row
                <div class="pipeline-save-row">
                    <button class="btn-primary" disabled=move || saving.get() on:click=on_save>
                        {move || if saving.get() { "Saving…" } else { "Save" }}
                    </button>
                    <span class="muted" style="font-size:0.8rem">
                        {if is_rust {
                            "Applies live via the Rust sidecar and remains persisted in server settings."
                        } else {
                            "Applies live to the Python ingest pipeline and remains persisted in server settings."
                        }}
                    </span>
                    {move || edits.get().get(&nid_dirty).filter(|e| e.dirty).map(|_| view! {
                        <span class="muted" style="font-size:0.8rem">"Unsaved changes"</span>
                    })}
                    {move || save_ok.get().then_some(view! {
                        <span style="color:var(--mmp-sys-color-ok);font-size:0.8rem">"Saved"</span>
                    })}
                    {move || save_error.get().map(|e| view! {
                        <span class="daily-error">{e}</span>
                    })}
                </div>
            </div>

            // Footer
            <div class="pipeline-node-footer muted">
                {has_gaps.then(|| view! {
                    <span>{format!("⚠ {} frame gaps", frame_gaps)}</span>
                })}
                {has_zero_padded.then(|| view! {
                    <span>{format!("{} zero-padded", zero_padded)}</span>
                })}
            </div>
        </div>
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_node(audio_override: Option<NodeAudioOverride>) -> PipelineNode {
        PipelineNode {
            node_id: "node-1".into(),
            node_type: "sirith_tetra".into(),
            mics: vec![MicData {
                index: 0,
                label: "CH0".into(),
                gain_db: 3.0,
                hp_hz: 120.0,
                lp_hz: 8_000.0,
                smoothing: "ema_50ms".into(),
                rms_recent: Vec::new(),
            }],
            audio_override,
            ..Default::default()
        }
    }

    fn sample_edit(stage_mode: bool, stage_json: &str) -> NodeEdit {
        NodeEdit {
            mic_gains_db: vec![2.5],
            hp_hz: 90.0,
            lp_hz: 7_500.0,
            smoothing: "ema_200ms".into(),
            stage_mode,
            stage_json: stage_json.into(),
            dirty: true,
        }
    }

    #[test]
    fn node_edit_prefers_stage_override_when_present() {
        let edit = NodeEdit::from_node(&sample_node(Some(NodeAudioOverride {
            stages: Some(vec![json!({"type": "gain", "db": 6.0})]),
            ..Default::default()
        })));

        assert!(edit.stage_mode);
        assert!(edit.stage_json.contains("\"type\": \"gain\""));
        assert_eq!(edit.hp_hz, 120.0);
        assert_eq!(edit.lp_hz, 8_000.0);
    }

    #[test]
    fn build_audio_override_body_clears_stages_in_legacy_mode() {
        let payload = build_audio_override_body(&sample_edit(false, &default_stage_json())).unwrap();

        assert_eq!(
            payload,
            json!({
                "mic_gains_db": [2.5],
                "hp_hz": 90.0,
                "lp_hz": 7_500.0,
                "smoothing": "ema_200ms",
                "stages": [],
            })
        );
    }

    #[test]
    fn build_audio_override_body_uses_stage_json_in_stage_mode() {
        let payload = build_audio_override_body(&sample_edit(true, r#"[{"type":"highpass","cutoff_hz":120.0}]"#)).unwrap();

        assert_eq!(
            payload,
            json!({
                "stages": [{"type": "highpass", "cutoff_hz": 120.0}],
            })
        );
    }

    #[test]
    fn build_audio_override_body_rejects_non_array_stage_json() {
        let error = build_audio_override_body(&sample_edit(true, r#"{"type":"gain"}"#)).unwrap_err();

        assert!(error.contains("JSON array"));
    }
}
