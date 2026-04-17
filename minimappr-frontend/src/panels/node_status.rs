use crate::state::{AppState, NodeStatus};
use leptos::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlAudioElement;

fn health_chip_class(health: &str) -> &'static str {
    match health {
        "online"   => "health-chip online",
        "degraded" => "health-chip degraded",
        "offline"  => "health-chip offline",
        _          => "health-chip unknown",
    }
}

fn sparkline_path(rms: &[f64], width: f64, height: f64) -> String {
    if rms.is_empty() { return String::new(); }
    let max = rms.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(1e-6);
    let n = rms.len();
    let points: Vec<String> = rms.iter().enumerate().map(|(i, &v)| {
        let x = i as f64 / (n - 1).max(1) as f64 * width;
        let y = height - (v / max) * height;
        format!("{x:.1},{y:.1}")
    }).collect();
    format!("M {}", points.join(" L "))
}

#[component]
pub fn NodeStatusPanel() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let nodes = state.nodes;

    view! {
        <div class="panel">
            <div class="panel-header">"Node Status"</div>
            <div class="panel-scroll">
                {move || {
                    let ns = nodes.get();
                    if ns.is_empty() {
                        view! { <div class="empty-state">"No nodes reported yet"</div> }.into_any()
                    } else {
                        ns.into_iter().map(|n| view! { <NodeCard node=n /> }).collect_view().into_any()
                    }
                }}
            </div>
        </div>
    }
}

#[component]
fn NodeCard(node: NodeStatus) -> impl IntoView {
    let node_id = node.node_id.clone();
    let health = node.health.clone();
    let chip_class = health_chip_class(&health);

    let gps = node.gps_fix.map(|v| if v { "GPS ✓" } else { "GPS ✗" }).unwrap_or("");
    let temp = node.temperature_c.map(|t| format!("{t:.1}°C")).unwrap_or_default();
    let hum  = node.humidity.map(|h| format!("{:.0}% RH", h * 100.0)).unwrap_or_default();
    let fw   = node.firmware_version.clone().unwrap_or_default();
    let ch   = node.channel_count.map(|c| format!("{c}ch")).unwrap_or_default();

    let rms = node.rms_history.clone().unwrap_or_default();
    let path = sparkline_path(&rms, 180.0, 24.0);

    let id_clone  = node_id.clone();
    let id_clone2 = node_id.clone();

    view! {
        <div class="node-card">
            <div class="node-card-header">
                <span class="node-id">{node_id.clone()}</span>
                <span class=chip_class>{health.clone()}</span>
            </div>
            <div class="node-meta">
                {if !gps.is_empty() { view!(<span>{gps}</span>).into_any() } else { view!(<span></span>).into_any() }}
                {if !temp.is_empty() { view!(<span>{temp}</span>).into_any() } else { view!(<span></span>).into_any() }}
                {if !hum.is_empty()  { view!(<span>{hum}</span>).into_any()  } else { view!(<span></span>).into_any() }}
                {if !ch.is_empty()   { view!(<span>{ch}</span>).into_any()   } else { view!(<span></span>).into_any() }}
                {if !fw.is_empty()   { view!(<span>{fw}</span>).into_any()   } else { view!(<span></span>).into_any() }}
            </div>
            {if !path.is_empty() {
                view! {
                    <div class="sparkline-wrap">
                        <svg viewBox="0 0 180 24" preserveAspectRatio="none">
                            <path class="sparkline-line" d=path />
                        </svg>
                    </div>
                }.into_any()
            } else { view!(<div></div>).into_any() }}
            <div class="node-actions">
                <button class="btn-sm" on:click=move |_| play_node_audio(&id_clone, None)>
                    "▶ Mix"
                </button>
                {(0..node.channel_count.unwrap_or(0)).map(|ch_idx| {
                    let id = id_clone2.clone();
                    view! {
                        <button class="btn-sm" on:click=move |_| play_node_audio(&id, Some(ch_idx))>
                            {format!("Ch {}", ch_idx + 1)}
                        </button>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}

fn play_node_audio(node_id: &str, channel: Option<u32>) {
    let mut url = format!(
        "/api/v1/nodes/{}/audio/recent?seconds=10",
        js_sys::encode_uri_component(node_id)
    );
    if let Some(ch) = channel {
        url.push_str(&format!("&channel={ch}"));
    }
    if let Some(audio) = audio_element() {
        audio.set_src(&url);
        let _ = audio.play();
    }
}

fn audio_element() -> Option<HtmlAudioElement> {
    web_sys::window()?
        .document()?
        .get_element_by_id("audio-player")?
        .dyn_into::<HtmlAudioElement>()
        .ok()
}
