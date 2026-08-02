use crate::state::{AppState, NodeStatus};
use crate::ui::health_chip_class;
use leptos::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::HtmlAudioElement;

fn sparkline_path(rms: &[f64], width: f64, height: f64) -> String {
    if rms.is_empty() {
        return String::new();
    }
    let max = rms
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-6);
    let n = rms.len();
    let points: Vec<String> = rms
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = i as f64 / (n - 1).max(1) as f64 * width;
            let y = height - (v / max) * height;
            format!("{x:.1},{y:.1}")
        })
        .collect();
    format!("M {}", points.join(" L "))
}

fn time_quality_label_and_color(tq: &str) -> (&'static str, &'static str) {
    match tq {
        "gps_locked" => ("GPS PPS Lock", "var(--mmp-sys-color-ok)"),
        "gps_holdover" => ("GPS Holdover", "var(--mmp-sys-color-warn)"),
        "ntp_disciplined" | "ntp_sync" => ("NTP Sync", "var(--md-sys-color-tertiary)"),
        "free_running" => ("Server Receipt", "var(--mmp-sys-color-danger)"),
        _ => ("Unknown", "var(--md-sys-color-on-surface-variant)"),
    }
}

/// Format a predicted holdover phase error (nanoseconds) as a signed magnitude
/// with an adaptive unit: ns under 1 us, us under 1 ms, otherwise ms.
fn format_predicted_error_ns(error_ns: f64) -> String {
    let e = error_ns.abs();
    if e < 1_000.0 {
        format!("±{e:.0} ns")
    } else if e < 1_000_000.0 {
        format!("±{:.1} µs", e / 1_000.0)
    } else {
        format!("±{:.1} ms", e / 1_000_000.0)
    }
}

/// When a node is in GPS holdover, extract "age · ±error" from its timing
/// diagnostics `clock_holdover` block. Returns None if not actively holding over
/// or the diagnostics are absent. Tolerant of missing/partial fields.
fn holdover_detail_text(diagnostics: Option<&serde_json::Value>) -> Option<String> {
    let clock = diagnostics?.get("clock_holdover")?;
    if !clock
        .get("holdover_active")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return None;
    }
    let age_ms = clock
        .get("holdover_age_ms")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let age = age_text_from_seconds(age_ms / 1000.0);
    let detail = match clock
        .get("predicted_error_ns")
        .and_then(serde_json::Value::as_f64)
    {
        Some(error_ns) => format!("{age} · {}", format_predicted_error_ns(error_ns)),
        None => age,
    };
    Some(detail)
}

/// Learned long-term crystal frequency as "-3.5 ppm ±0.25" once the model is
/// characterized. Returns None when the model has not yet validated or the
/// diagnostics are absent.
fn crystal_drift_text(diagnostics: Option<&serde_json::Value>) -> Option<String> {
    let clock = diagnostics?.get("clock_holdover")?;
    if !clock
        .get("lt_valid")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return None;
    }
    let lt_ppm = clock.get("lt_ppm").and_then(serde_json::Value::as_f64)?;
    let sigma = clock
        .get("lt_ppm_sigma")
        .and_then(serde_json::Value::as_f64);
    Some(match sigma {
        Some(sigma) => format!("{lt_ppm:.2} ppm ±{sigma:.2}"),
        None => format!("{lt_ppm:.2} ppm"),
    })
}

fn age_text_from_seconds(age_seconds: f64) -> String {
    if age_seconds < 60.0 {
        format!("{:.0}s", age_seconds)
    } else if age_seconds < 3600.0 {
        format!("{:.0}m", age_seconds / 60.0)
    } else {
        format!("{:.1}h", age_seconds / 3600.0)
    }
}

fn age_text_from_ns(timestamp_ns: i64) -> Option<String> {
    let now_ms = js_sys::Date::now();
    let sample_ms = (timestamp_ns as f64) / 1_000_000.0;
    if !now_ms.is_finite() || !sample_ms.is_finite() {
        return None;
    }
    let age_seconds = ((now_ms - sample_ms) / 1000.0).max(0.0);
    Some(age_text_from_seconds(age_seconds))
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

    // Stale-node deletion lives in the details panel and is only offered for
    // offline nodes — an active node would just repopulate on the next poll
    // (and the backend rejects deleting one with 409).
    let nodes_signal = use_context::<AppState>().expect("AppState").nodes;
    let health_is_offline = health == "offline";
    let id_for_delete = node.node_id.clone();
    let confirm_delete = RwSignal::new(false);
    let deleting = RwSignal::new(false);
    let delete_error = RwSignal::new(None::<String>);

    let gps_signal = node
        .metadata
        .as_ref()
        .and_then(|m| m.gps.as_ref())
        .and_then(|g| g.signal.clone());
    let gps_source = node
        .metadata
        .as_ref()
        .and_then(|m| m.gps.as_ref())
        .and_then(|g| g.position_source.clone());

    let gps = if let Some(signal) = gps_signal.clone() {
        format!("GPS {signal}")
    } else {
        node.gps_fix
            .map(|v| {
                if v {
                    "GPS fix".to_string()
                } else {
                    "GPS no-fix".to_string()
                }
            })
            .unwrap_or_default()
    };

    let effective_temp_c = node
        .latest_environment
        .as_ref()
        .and_then(|e| e.temperature_c)
        .or(node.temperature_c);
    let effective_humidity_fraction = node
        .latest_environment
        .as_ref()
        .and_then(|e| e.humidity_fraction)
        .or(node.humidity);

    let temp = effective_temp_c
        .map(|t| format!("{t:.1}°C"))
        .unwrap_or_default();
    let hum = effective_humidity_fraction
        .map(|h| format!("{:.0}% RH", h * 100.0))
        .unwrap_or_default();
    let fw = node.firmware_version.clone().unwrap_or_default();

    let channel_total = node
        .audio_debug
        .as_ref()
        .and_then(|a| a.sensor_count)
        .or(node.channel_count)
        .unwrap_or(0);
    let active_channel_count = node
        .audio_debug
        .as_ref()
        .and_then(|a| a.active_sensor_count)
        .unwrap_or(0);
    let ch = if channel_total > 0 {
        format!("ch {active_channel_count}/{channel_total}")
    } else {
        String::new()
    };

    let last_seen = node
        .last_seen_seconds_ago
        .map(age_text_from_seconds)
        .or_else(|| node.last_seen_ns.and_then(age_text_from_ns))
        .map(|age| format!("seen {age} ago"))
        .unwrap_or_default();

    let env_age = node
        .latest_environment
        .as_ref()
        .and_then(|e| e.timestamp_ns)
        .and_then(age_text_from_ns)
        .map(|age| format!("env {age} ago"))
        .unwrap_or_default();

    let local_position_text = node.position_m.as_ref().and_then(|position| {
        if position.len() >= 3 {
            Some(format!(
                "[{:.2}, {:.2}, {:.2}] m",
                position[0], position[1], position[2]
            ))
        } else {
            None
        }
    });
    let geo_position_text = node.position_geo.as_ref().map(|geo| {
        format!(
            "{:.5}, {:.5}, {:.1} m",
            geo.lat,
            geo.lon,
            geo.alt_m.unwrap_or(0.0)
        )
    });

    let audio_status = node
        .audio_debug
        .as_ref()
        .and_then(|audio| audio.status.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let audio_rate_text = node
        .audio_debug
        .as_ref()
        .and_then(|audio| audio.sample_rate_hz)
        .map(|rate| format!("{rate:.0} Hz"))
        .unwrap_or_else(|| "-".to_string());
    let audio_age_text = node
        .audio_debug
        .as_ref()
        .and_then(|audio| audio.age_seconds)
        .map(age_text_from_seconds)
        .unwrap_or_else(|| "-".to_string());
    let audio_rms_text = node
        .audio_debug
        .as_ref()
        .and_then(|audio| audio.rms)
        .map(|rms| format!("{rms:.5}"))
        .unwrap_or_else(|| "-".to_string());

    let pressure_text = node
        .latest_environment
        .as_ref()
        .and_then(|env| env.pressure_pa)
        .map(|pressure| format!("{pressure:.0} Pa"))
        .unwrap_or_else(|| "-".to_string());
    let wind_speed_text = node
        .latest_environment
        .as_ref()
        .and_then(|env| env.wind_speed_mps)
        .map(|speed| format!("{speed:.1} m/s"))
        .unwrap_or_else(|| "-".to_string());
    let wind_direction_text = node
        .latest_environment
        .as_ref()
        .and_then(|env| env.wind_dir_deg)
        .map(|degrees| format!("{degrees:.0}°"))
        .unwrap_or_else(|| "-".to_string());
    let solar_text = node
        .latest_environment
        .as_ref()
        .and_then(|env| env.solar_lux)
        .map(|lux| format!("{lux:.0} lux"))
        .unwrap_or_else(|| "-".to_string());
    let environment_source_text = node
        .latest_environment
        .as_ref()
        .and_then(|env| env.metadata.as_ref())
        .and_then(|metadata| metadata.get("source"))
        .and_then(|source| source.as_str())
        .map(ToString::to_string)
        .unwrap_or_else(|| "-".to_string());

    let time_quality_display = node
        .latest_time_quality
        .as_deref()
        .map(time_quality_label_and_color);
    // Only surface the holdover age/error suffix while the node reports it as
    // actively holding over (defends against stale diagnostics after re-lock).
    let holdover_detail = if node.latest_time_quality.as_deref() == Some("gps_holdover") {
        holdover_detail_text(node.latest_timing_diagnostics.as_ref())
    } else {
        None
    };
    let crystal_drift = crystal_drift_text(node.latest_timing_diagnostics.as_ref());

    let node_type_text = node.node_type.clone().unwrap_or_else(|| "-".to_string());
    let mobility_text = node.mobility.clone().unwrap_or_else(|| "-".to_string());
    let capabilities_text = node
        .capabilities
        .as_ref()
        .filter(|caps| !caps.is_empty())
        .map(|caps| caps.join(", "))
        .unwrap_or_else(|| "-".to_string());
    let capability_chips = node.capabilities.clone().unwrap_or_default();
    let bit_codes_text = node
        .bit_failure_codes
        .as_ref()
        .filter(|codes| !codes.is_empty())
        .map(|codes| codes.join(", "))
        .unwrap_or_else(|| "none".to_string());

    let rms = node.rms_history.clone().unwrap_or_default();
    let path = sparkline_path(&rms, 180.0, 24.0);

    // Only `no_audio` means "this node is not delivering audio". Under a
    // split-process deployment the API process does not own the audio buffer,
    // so it reports `external_ingest_process` — freshness is unmeasured here,
    // not zero. Falling that through to "Audio offline" made a healthy,
    // actively-ingesting node read as dead.
    let audio_status_label: Option<&'static str> = node
        .audio_debug
        .as_ref()
        .and_then(|a| a.status.as_deref())
        .map(|s| match s {
            "recent" => "Audio ✓",
            "stale" => "Audio stale",
            "no_audio" => "Audio offline",
            "external_ingest_process" => "Audio → ingest proc",
            _ => "Audio unknown",
        });

    let id_clone = node_id.clone();
    let id_clone2 = node_id.clone();

    view! {
        <div class="node-card">
            <div class="node-card-header">
                <span class="node-id">{node_id.clone()}</span>
                <span class=chip_class>{health.clone()}</span>
                {audio_status_label.map(|lbl| view! {
                    <span class="node-audio-status-label muted">{lbl}</span>
                })}
            </div>
            <div class="node-meta">
                {if !gps.is_empty() { view!(<span>{gps}</span>).into_any() } else { view!(<span></span>).into_any() }}
                {if !temp.is_empty() { view!(<span>{temp}</span>).into_any() } else { view!(<span></span>).into_any() }}
                {if !hum.is_empty()  { view!(<span>{hum}</span>).into_any()  } else { view!(<span></span>).into_any() }}
                {if !ch.is_empty()   { view!(<span>{ch}</span>).into_any()   } else { view!(<span></span>).into_any() }}
                {if !last_seen.is_empty() { view!(<span>{last_seen}</span>).into_any() } else { view!(<span></span>).into_any() }}
                {if !env_age.is_empty() { view!(<span>{env_age}</span>).into_any() } else { view!(<span></span>).into_any() }}
                {if !fw.is_empty()   { view!(<span>{fw}</span>).into_any()   } else { view!(<span></span>).into_any() }}
            </div>
            <div class="node-meta">
                {capability_chips.into_iter().map(|capability| view! {
                    <span class="tone-badge tone-blue">{capability}</span>
                }).collect_view()}
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
                {if channel_total > 0 {
                    view! { <span class="node-action-hint">{format!("{channel_total} channels (expand for list)")}</span> }.into_any()
                } else {
                    view! { <span class="node-action-hint">"No channel telemetry"</span> }.into_any()
                }}
                <a class="btn-sm" href=format!("/settings/nodes/{}", node_id.clone())>"Configure"</a>
            </div>
            <details class="node-details">
                <summary>"Show details"</summary>
                <div class="node-details-grid">
                    <div class="node-detail-label">"Geo"</div>
                    <div>{geo_position_text.unwrap_or_else(|| "-".to_string())}</div>
                    <div class="node-detail-label">"Local"</div>
                    <div>{local_position_text.unwrap_or_else(|| "-".to_string())}</div>
                    <div class="node-detail-label">"GPS source"</div>
                    <div>{gps_source.unwrap_or_else(|| "-".to_string())}</div>
                    <div class="node-detail-label">"Time precision"</div>
                    <div>
                        {match time_quality_display {
                            Some((label, color)) => view! {
                                <span style=format!("color: {color}; font-weight: 600")>{label}</span>
                                {match holdover_detail {
                                    Some(detail) => view! {
                                        <span class="node-holdover-detail">{format!(" · {detail}")}</span>
                                    }.into_any(),
                                    None => ().into_any(),
                                }}
                            }.into_any(),
                            None => view! { <span>"-"</span> }.into_any(),
                        }}
                    </div>
                    {match crystal_drift {
                        Some(drift) => view! {
                            <div class="node-detail-label">"Clock drift"</div>
                            <div>{drift}</div>
                        }.into_any(),
                        None => ().into_any(),
                    }}
                    <div class="node-detail-label">"Type"</div>
                    <div>{node_type_text}</div>
                    <div class="node-detail-label">"Mobility"</div>
                    <div>{mobility_text}</div>
                    <div class="node-detail-label">"Capabilities"</div>
                    <div>{capabilities_text}</div>
                    <div class="node-detail-label">"BIT failures"</div>
                    <div>{bit_codes_text}</div>

                    <div class="node-detail-label">"Env pressure"</div>
                    <div>{pressure_text}</div>
                    <div class="node-detail-label">"Env wind"</div>
                    <div>{format!("{wind_speed_text} @ {wind_direction_text}")}</div>
                    <div class="node-detail-label">"Env solar"</div>
                    <div>{solar_text}</div>
                    <div class="node-detail-label">"Env source"</div>
                    <div>{environment_source_text}</div>

                    <div class="node-detail-label">"Audio status"</div>
                    <div>{audio_status}</div>
                    <div class="node-detail-label">"Audio sample rate"</div>
                    <div>{audio_rate_text}</div>
                    <div class="node-detail-label">"Audio age"</div>
                    <div>{audio_age_text}</div>
                    <div class="node-detail-label">"Audio RMS"</div>
                    <div>{audio_rms_text}</div>
                    <div class="node-detail-label">"Listen channels"</div>
                    <div class="node-channel-list">
                        {(0..channel_total).map(|ch_idx| {
                            let id = id_clone2.clone();
                            view! {
                                <button class="btn-sm" on:click=move |_| play_node_audio(&id, Some(ch_idx))>
                                    {format!("Ch {}", ch_idx + 1)}
                                </button>
                            }
                        }).collect_view()}
                    </div>
                </div>
                {health_is_offline.then(move || view! {
                    <div class="node-delete-row" style="margin-top: 8px">
                        {move || if confirm_delete.get() {
                            let id = id_for_delete.clone();
                            view! {
                                <div class="row-actions">
                                    <button
                                        class="btn-sm btn-sm--danger"
                                        disabled=move || deleting.get()
                                        on:click=move |_| {
                                            deleting.set(true);
                                            delete_error.set(None);
                                            let id = id.clone();
                                            spawn_local(async move {
                                                match crate::api::delete_node(&id).await {
                                                    Ok(()) => {
                                                        nodes_signal.update(|list| {
                                                            list.retain(|n| n.node_id != id)
                                                        });
                                                    }
                                                    Err(e) => {
                                                        delete_error.set(Some(e));
                                                        deleting.set(false);
                                                        confirm_delete.set(false);
                                                    }
                                                }
                                            });
                                        }
                                    >
                                        {move || if deleting.get() { "Deleting…" } else { "Confirm delete" }}
                                    </button>
                                    <button class="btn-sm" on:click=move |_| confirm_delete.set(false)>
                                        "Cancel"
                                    </button>
                                </div>
                            }.into_any()
                        } else {
                            view! {
                                <button
                                    class="btn-sm btn-sm--danger"
                                    on:click=move |_| confirm_delete.set(true)
                                >
                                    "Delete stale node"
                                </button>
                            }.into_any()
                        }}
                        {move || delete_error.get().map(|e| view! {
                            <p class="row-error">{e}</p>
                        })}
                    </div>
                })}
            </details>
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
