use crate::recording::api;
use crate::recording::{
    GroundTruthEvent, GroundTruthEventIn, RecordingLibraryEntry, RecordingStatus,
};
use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

// ── Library component ────────────────────────────────────────────

#[component]
pub fn RecordingsLibrary() -> impl IntoView {
    let entries: RwSignal<Vec<RecordingLibraryEntry>> = RwSignal::new(vec![]);
    let loading = RwSignal::new(true);
    let error: RwSignal<Option<String>> = RwSignal::new(None);

    let load = move || {
        loading.set(true);
        error.set(None);
        spawn_local(async move {
            match api::fetch_recordings().await {
                Ok(list) => {
                    entries.set(list);
                    error.set(None);
                }
                Err(e) => error.set(Some(e)),
            }
            loading.set(false);
        });
    };

    // Initial fetch on mount.
    load();

    view! {
        <div class="recordings-library">
            <div class="library-header">
                <h3 class="section-title">"Past Recordings"</h3>
                <button
                    class="btn-sm"
                    on:click=move |_| load()
                    title="Refresh recordings list"
                >
                    <span class="material-symbols-rounded">"refresh"</span>
                    "Refresh"
                </button>
            </div>

            {move || if loading.get() {
                view! { <p class="muted">"Loading…"</p> }.into_any()
            } else if let Some(err) = error.get() {
                view! {
                    <div class="library-error">
                        <p class="muted">{err}</p>
                        <button class="btn-sm" on:click=move |_| load()>"Retry"</button>
                    </div>
                }.into_any()
            } else if entries.get().is_empty() {
                view! {
                    <p class="muted">"No recordings yet. Start a new recording above."</p>
                }.into_any()
            } else {
                view! {
                    <div class="library-table-wrap">
                        <table class="library-table">
                            <thead>
                                <tr>
                                    <th>"Started"</th>
                                    <th>"Duration"</th>
                                    <th>"Listener"</th>
                                    <th>"Formats"</th>
                                    <th>"Size"</th>
                                    <th>"Actions"</th>
                                </tr>
                            </thead>
                            <tbody>
                                {move || entries.get().into_iter().map(|entry| {
                                    view! { <RecordingRow entry entries /> }
                                }).collect::<Vec<_>>()}
                            </tbody>
                        </table>
                    </div>
                }.into_any()
            }}
        </div>
    }
}

// ── Single row ───────────────────────────────────────────────────

#[component]
fn RecordingRow(
    entry: RecordingLibraryEntry,
    entries: RwSignal<Vec<RecordingLibraryEntry>>,
) -> impl IntoView {
    let confirm_delete = RwSignal::new(false);
    let deleting = RwSignal::new(false);
    let row_error: RwSignal<Option<String>> = RwSignal::new(None);

    let started = format_ms(entry.started_at_ms);
    let duration = entry
        .duration_seconds
        .map(format_duration)
        .unwrap_or_else(|| "—".into());
    let listener = entry.listener_node_id.clone();
    let size = entry
        .size_bytes
        .map(format_bytes)
        .unwrap_or_else(|| "—".into());
    let is_completed = matches!(entry.status, RecordingStatus::Completed);
    let is_calibration = entry.capture_kind == "calibration";
    let show_ground_truth = RwSignal::new(false);
    let bundle_url = api::calibration_bundle_url(&entry.session_id);
    let sid_for_gt = entry.session_id.clone();

    let ambi_url = api::download_url(&entry.session_id, "ambisonics");
    let object_url = api::download_url(&entry.session_id, "object");
    let iamf_url = api::download_url(&entry.session_id, "iamf");
    let visual_url = api::download_url(&entry.session_id, "visual");
    let video_url = api::download_url(&entry.session_id, "video");
    let show_object = entry.object_available;
    let show_iamf = entry.iamf_available;
    let show_visual = entry.visual_available;
    let show_video = entry.video_available;
    let error_message = entry.error_message.clone();
    // Clone session_id for use in the reactive delete closure.
    let sid_for_delete = entry.session_id.clone();

    view! {
        <tr>
            <td class="td-mono">{started}</td>
            <td class="td-mono">{duration}</td>
            <td class="td-node">{listener}</td>
            <td>
                <div class="format-badges">
                    {is_calibration.then(|| view! {
                        <span class="badge badge--calibration">"Calibration"</span>
                    })}
                    {(!is_calibration && entry.ambisonics_available).then(|| view! {
                        <span class="badge">"Ambi"</span>
                    })}
                    {show_iamf.then(|| view! {
                        <span class="badge">"IAMF"</span>
                    })}
                    {show_object.then(|| view! {
                        <span class="badge">"Obj"</span>
                    })}
                    {show_visual.then(|| view! {
                        <span class="badge">"Visual"</span>
                    })}
                    {show_video.then(|| view! {
                        <span class="badge">"Video"</span>
                    })}
                    {(!is_completed).then(|| view! {
                        <span class="badge badge--dim">{entry.status.label()}</span>
                    })}
                </div>
                {error_message.clone().filter(|msg| !msg.is_empty()).map(|msg| view! {
                    <p class="row-error">{msg}</p>
                })}
            </td>
            <td class="td-mono">{size}</td>
            <td>
                <div class="row-actions">
                    {(is_completed && is_calibration).then({
                        let bundle_url = bundle_url.clone();
                        move || view! {
                            <a class="btn-sm" href=bundle_url.clone() download
                               title="Replayable calibration bundle (audio + geometry + ground truth)">
                                "↓ Bundle"
                            </a>
                        }
                    })}
                    {is_calibration.then(|| view! {
                        <button
                            class="btn-sm"
                            class:btn-sm--active=move || show_ground_truth.get()
                            on:click=move |_| show_ground_truth.update(|v| *v = !*v)
                        >
                            <span class="material-symbols-rounded">"my_location"</span>
                            "Ground Truth"
                        </button>
                    })}
                    {(is_completed && !is_calibration).then({
                        let ambi_url = ambi_url.clone();
                        move || view! {
                            <a class="btn-sm" href=ambi_url.clone() download>
                                "↓ Ambi"
                            </a>
                        }
                    })}
                    {(is_completed && show_object).then({
                        let object_url = object_url.clone();
                        move || view! {
                            <a class="btn-sm" href=object_url.clone() download>
                                "↓ Obj"
                            </a>
                        }
                    })}
                    {(is_completed && show_iamf).then({
                        let iamf_url = iamf_url.clone();
                        move || view! {
                            <a class="btn-sm" href=iamf_url.clone() download>
                                "↓ IAMF"
                            </a>
                        }
                    })}
                    {(is_completed && show_visual).then({
                        let visual_url = visual_url.clone();
                        move || view! {
                            <a class="btn-sm" href=visual_url.clone() download>
                                "↓ Visual"
                            </a>
                        }
                    })}
                    {(is_completed && show_video).then({
                        let video_url = video_url.clone();
                        move || view! {
                            <a class="btn-sm" href=video_url.clone() download>
                                "↓ Video"
                            </a>
                        }
                    })}
                    {move || if confirm_delete.get() {
                        let sid = sid_for_delete.clone();
                        view! {
                            <button
                                class="btn-sm btn-sm--danger"
                                disabled=move || deleting.get()
                                on:click=move |_| {
                                    deleting.set(true);
                                    let sid = sid.clone();
                                    spawn_local(async move {
                                        match api::delete_recording(&sid).await {
                                            Ok(()) => {
                                                entries.update(|list| {
                                                    list.retain(|e| e.session_id != sid)
                                                });
                                            }
                                            Err(e) => {
                                                row_error.set(Some(e));
                                                deleting.set(false);
                                                confirm_delete.set(false);
                                            }
                                        }
                                    });
                                }
                            >
                                {move || if deleting.get() { "Deleting…" } else { "Confirm" }}
                            </button>
                            <button
                                class="btn-sm"
                                on:click=move |_| confirm_delete.set(false)
                            >
                                "Cancel"
                            </button>
                        }.into_any()
                    } else {
                        view! {
                            <button
                                class="btn-sm btn-sm--danger"
                                on:click=move |_| confirm_delete.set(true)
                            >
                                "Delete"
                            </button>
                        }.into_any()
                    }}
                </div>
                {move || row_error.get().map(|e| view! {
                    <p class="row-error">{e}</p>
                })}
            </td>
        </tr>
        {is_calibration.then(|| view! {
            <tr class="ground-truth-row" class:hidden=move || !show_ground_truth.get()>
                <td colspan="6">
                    {move || show_ground_truth.get().then({
                        let session_id = sid_for_gt.clone();
                        move || view! { <GroundTruthPanel session_id=session_id.clone() /> }
                    })}
                </td>
            </tr>
        })}
    }
}

// ── Ground truth panel (calibration sessions) ────────────────────

#[component]
fn GroundTruthPanel(session_id: String) -> impl IntoView {
    let events: RwSignal<Vec<GroundTruthEvent>> = RwSignal::new(vec![]);
    let error: RwSignal<Option<String>> = RwSignal::new(None);

    // Form state.
    let label = RwSignal::new(String::new());
    let category = RwSignal::new(String::from("unknown"));
    let lat = RwSignal::new(String::new());
    let lon = RwSignal::new(String::new());
    let alt_m = RwSignal::new(String::from("0"));
    let start_ns = RwSignal::new(String::new());
    let end_ns = RwSignal::new(String::new());
    let notes = RwSignal::new(String::new());
    let saving = RwSignal::new(false);

    let load = {
        let session_id = session_id.clone();
        move || {
            let session_id = session_id.clone();
            spawn_local(async move {
                match api::fetch_ground_truth(&session_id).await {
                    Ok(list) => events.set(list),
                    Err(e) => error.set(Some(e)),
                }
            });
        }
    };
    load();

    // "Mark now" avoids error-prone wall-clock ↔ nanosecond conversions.
    let mark_now = move |target: RwSignal<String>| {
        let ns = js_sys::Date::now() * 1_000_000.0;
        target.set(format!("{}", ns as u64));
    };

    let on_add = {
        let session_id = session_id.clone();
        let load = load.clone();
        move |_| {
            let session_id = session_id.clone();
            let load = load.clone();
            let parse = |signal: RwSignal<String>| signal.get_untracked().trim().parse::<f64>();
            let (Ok(lat_v), Ok(lon_v), Ok(alt_v), Ok(start_v), Ok(end_v)) = (
                parse(lat),
                parse(lon),
                parse(alt_m),
                parse(start_ns),
                parse(end_ns),
            ) else {
                error.set(Some(
                    "Latitude, longitude, altitude, and start/end times are required numbers."
                        .into(),
                ));
                return;
            };
            let label_v = label.get_untracked().trim().to_string();
            if label_v.is_empty() {
                error.set(Some("Label is required.".into()));
                return;
            }
            if end_v < start_v {
                error.set(Some("End time must be at or after start time.".into()));
                return;
            }
            let event = GroundTruthEventIn {
                label: label_v,
                label_category: category.get_untracked().trim().to_string(),
                lat: lat_v,
                lon: lon_v,
                alt_m: alt_v,
                start_ns: start_v,
                end_ns: end_v,
                notes: {
                    let n = notes.get_untracked().trim().to_string();
                    if n.is_empty() { None } else { Some(n) }
                },
            };
            saving.set(true);
            error.set(None);
            spawn_local(async move {
                match api::add_ground_truth(&session_id, event).await {
                    Ok(_) => {
                        label.set(String::new());
                        notes.set(String::new());
                        load();
                    }
                    Err(e) => error.set(Some(e)),
                }
                saving.set(false);
            });
        }
    };

    view! {
        <div class="ground-truth-panel">
            <div class="ground-truth-header">
                <h4 class="ground-truth-title">
                    <span class="material-symbols-rounded">"my_location"</span>
                    "Ground Truth Events"
                </h4>
                <span class="muted">
                    "Known source position + time window for this capture (static sources)"
                </span>
            </div>

            {move || {
                let list = events.get();
                if list.is_empty() {
                    view! { <p class="muted">"No ground-truth events yet."</p> }.into_any()
                } else {
                    view! {
                        <table class="ground-truth-table">
                            <thead>
                                <tr>
                                    <th>"Label"</th>
                                    <th>"Category"</th>
                                    <th>"Position"</th>
                                    <th>"Window"</th>
                                    <th>"Notes"</th>
                                    <th></th>
                                </tr>
                            </thead>
                            <tbody>
                                {list.into_iter().map(|event| {
                                    let event_id = event.event_id.clone();
                                    let load = load.clone();
                                    let position = match (event.lat, event.lon) {
                                        (Some(lat), Some(lon)) => format!(
                                            "{lat:.6}, {lon:.6} @ {:.0} m",
                                            event.alt_m.unwrap_or(0.0)
                                        ),
                                        _ => "—".into(),
                                    };
                                    let window_s =
                                        ((event.end_ns - event.start_ns) / 1e9).max(0.0);
                                    view! {
                                        <tr>
                                            <td>{event.label.clone()}</td>
                                            <td>{event.label_category.clone()}</td>
                                            <td class="td-mono">{position}</td>
                                            <td class="td-mono">{format!("{window_s:.1} s")}</td>
                                            <td class="ground-truth-notes">
                                                {event.notes.clone().unwrap_or_default()}
                                            </td>
                                            <td>
                                                <button
                                                    class="btn-sm btn-sm--danger"
                                                    on:click=move |_| {
                                                        let event_id = event_id.clone();
                                                        let load = load.clone();
                                                        spawn_local(async move {
                                                            match api::delete_ground_truth(&event_id).await {
                                                                Ok(()) => load(),
                                                                Err(e) => error.set(Some(e)),
                                                            }
                                                        });
                                                    }
                                                >
                                                    "Delete"
                                                </button>
                                            </td>
                                        </tr>
                                    }.into_any()
                                }).collect::<Vec<_>>()}
                            </tbody>
                        </table>
                    }.into_any()
                }
            }}

            <div class="ground-truth-form">
                <div class="ground-truth-form-grid">
                    <label class="gt-field">
                        <span>"Label"</span>
                        <input type="text" class="form-input" placeholder="drone"
                            prop:value=move || label.get()
                            on:input=move |ev| label.set(event_target_value(&ev)) />
                    </label>
                    <label class="gt-field">
                        <span>"Category"</span>
                        <input type="text" class="form-input" placeholder="drone"
                            prop:value=move || category.get()
                            on:input=move |ev| category.set(event_target_value(&ev)) />
                    </label>
                    <label class="gt-field">
                        <span>"Latitude"</span>
                        <input type="text" class="form-input" placeholder="44.9871"
                            prop:value=move || lat.get()
                            on:input=move |ev| lat.set(event_target_value(&ev)) />
                    </label>
                    <label class="gt-field">
                        <span>"Longitude"</span>
                        <input type="text" class="form-input" placeholder="-93.2582"
                            prop:value=move || lon.get()
                            on:input=move |ev| lon.set(event_target_value(&ev)) />
                    </label>
                    <label class="gt-field">
                        <span>"Altitude (m)"</span>
                        <input type="text" class="form-input"
                            prop:value=move || alt_m.get()
                            on:input=move |ev| alt_m.set(event_target_value(&ev)) />
                    </label>
                    <label class="gt-field">
                        <span>"Start (ns)"</span>
                        <div class="gt-time-input">
                            <input type="text" class="form-input"
                                prop:value=move || start_ns.get()
                                on:input=move |ev| start_ns.set(event_target_value(&ev)) />
                            <button class="btn-sm" title="Use the current time"
                                on:click=move |_| mark_now(start_ns)>
                                "now"
                            </button>
                        </div>
                    </label>
                    <label class="gt-field">
                        <span>"End (ns)"</span>
                        <div class="gt-time-input">
                            <input type="text" class="form-input"
                                prop:value=move || end_ns.get()
                                on:input=move |ev| end_ns.set(event_target_value(&ev)) />
                            <button class="btn-sm" title="Use the current time"
                                on:click=move |_| mark_now(end_ns)>
                                "now"
                            </button>
                        </div>
                    </label>
                    <label class="gt-field gt-field--wide">
                        <span>"Notes"</span>
                        <input type="text" class="form-input"
                            placeholder="DJI Mini hover at 30 m AGL"
                            prop:value=move || notes.get()
                            on:input=move |ev| notes.set(event_target_value(&ev)) />
                    </label>
                </div>
                <div class="ground-truth-form-actions">
                    <button class="btn-sm btn-sm--primary"
                        disabled=move || saving.get()
                        on:click=on_add>
                        {move || if saving.get() { "Adding…" } else { "Add event" }}
                    </button>
                </div>
            </div>

            {move || error.get().map(|e| view! { <p class="row-error">{e}</p> })}
        </div>
    }
}

// ── Formatting helpers ────────────────────────────────────────────

fn format_ms(ms: f64) -> String {
    let d = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(ms));
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}",
        d.get_full_year(),
        d.get_month() + 1,
        d.get_date(),
        d.get_hours(),
        d.get_minutes(),
    )
}

fn format_duration(secs: f64) -> String {
    let s = secs as u64;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else {
        format!("{}h {:02}m", s / 3600, (s % 3600) / 60)
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
