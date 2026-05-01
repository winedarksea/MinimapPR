use crate::recording::api;
use crate::recording::{RecordingLibraryEntry, RecordingStatus};
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

    let ambi_url = api::download_url(&entry.session_id, "ambisonics");
    let iamf_url = api::download_url(&entry.session_id, "iamf");
    let video_url = api::download_url(&entry.session_id, "video");
    let show_iamf = entry.iamf_available;
    let show_video = entry.video_available;
    // Clone session_id for use in the reactive delete closure.
    let sid_for_delete = entry.session_id.clone();

    view! {
        <tr>
            <td class="td-mono">{started}</td>
            <td class="td-mono">{duration}</td>
            <td class="td-node">{listener}</td>
            <td>
                <div class="format-badges">
                    {entry.ambisonics_available.then(|| view! {
                        <span class="badge">"Ambi"</span>
                    })}
                    {show_iamf.then(|| view! {
                        <span class="badge">"IAMF"</span>
                    })}
                    {show_video.then(|| view! {
                        <span class="badge">"Video"</span>
                    })}
                    {(!is_completed).then(|| view! {
                        <span class="badge badge--dim">{entry.status.label()}</span>
                    })}
                </div>
            </td>
            <td class="td-mono">{size}</td>
            <td>
                <div class="row-actions">
                    {is_completed.then({
                        let ambi_url = ambi_url.clone();
                        move || view! {
                            <a class="btn-sm" href=ambi_url.clone() download>
                                "↓ Ambi"
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
