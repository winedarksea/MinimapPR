use crate::recording::api;
use crate::recording::{CameraDevice, RecordingStatus, StartRecordingRequest};
use crate::state::AppState;
use futures::StreamExt;
use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

// ── Top-level page ───────────────────────────────────────────────

#[component]
pub fn RecordingPage() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");

    // ── Configuration signals ──
    let selected_node_id = RwSignal::new(String::new());
    let include_iamf = RwSignal::new(false);
    let include_video = RwSignal::new(false);
    // Camera source: populated from API or typed manually.
    let camera_source = RwSignal::new(String::new());
    let cameras: RwSignal<Vec<CameraDevice>> = RwSignal::new(vec![]);
    let cameras_loaded = RwSignal::new(false);

    // ── Session / error state ──
    let session_error: RwSignal<Option<String>> = RwSignal::new(None);

    // Seed selected_node_id from the first available node once nodes load.
    let nodes = state.nodes;
    Effect::new(move |_| {
        let ns = nodes.get();
        if !ns.is_empty() && selected_node_id.get_untracked().is_empty() {
            selected_node_id.set(ns[0].node_id.clone());
        }
    });

    // Load camera list once on mount.
    spawn_local(async move {
        if let Ok(devs) = api::fetch_cameras().await {
            if !devs.is_empty() {
                camera_source.set(devs[0].id.clone());
            }
            cameras.set(devs);
        }
        cameras_loaded.set(true);
    });

    view! {
        <div class="recording-page">
            <div class="recording-config-card">
                <h3 class="section-title">"Spatial Recording"</h3>

                <ListenerSelector selected_node_id />

                <FormatOptions
                    include_iamf
                    include_video
                    camera_source
                    cameras
                    cameras_loaded
                />

                <RecordingControls
                    selected_node_id
                    include_iamf
                    include_video
                    camera_source
                    session_error
                />
            </div>

            <crate::pages::audio::recordings_library::RecordingsLibrary />
        </div>
    }
}

// ── Listener location selector ───────────────────────────────────

#[component]
fn ListenerSelector(selected_node_id: RwSignal<String>) -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let nodes = state.nodes;

    view! {
        <div class="form-row">
            <label class="form-label">"Listener location"</label>
            {move || {
                let ns = nodes.get();
                if ns.is_empty() {
                    view! {
                        <span class="muted">"No nodes available — connect a node to proceed."</span>
                    }.into_any()
                } else {
                    view! {
                        <select
                            class="form-select"
                            on:change=move |ev| {
                                selected_node_id.set(event_target_value(&ev));
                            }
                        >
                            {ns.iter().map(|n| {
                                let id = n.node_id.clone();
                                let label = n.metadata
                                    .as_ref()
                                    .and_then(|_| None::<String>)  // future: node label field
                                    .unwrap_or_else(|| id.clone());
                                                let pos_str = n.position_m.as_ref()
                                    .filter(|p| p.len() >= 2)
                                    .map(|p| format!(" ({:.1}, {:.1})", p[0], p[1]))
                                    .unwrap_or_default();
                                let display = format!("{label}{pos_str}");
                                view! {
                                    <option
                                        value=id.clone()
                                        selected=move || selected_node_id.get() == id
                                    >
                                        {display}
                                    </option>
                                }
                                .into_any()
                            }).collect::<Vec<_>>()}
                        </select>
                    }.into_any()
                }
            }}
        </div>
    }
}

// ── Format options (ambisonics / IAMF / video) ───────────────────

#[component]
fn FormatOptions(
    include_iamf: RwSignal<bool>,
    include_video: RwSignal<bool>,
    camera_source: RwSignal<String>,
    cameras: RwSignal<Vec<CameraDevice>>,
    cameras_loaded: RwSignal<bool>,
) -> impl IntoView {
    view! {
        <div class="form-row">
            <label class="form-label">"Formats"</label>
            <div class="format-checks">
                // Ambisonics is always enabled — it is the base spatial output.
                <label class="check-item check-item--disabled" title="First-order ambisonics B-Format WAV — always recorded">
                    <input type="checkbox" checked disabled />
                    <span>"Ambisonics B-Format"</span>
                    <span class="badge">"WAV"</span>
                </label>

                <label class="check-item" title="Encode B-Format through iamf-tools for object-based IAMF delivery">
                    <input
                        type="checkbox"
                        prop:checked=move || include_iamf.get()
                        on:change=move |ev| include_iamf.set(event_target_checked(&ev))
                    />
                    <span>"IAMF"</span>
                    <span class="badge">"iamf-tools"</span>
                </label>

                <label class="check-item" title="Capture video via ffmpeg for YouTube-ready upload">
                    <input
                        type="checkbox"
                        prop:checked=move || include_video.get()
                        on:change=move |ev| include_video.set(event_target_checked(&ev))
                    />
                    <span>"Video"</span>
                    <span class="badge">"ffmpeg"</span>
                </label>
            </div>
        </div>

        // Camera selector — shown only when video is enabled.
        {move || if include_video.get() {
            view! {
                <CameraSelector
                    camera_source
                    cameras
                    cameras_loaded
                />
            }.into_any()
        } else {
            view! { <></> }.into_any()
        }}
    }
}

// ── Camera source picker ─────────────────────────────────────────

#[component]
fn CameraSelector(
    camera_source: RwSignal<String>,
    cameras: RwSignal<Vec<CameraDevice>>,
    cameras_loaded: RwSignal<bool>,
) -> impl IntoView {
    view! {
        <div class="form-row form-row--indent">
            <label class="form-label">"Camera source"</label>
            {move || {
                let devs = cameras.get();
                if !cameras_loaded.get() {
                    return view! { <span class="muted">"Detecting cameras…"</span> }.into_any();
                }
                if devs.is_empty() {
                    // Backend not available or no cameras found — fall back to text entry.
                    view! {
                        <input
                            type="text"
                            class="form-input"
                            placeholder="Device ID or path (e.g. 0 or /dev/video0)"
                            prop:value=move || camera_source.get()
                            on:input=move |ev| camera_source.set(event_target_value(&ev))
                        />
                    }.into_any()
                } else {
                    view! {
                        <select
                            class="form-select"
                            on:change=move |ev| camera_source.set(event_target_value(&ev))
                        >
                            {devs.iter().map(|d| {
                                let id = d.id.clone();
                                let label = format!("{} ({})", d.label, d.id);
                                view! {
                                    <option
                                        value=id.clone()
                                        selected=move || camera_source.get() == id
                                    >{label}</option>
                                }.into_any()
                            }).collect::<Vec<_>>()}
                        </select>
                    }.into_any()
                }
            }}
        </div>
    }
}

// ── Recording controls (start / stop / status) ───────────────────

#[component]
fn RecordingControls(
    selected_node_id: RwSignal<String>,
    include_iamf: RwSignal<bool>,
    include_video: RwSignal<bool>,
    camera_source: RwSignal<String>,
    session_error: RwSignal<Option<String>>,
) -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let active_recording = state.active_recording;

    // Elapsed seconds ticker — only runs while status is Active.
    let elapsed: RwSignal<u32> = RwSignal::new(0);

    // Start ticker when recording goes Active; reset when session ends.
    Effect::new(move |_| {
        let status = active_recording
            .get()
            .map(|s| s.status.clone())
            .unwrap_or(RecordingStatus::Idle);
        if status == RecordingStatus::Active {
            let elapsed_clone = elapsed;
            spawn_local(async move {
                let mut stream = gloo_timers::future::IntervalStream::new(1_000);
                while stream.next().await.is_some() {
                    // Stop ticking once no longer active.
                    let still_active = active_recording
                        .get_untracked()
                        .map(|s| s.status == RecordingStatus::Active)
                        .unwrap_or(false);
                    if !still_active {
                        break;
                    }
                    elapsed_clone.update(|e| *e += 1);
                }
            });
        } else if !status.is_active() {
            elapsed.set(0);
        }
    });

    let is_busy = move || {
        active_recording
            .get()
            .map(|s| s.status.is_active())
            .unwrap_or(false)
    };

    let can_start = move || !selected_node_id.get().is_empty() && !is_busy();

    let status_label = move || {
        active_recording
            .get()
            .map(|s| s.status.label().to_string())
            .unwrap_or_else(|| "Idle".into())
    };

    let on_start = move |_| {
        let node_id = selected_node_id.get_untracked();
        if node_id.is_empty() {
            session_error.set(Some("Select a listener node first.".into()));
            return;
        }
        session_error.set(None);
        let req = StartRecordingRequest {
            listener_node_id: node_id,
            include_ambisonics: true,
            include_iamf: include_iamf.get_untracked(),
            include_video: include_video.get_untracked(),
            camera_source: {
                let s = camera_source.get_untracked();
                if s.is_empty() { None } else { Some(s) }
            },
        };
        spawn_local(async move {
            match api::start_recording(req).await {
                Ok(session) => {
                    active_recording.set(Some(session));
                    session_error.set(None);
                }
                Err(e) => session_error.set(Some(e)),
            }
        });
    };

    let on_stop = move |_| {
        let session_id = active_recording
            .get_untracked()
            .map(|s| s.session_id.clone())
            .unwrap_or_default();
        if session_id.is_empty() {
            return;
        }
        session_error.set(None);
        spawn_local(async move {
            match api::stop_recording(&session_id).await {
                Ok(session) => {
                    active_recording.set(Some(session));
                    session_error.set(None);
                }
                Err(e) => session_error.set(Some(e)),
            }
        });
    };

    view! {
        <div class="recording-controls">
            {move || {
                let busy = is_busy();
                if busy {
                    let session = active_recording.get().unwrap();
                    let secs = elapsed.get();
                    let elapsed_str = format!(
                        "{:02}:{:02}:{:02}",
                        secs / 3600,
                        (secs % 3600) / 60,
                        secs % 60
                    );
                    view! {
                        <div class="active-session-banner">
                            <span class="rec-dot"></span>
                            <span class="rec-status">{status_label()}</span>
                            <span class="rec-elapsed">{elapsed_str}</span>
                            <div class="rec-format-badges">
                                <span class="badge badge--active">"Ambisonics"</span>
                                {session.include_iamf.then(|| view! {
                                    <span class="badge badge--active">"IAMF"</span>
                                })}
                                {session.include_video.then(|| view! {
                                    <span class="badge badge--active">"Video"</span>
                                })}
                            </div>
                        </div>
                    }.into_any()
                } else {
                    view! { <></> }.into_any()
                }
            }}

            <div class="recording-btn-row">
                {move || if is_busy() {
                    view! {
                        <button class="btn-record btn-record--stop" on:click=on_stop>
                            <span class="material-symbols-rounded">"stop_circle"</span>
                            "Stop Recording"
                        </button>
                    }.into_any()
                } else {
                    view! {
                        <button
                            class="btn-record btn-record--start"
                            disabled=move || !can_start()
                            on:click=on_start
                        >
                            <span class="material-symbols-rounded">"fiber_manual_record"</span>
                            "Start Recording"
                        </button>
                    }.into_any()
                }}
            </div>

            {move || session_error.get().map(|err| view! {
                <p class="recording-error">{err}</p>
            })}
        </div>
    }
}
