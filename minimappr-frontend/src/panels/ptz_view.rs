use crate::api::snapshot_ptz_node;
use futures::StreamExt;
use gloo_timers::future::IntervalStream;
use leptos::prelude::*;
use leptos::task::spawn_local;

/// Snapshot-refresh live view for a PTZ-capable node: an `<img>` whose `src` is
/// re-fetched on a timer (no HLS/WebRTC in v1 — see plan scope boundary),
/// plus a "Capture still" button that persists + links an evidence artifact.
#[component]
pub fn PtzLiveView(
    node_id: String,
    track_id: Option<String>,
    #[prop(into)] on_close: Callback<()>,
) -> impl IntoView {
    let refresh_tick: RwSignal<f64> = RwSignal::new(js_sys::Date::now());
    let capturing = RwSignal::new(false);
    let capture_error: RwSignal<Option<String>> = RwSignal::new(None);
    let captured_artifact_id: RwSignal<Option<String>> = RwSignal::new(None);

    Effect::new(move |_| {
        spawn_local(async move {
            let mut interval = IntervalStream::new(1_000);
            while interval.next().await.is_some() {
                refresh_tick.set(js_sys::Date::now());
            }
        });
    });

    let img_src = {
        let node_id = node_id.clone();
        move || {
            format!(
                "/api/v1/nodes/{}/effector/snapshot.jpg?t={}",
                js_sys::encode_uri_component(&node_id)
                    .as_string()
                    .unwrap_or_default(),
                refresh_tick.get()
            )
        }
    };

    let on_capture = {
        let node_id = node_id.clone();
        let track_id = track_id.clone();
        move |_| {
            let node_id = node_id.clone();
            let track_id = track_id.clone();
            capturing.set(true);
            capture_error.set(None);
            captured_artifact_id.set(None);
            spawn_local(async move {
                match snapshot_ptz_node(&node_id, track_id.as_deref()).await {
                    Ok(artifact_id) => {
                        captured_artifact_id.set(Some(artifact_id));
                    }
                    Err(e) => capture_error.set(Some(e)),
                }
                capturing.set(false);
            });
        }
    };

    view! {
        <div class="effector-live-view">
            <div class="effector-live-view-header">
                <span class="muted" style="font-size:0.8rem">{format!("Camera: {}", node_id)}</span>
                <button class="btn-sm" title="Close live view" on:click=move |_| on_close.run(())>
                    "✕"
                </button>
            </div>
            <img class="effector-live-view-img" src=img_src alt="Live camera view" />
            <div class="effector-live-view-actions">
                <button class="btn-sm" disabled=move || capturing.get() on:click=on_capture>
                    {move || if capturing.get() { "Capturing…" } else { "Capture still" }}
                </button>
                {move || captured_artifact_id.get().map(|id| view! {
                    <span style="color:var(--mmp-sys-color-ok);font-size:0.8rem">
                        {format!("Saved (artifact {})", id)}
                    </span>
                })}
                {move || capture_error.get().map(|e| view! {
                    <span class="daily-error">{e}</span>
                })}
            </div>
        </div>
    }
}
