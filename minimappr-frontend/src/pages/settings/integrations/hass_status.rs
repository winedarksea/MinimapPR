//! Live bridge connection state, counters, and the two recovery actions.
//!
//! The server pushes `hass_status` on transition only, so this also polls while
//! mounted — otherwise the panel would show stale state whenever the websocket
//! is down, which is exactly when an operator opens it.

use crate::api::{get_hass_status, hass_purge_discovery, hass_republish_discovery};
use crate::state::AppState;
use futures::future::{AbortHandle, Abortable};
use futures::StreamExt;
use gloo_timers::future::IntervalStream;
use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

const POLL_INTERVAL_MS: u32 = 5_000;

/// Maps connection state onto the shared tone-badge vocabulary.
fn state_tone(connection_state: &str) -> &'static str {
    match connection_state {
        "connected" => "ok",
        "connecting" => "warn",
        "error" => "bad",
        _ => "neutral",
    }
}

fn state_label(connection_state: &str) -> String {
    match connection_state {
        "connected" => "connected".to_string(),
        "connecting" => "connecting".to_string(),
        "disconnected" => "disconnected".to_string(),
        "error" => "error".to_string(),
        "disabled" => "disabled".to_string(),
        other => other.to_string(),
    }
}

#[component]
pub fn HassStatusCard() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let status = state.hass_status;
    let action_pending = RwSignal::new(false);
    let purge_armed = RwSignal::new(false);
    let message = RwSignal::new(None::<String>);
    let error = RwSignal::new(None::<String>);

    // The server pushes `hass_status` only on transition, so poll while mounted —
    // otherwise the panel is stale exactly when the websocket is down, which is
    // when an operator is most likely to be looking at it. Abortable + on_cleanup
    // so navigating away stops the polling (matching pages/audio/recording.rs).
    let refresh = move || {
        spawn_local(async move {
            match get_hass_status().await {
                Ok(value) => status.set(Some(value)),
                Err(reason) => error.set(Some(reason)),
            }
        });
    };
    refresh();

    let (poll_abort_handle, poll_abort_registration) = AbortHandle::new_pair();
    spawn_local(async move {
        let mut stream = IntervalStream::new(POLL_INTERVAL_MS);
        let poll_loop = async move {
            while stream.next().await.is_some() {
                refresh();
            }
        };
        let _ = Abortable::new(poll_loop, poll_abort_registration).await;
    });
    on_cleanup(move || poll_abort_handle.abort());

    let republish = move |_| {
        if action_pending.get_untracked() {
            return;
        }
        error.set(None);
        message.set(None);
        action_pending.set(true);
        spawn_local(async move {
            match hass_republish_discovery().await {
                Ok(()) => message.set(Some(
                    "Discovery will be republished on the next publish cycle".to_string(),
                )),
                Err(reason) => error.set(Some(reason)),
            }
            action_pending.set(false);
        });
    };

    let purge = move |_| {
        if action_pending.get_untracked() {
            return;
        }
        // Two-click confirm: this removes every entity from Home Assistant.
        if !purge_armed.get_untracked() {
            purge_armed.set(true);
            message.set(Some(
                "Click again to remove every MinimapPR entity from Home Assistant".to_string(),
            ));
            return;
        }
        purge_armed.set(false);
        error.set(None);
        message.set(None);
        action_pending.set(true);
        spawn_local(async move {
            match hass_purge_discovery().await {
                Ok(()) => message.set(Some("Retained discovery and state cleared".to_string())),
                Err(reason) => error.set(Some(reason)),
            }
            action_pending.set(false);
        });
    };

    view! {
        <section class="card integration-card">
            <div class="settings-card-head">
                <div>
                    <h2>"Bridge status"</h2>
                    <p class="muted">
                        {move || {
                            status.get().map(|value| {
                                if value.mqtt_host.is_empty() {
                                    "No broker configured.".to_string()
                                } else {
                                    format!(
                                        "{}:{} · base topic {} · device {}",
                                        value.mqtt_host,
                                        value.mqtt_port,
                                        value.base_topic,
                                        value.device_id,
                                    )
                                }
                            })
                        }}
                    </p>
                </div>
                {move || {
                    let connection_state = status
                        .get()
                        .map(|value| value.connection_state)
                        .unwrap_or_else(|| "disabled".to_string());
                    let tone = state_tone(&connection_state);
                    view! { <span class=format!("tone-badge {tone}")>{state_label(&connection_state)}</span> }
                }}
            </div>

            {move || {
                status
                    .get()
                    .and_then(|value| value.last_connect_error)
                    .map(|reason| view! { <p class="daily-error">{reason}</p> })
            }}
            {move || {
                status.get().filter(|value| !value.transport_available).map(|_| view! {
                    <p class="muted">
                        "The optional aiomqtt package is not installed, so nothing can be published. Install the 'hass' extra: pip install -e '.[hass]'"
                    </p>
                })
            }}

            <div class="integration-output-grid">
                {move || {
                    let Some(value) = status.get() else {
                        return Vec::new();
                    };
                    let mut rows = vec![
                        ("Discovery entities".to_string(), value.discovery_entity_count.to_string()),
                        (
                            "Retained state topics".to_string(),
                            value.published_state_topic_count.to_string(),
                        ),
                        (
                            "Queue".to_string(),
                            format!("{} / {}", value.queue_depth, value.queue_capacity),
                        ),
                        (
                            "Transport".to_string(),
                            value.transport.clone().unwrap_or_else(|| "none".to_string()),
                        ),
                    ];
                    // Counters come straight from the backend metrics map so a new
                    // counter shows up here without a frontend change.
                    for (name, count) in value.metrics {
                        rows.push((name.replace('_', " "), count.to_string()));
                    }
                    rows
                        .into_iter()
                        .map(|(label, count)| {
                            view! { <div><strong>{label}</strong><span>{count}</span></div> }
                        })
                        .collect::<Vec<_>>()
                }}
            </div>

            <div class="overlay-card-actions">
                <button
                    class="btn-sm"
                    disabled=move || action_pending.get()
                    on:click=republish
                    title="Re-send every discovery config and a full state snapshot"
                >
                    "Republish discovery"
                </button>
                <button
                    class="btn-sm"
                    disabled=move || action_pending.get()
                    on:click=purge
                    title="Remove every MinimapPR entity from Home Assistant"
                >
                    {move || if purge_armed.get() { "Confirm purge" } else { "Purge discovery" }}
                </button>
            </div>
            {move || message.get().map(|text| view! { <span class="review-status-ok">{text}</span> })}
            {move || error.get().map(|text| view! { <span class="daily-error">{text}</span> })}
        </section>
    }
}
