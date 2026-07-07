use crate::audio::detection_actions::DetectionAudioActions;
use crate::state::{Alert, AppState, CopItemKind, CopSelection};
use crate::ui::{
    alert_status_badge_class, classify_age_from_ns, cop_sidebar_element_id, is_cop_item_selected,
    severity_badge_class, short_id,
};
use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

#[derive(Clone, Copy, PartialEq, Eq)]
enum AlertFilter {
    Open,
    Sent,
    Escalated,
    Acked,
    Dismissed,
    All,
}

impl AlertFilter {
    fn label(self) -> &'static str {
        match self {
            Self::Open => "Open",
            Self::Sent => "Sent",
            Self::Escalated => "Escalated",
            Self::Acked => "Acked",
            Self::Dismissed => "Dismissed",
            Self::All => "All",
        }
    }
}

fn alert_status(alert: &Alert) -> &str {
    alert.status.as_deref().unwrap_or("sent")
}

fn alert_severity(alert: &Alert) -> &str {
    alert.severity.as_deref().unwrap_or("normal")
}

fn is_actionable_alert(alert: &Alert) -> bool {
    matches!(alert_status(alert), "sent" | "escalated")
}

fn alert_matches_filter(alert: &Alert, filter: AlertFilter) -> bool {
    match filter {
        AlertFilter::Open => is_actionable_alert(alert),
        AlertFilter::Sent => alert_status(alert) == "sent",
        AlertFilter::Escalated => alert_status(alert) == "escalated",
        AlertFilter::Acked => alert_status(alert) == "acknowledged",
        AlertFilter::Dismissed => alert_status(alert) == "dismissed",
        AlertFilter::All => true,
    }
}

fn alert_sort_rank(alert: &Alert) -> (u8, u8, i64) {
    let status_rank = match alert_status(alert) {
        "sent" => 0,
        "escalated" => 1,
        "acknowledged" => 2,
        "dismissed" => 3,
        _ => 4,
    };
    let severity_rank = match alert_severity(alert) {
        "critical" => 0,
        "high" => 1,
        "medium" | "warn" => 2,
        "low" | "normal" => 3,
        _ => 4,
    };
    (
        status_rank,
        severity_rank,
        -alert.triggered_ns.unwrap_or_default(),
    )
}

fn alert_card_severity_class(severity: &str) -> &'static str {
    match severity {
        "critical" => "severity-critical",
        "high" => "severity-high",
        "medium" | "warn" => "severity-medium",
        "low" => "severity-low",
        "normal" => "severity-info",
        _ => "severity-neutral",
    }
}

fn update_alert_status_signal(state: &AppState, alert_id: &str, next_status: String) {
    state.alerts.update(|alerts| {
        if let Some(existing) = alerts.iter_mut().find(|item| item.alert_id == alert_id) {
            existing.status = Some(next_status);
        }
    });
}

#[component]
pub fn AlertsPane() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let alerts = state.alerts;
    let active_filter = RwSignal::new(AlertFilter::Open);

    view! {
        <div class="tab-pane alerts-triage-pane">
            {move || {
                let all_alerts: Vec<_> = alerts.get().into_iter().collect();
                if all_alerts.is_empty() {
                    return view! { <div class="empty-state">"No active alerts"</div> }.into_any();
                }

                let open_count = all_alerts.iter().filter(|alert| is_actionable_alert(alert)).count();
                let critical_count = all_alerts
                    .iter()
                    .filter(|alert| is_actionable_alert(alert) && matches!(alert_severity(alert), "critical" | "high"))
                    .count();
                let critical_counter_class = if critical_count > 0 {
                    "tone-badge danger"
                } else {
                    "tone-badge neutral"
                };
                let selected_filter = active_filter.get();
                let mut filtered_alerts: Vec<_> = all_alerts
                    .into_iter()
                    .filter(|alert| alert_matches_filter(alert, selected_filter))
                    .collect();
                filtered_alerts.sort_by_key(alert_sort_rank);

                view! {
                    <div class="alert-triage-toolbar">
                        <div class="alert-triage-counters">
                            <span class="tone-badge warn">{format!("{open_count} open")}</span>
                            <span class=critical_counter_class>
                                {format!("{critical_count} critical/high")}
                            </span>
                        </div>
                        <div class="alert-filter-row" role="tablist" aria-label="Alert filters">
                            <AlertFilterButton filter=AlertFilter::Open active_filter />
                            <AlertFilterButton filter=AlertFilter::Sent active_filter />
                            <AlertFilterButton filter=AlertFilter::Escalated active_filter />
                            <AlertFilterButton filter=AlertFilter::Acked active_filter />
                            <AlertFilterButton filter=AlertFilter::Dismissed active_filter />
                            <AlertFilterButton filter=AlertFilter::All active_filter />
                        </div>
                    </div>

                    {if filtered_alerts.is_empty() {
                        view! { <div class="empty-state">"No alerts match this filter"</div> }.into_any()
                    } else {
                        view! {
                            <div class="alert-stack">
                                {filtered_alerts.into_iter().map(|alert| view! { <AlertCard alert /> }).collect_view()}
                            </div>
                        }.into_any()
                    }}
                }.into_any()
            }}
        </div>
    }
}

#[component]
fn AlertFilterButton(filter: AlertFilter, active_filter: RwSignal<AlertFilter>) -> impl IntoView {
    view! {
        <button
            type="button"
            class=move || {
                if active_filter.get() == filter {
                    "btn-sm active"
                } else {
                    "btn-sm"
                }
            }
            aria-selected=move || (active_filter.get() == filter).to_string()
            on:click=move |_| active_filter.set(filter)
        >
            {filter.label()}
        </button>
    }
}

#[component]
fn AlertCard(alert: Alert) -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let selected_cop_item = state.selected_cop_item;
    let is_pending = RwSignal::new(false);
    let error_message = RwSignal::new(None::<String>);

    let rule_name = alert
        .rule_name
        .clone()
        .unwrap_or_else(|| "Unnamed rule".to_string());
    let message = alert
        .message
        .clone()
        .unwrap_or_else(|| "No operator detail".to_string());
    let severity = alert_severity(&alert).to_string();
    let status = alert_status(&alert).to_string();
    let (age_text, age_class) = classify_age_from_ns(alert.triggered_ns, 60.0, 300.0);
    let severity_class = severity_badge_class(&severity);
    let status_class = alert_status_badge_class(&status);
    let card_severity_class = alert_card_severity_class(&severity);
    let alert_id = alert.alert_id.clone();
    let hover_id = alert_id.clone();
    let leave_id = alert_id.clone();
    let click_id = alert_id.clone();
    let row_id = alert_id.clone();
    let row_element_id = cop_sidebar_element_id(CopItemKind::Alert, &alert_id);
    let is_actionable = is_actionable_alert(&alert);
    let detection_id = alert.detection_id.clone();
    let track_id = alert.track_id.clone();
    let destination = alert
        .destination
        .clone()
        .unwrap_or_else(|| "cop".to_string());
    let payload_summary =
        if alert.payload.is_object() && !alert.payload.as_object().unwrap().is_empty() {
            serde_json::to_string(&alert.payload).unwrap_or_else(|_| "{}".to_string())
        } else {
            String::new()
        };

    let patch_status = move |next_status: &'static str, reason: Option<&'static str>| {
        if !is_actionable || is_pending.get_untracked() {
            return;
        }

        let state = state.clone();
        let alert_id = alert_id.clone();
        is_pending.set(true);
        error_message.set(None);

        spawn_local(async move {
            match crate::api::patch_alert_status(&alert_id, next_status, reason).await {
                Ok(next_status) => update_alert_status_signal(&state, &alert_id, next_status),
                Err(error) => error_message.set(Some(error)),
            }
            is_pending.set(false);
        });
    };

    let acknowledge = {
        let patch_status = patch_status.clone();
        move |_| patch_status("acknowledged", None)
    };
    let dismiss = {
        let patch_status = patch_status.clone();
        move |_| patch_status("dismissed", Some("dismissed from COP triage"))
    };
    let escalate = move |_| patch_status("escalated", Some("operator escalation from COP triage"));

    view! {
        <article
            id=row_element_id
            class=move || format!(
                "alert-card {card_severity_class}{}{}",
                if is_actionable { " alert-live" } else { "" },
                if is_cop_item_selected(&selected_cop_item.get(), CopItemKind::Alert, &row_id) {
                    " cop-row-selected"
                } else {
                    ""
                },
            )
            on:mouseenter=move |_| {
                selected_cop_item.set(Some(CopSelection::hovered(
                    CopItemKind::Alert,
                    hover_id.clone(),
                )));
            }
            on:click=move |_| {
                selected_cop_item.set(Some(CopSelection::pinned(
                    CopItemKind::Alert,
                    click_id.clone(),
                )));
            }
            on:mouseleave={
                let leave_id = leave_id.clone();
                move |_| {
                    let should_clear = selected_cop_item
                        .get_untracked()
                        .as_ref()
                        .is_some_and(|selected| {
                            selected.kind == CopItemKind::Alert
                                && selected.id == leave_id
                                && !selected.pinned
                        });
                    if should_clear {
                        selected_cop_item.set(None);
                    }
                }
            }
        >
            <header class="alert-card-head">
                <div class="alert-card-titleblock">
                    <div class="alert-card-title">{rule_name}</div>
                    <div class="alert-card-age">
                        <span class=age_class>{age_text}</span>
                        <span class="alert-card-destination">{destination}</span>
                    </div>
                </div>
                <div class="alert-card-badges">
                    <span class=severity_class>{severity.clone()}</span>
                    <span class=status_class>{status.clone()}</span>
                </div>
            </header>

            <p class="alert-card-message">{message}</p>

            <div class="alert-evidence-grid">
                <EvidencePivots detection_id=detection_id.clone() track_id=track_id.clone() />
                <div class="alert-card-id-block">
                    <span>"Alert"</span>
                    <code class="alert-card-id">{short_id(&alert.alert_id, 18)}</code>
                </div>
            </div>

            {if payload_summary.is_empty() {
                ().into_any()
            } else {
                view! {
                    <details class="alert-payload">
                        <summary>"Payload"</summary>
                        <code>{payload_summary}</code>
                    </details>
                }.into_any()
            }}

            <div class="alert-card-actions">
                <button
                    class="btn-sm btn-ack"
                    disabled=move || !is_actionable || is_pending.get()
                    on:click=acknowledge
                >
                    {move || if is_pending.get() { "Working" } else { "Ack" }}
                </button>
                <button
                    class="btn-sm"
                    disabled=move || !is_actionable || is_pending.get()
                    on:click=dismiss
                >
                    "Dismiss"
                </button>
                <button
                    class="btn-sm btn-danger"
                    disabled=move || !is_actionable || is_pending.get()
                    on:click=escalate
                >
                    "Escalate"
                </button>
            </div>

            {move || match error_message.get() {
                Some(error) => view! { <div class="alert-card-error">{error}</div> }.into_any(),
                None => ().into_any(),
            }}
        </article>
    }
}

#[component]
fn EvidencePivots(detection_id: Option<String>, track_id: Option<String>) -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let selected_cop_item = state.selected_cop_item;

    let detection_button = detection_id.clone().map(|id| {
        let select_id = id.clone();
        view! {
            <button
                type="button"
                class="alert-evidence-chip"
                on:click=move |event| {
                    event.stop_propagation();
                    selected_cop_item.set(Some(CopSelection::pinned(
                        CopItemKind::Detection,
                        select_id.clone(),
                    )));
                }
            >
                <span>"Detection"</span>
                <code>{short_id(&id, 12)}</code>
            </button>
        }
    });

    let track_button = track_id.clone().map(|id| {
        let select_id = id.clone();
        view! {
            <button
                type="button"
                class="alert-evidence-chip"
                on:click=move |event| {
                    event.stop_propagation();
                    selected_cop_item.set(Some(CopSelection::pinned(
                        CopItemKind::Track,
                        select_id.clone(),
                    )));
                }
            >
                <span>"Track"</span>
                <code>{short_id(&id, 12)}</code>
            </button>
        }
    });

    view! {
        <div class="alert-evidence-pivots">
            {detection_button}
            {track_button}
            {if detection_id.is_none() && track_id.is_none() {
                view! { <span class="alert-evidence-missing">"No trigger reference in alert payload"</span> }.into_any()
            } else {
                ().into_any()
            }}
            {detection_id.map(|id| view! {
                <div class="alert-audio-actions" on:click=move |event| event.stop_propagation()>
                    <DetectionAudioActions event_id=id />
                </div>
            })}
        </div>
    }
}
