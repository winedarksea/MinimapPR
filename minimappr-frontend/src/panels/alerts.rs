use crate::state::{AppState, CopItemKind, CopSelection};
use crate::ui::{
    alert_status_badge_class, classify_age_from_ns, cop_sidebar_element_id, is_cop_item_selected,
    severity_badge_class,
};
use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

fn alert_sort_rank(alert: &crate::state::Alert) -> (u8, u8, i64) {
    let status_rank = match alert.status.as_deref() {
        Some("sent") => 0,
        Some("escalated") => 1,
        Some("acknowledged") => 2,
        Some("dismissed") => 3,
        _ => 4,
    };
    let severity_rank = match alert.severity.as_deref() {
        Some("critical") => 0,
        Some("high") => 1,
        Some("medium") | Some("warn") => 2,
        Some("low") | Some("normal") => 3,
        _ => 4,
    };
    let triggered_rank = -alert.triggered_ns.unwrap_or_default();
    (status_rank, severity_rank, triggered_rank)
}

#[component]
pub fn AlertsPane() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let alerts = state.alerts;

    view! {
        <div class="tab-pane">
            {move || {
                let mut als: Vec<_> = alerts.get().into_iter().collect();
                if als.is_empty() {
                    return view! { <div class="empty-state">"No active alerts"</div> }.into_any();
                }
                als.sort_by_key(alert_sort_rank);
                view! {
                    <div class="alert-stack">
                        {als.into_iter().map(|alert| view! { <AlertCard alert /> }).collect_view()}
                    </div>
                }.into_any()
            }}
        </div>
    }
}

#[component]
fn AlertCard(alert: crate::state::Alert) -> impl IntoView {
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
    let severity = alert
        .severity
        .clone()
        .unwrap_or_else(|| "normal".to_string());
    let status = alert.status.clone().unwrap_or_else(|| "sent".to_string());
    let (age_text, age_class) = classify_age_from_ns(alert.triggered_ns, 60.0, 300.0);
    let severity_class = severity_badge_class(&severity);
    let status_class = alert_status_badge_class(&status);
    let alert_id = alert.alert_id.clone();
    let hover_id = alert_id.clone();
    let leave_id = alert_id.clone();
    let click_id = alert_id.clone();
    let row_id = alert_id.clone();
    let row_element_id = cop_sidebar_element_id(CopItemKind::Alert, &alert_id);
    let is_actionable = status == "sent" || status == "escalated";

    let on_acknowledge = move |_| {
        if !is_actionable || is_pending.get_untracked() {
            return;
        }

        let state = state.clone();
        let alert_id = alert_id.clone();
        is_pending.set(true);
        error_message.set(None);

        spawn_local(async move {
            match crate::api::patch_alert_status(&alert_id, "acknowledged", None).await {
                Ok(next_status) => {
                    state.alerts.update(|alerts| {
                        if let Some(existing) =
                            alerts.iter_mut().find(|item| item.alert_id == alert_id)
                        {
                            existing.status = Some(next_status);
                        }
                    });
                }
                Err(error) => {
                    error_message.set(Some(error));
                }
            }
            is_pending.set(false);
        });
    };

    view! {
        <article
            id=row_element_id
            class=move || format!(
                "alert-card {} {}{}{}",
                severity_class.replace("tone-badge ", "severity-"),
                status_class.replace("tone-badge ", "status-"),
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
            <details class="alert-details" open=is_actionable>
                <summary class="alert-card-head">
                    <div class="alert-card-titleblock">
                        <div class="alert-card-title">{rule_name}</div>
                        <div class="alert-card-age">
                            <span class=age_class>{age_text}</span>
                        </div>
                    </div>
                    <div class="alert-card-badges">
                        <span class=severity_class>{severity}</span>
                        <span class=status_class>{status}</span>
                        <span class="row-chevron" aria-hidden="true">"▾"</span>
                    </div>
                </summary>

                <p class="alert-card-message">{message}</p>

                <div class="alert-card-actions">
                    <button
                        class="btn-sm btn-ack"
                        disabled=move || !is_actionable || is_pending.get()
                        on:click=on_acknowledge
                    >
                        {move || if is_pending.get() { "Acknowledging…" } else { "Acknowledge" }}
                    </button>
                    <code class="alert-card-id">{alert.alert_id.clone()}</code>
                </div>

                {move || match error_message.get() {
                    Some(error) => view! { <div class="alert-card-error">{error}</div> }.into_any(),
                    None => ().into_any(),
                }}
            </details>
        </article>
    }
}
