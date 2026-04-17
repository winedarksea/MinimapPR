use crate::state::AppState;
use leptos::prelude::*;

fn severity_color(sev: &str) -> &'static str {
    match sev {
        "critical" | "high" => "var(--danger)",
        "medium"   | "warn" => "var(--warn)",
        _                   => "var(--text-muted)",
    }
}

#[component]
pub fn AlertsPane() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let alerts = state.alerts;

    view! {
        <div class="tab-pane">
            {move || {
                let als: Vec<_> = alerts.get().into_iter().collect();
                if als.is_empty() {
                    return view! { <div class="empty-state">"No active alerts"</div> }.into_any();
                }
                view! {
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>"Rule"</th>
                                <th>"Message"</th>
                                <th>"Severity"</th>
                                <th>"Status"</th>
                            </tr>
                        </thead>
                        <tbody>
                            {als.into_iter().map(|a| {
                                let rule    = a.rule_name.clone().unwrap_or_else(|| "—".to_string());
                                let msg     = a.message.clone().unwrap_or_else(|| "—".to_string());
                                let sev     = a.severity.clone().unwrap_or_default();
                                let color   = severity_color(&sev);
                                let status  = a.status.clone().unwrap_or_else(|| "—".to_string());

                                view! {
                                    <tr>
                                        <td style="font-weight:700">{rule}</td>
                                        <td>{msg}</td>
                                        <td style=format!("color:{color};font-weight:700")>{sev}</td>
                                        <td>{status}</td>
                                    </tr>
                                }
                            }).collect_view()}
                        </tbody>
                    </table>
                }.into_any()
            }}
        </div>
    }
}
