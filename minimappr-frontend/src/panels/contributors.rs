use crate::state::ContributorSummary;
use crate::ui::short_id;
use leptos::prelude::*;

fn contributor_role_tag(roles: &[String]) -> &'static str {
    if roles.iter().any(|role| role == "source") {
        "SRC"
    } else if roles.iter().any(|role| role == "reference") {
        "REF"
    } else if roles.iter().any(|role| role == "tdoa") {
        "TDOA"
    } else {
        "SUP"
    }
}

fn contributor_tooltip(contributor: &ContributorSummary) -> String {
    let roles = if contributor.roles.is_empty() {
        "roles: inferred".to_string()
    } else {
        format!("roles: {}", contributor.roles.join(", "))
    };
    let sensors = if contributor.sensor_ids.is_empty() {
        String::new()
    } else {
        format!("\nsensors: {}", contributor.sensor_ids.join(", "))
    };
    format!("{}\n{}{}", contributor.node_id, roles, sensors)
}

#[component]
pub fn CompactContributorChips(
    contributors: Vec<ContributorSummary>,
    fallback_node_id: Option<String>,
) -> impl IntoView {
    let mut display_contributors = contributors;
    if display_contributors.is_empty() {
        if let Some(node_id) = fallback_node_id {
            display_contributors.push(ContributorSummary {
                node_id,
                roles: vec!["source".to_string()],
                sensor_ids: vec![],
                contribution_count: 1,
                last_contributed_ns: None,
            });
        }
    }

    if display_contributors.is_empty() {
        return view! { <span class="age-unknown">"-"</span> }.into_any();
    }

    let overflow_count = display_contributors.len().saturating_sub(2);
    let visible_contributors: Vec<_> = display_contributors.into_iter().take(2).collect();

    view! {
        <div class="contributor-list">
            {visible_contributors.into_iter().map(|contributor| {
                let tooltip = contributor_tooltip(&contributor);
                let role_tag = contributor_role_tag(&contributor.roles).to_string();
                let node_label = short_id(&contributor.node_id, 10);
                view! {
                    <span class="contributor-chip-pair" title=tooltip>
                        <code class="node-chip contributor-chip">{node_label}</code>
                        <span class="contributor-role-pill">{role_tag}</span>
                    </span>
                }
            }).collect_view()}
            {if overflow_count > 0 {
                view! {
                    <span class="tone-badge neutral contributor-overflow">
                        {format!("+{overflow_count}")}
                    </span>
                }.into_any()
            } else {
                ().into_any()
            }}
        </div>
    }
    .into_any()
}
