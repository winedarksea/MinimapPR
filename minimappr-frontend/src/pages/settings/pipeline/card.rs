//! DAG card chip + selected-stage detail panel. Selecting a card never
//! changes its geometry (see layout.rs docstring) — it only updates the
//! `DetailPanel` content, so the SVG edges never need recomputation on click.

use super::layout::CardPosition;
use super::types::PipelineGraphNode;
use crate::components::icons::ArrayTypeIcon;
use leptos::prelude::*;
use leptos_router::components::A;

/// Health value → CSS class modifier, matching `.pipeline-dag-card.<mod>`.
fn health_class(health: &str) -> &'static str {
    match health {
        "ok" => "ok",
        "warn" => "warn",
        "danger" => "danger",
        "idle" => "idle",
        "off" => "off",
        _ => "unknown",
    }
}

#[component]
pub fn StageCard(
    node: PipelineGraphNode,
    position: CardPosition,
    selected: RwSignal<Option<String>>,
) -> impl IntoView {
    let node_id = node.id.clone();
    let is_selected = {
        let node_id = node_id.clone();
        move || selected.get().as_deref() == Some(node_id.as_str())
    };
    let health = health_class(&node.status.health);
    let style = format!(
        "left:{}px; top:{}px; width:{}px; height:{}px;",
        position.x, position.y, position.width, position.height
    );
    let enabled_cls = if node.enabled { "" } else { " disabled" };
    let selected_cls = move || if is_selected() { " selected" } else { "" };
    let onclick = {
        let node_id = node_id.clone();
        move |_| {
            selected.update(|cur| {
                *cur = if cur.as_deref() == Some(node_id.as_str()) {
                    None
                } else {
                    Some(node_id.clone())
                };
            });
        }
    };

    view! {
        <button
            type="button"
            class=move || format!("pipeline-dag-card {}{}{}", health, enabled_cls, selected_cls())
            style=style
            on:click=onclick
        >
            <span class="pipeline-dag-card-icon">
                {node.node_type.clone().map(|nt| view! { <ArrayTypeIcon node_type=nt size=16 /> })}
            </span>
            <span class="pipeline-dag-card-body">
                <strong>{node.title.clone()}</strong>
                <span class="pipeline-dag-card-subtitle">{node.subtitle.clone()}</span>
            </span>
            <span class=format!("pipeline-dag-health-dot {health}")></span>
        </button>
    }
}

#[component]
pub fn DetailPanel(node: PipelineGraphNode) -> impl IntoView {
    let health = health_class(&node.status.health);
    view! {
        <div class="pipeline-dag-detail">
            <div class="pipeline-dag-detail-header">
                <h3>{node.title.clone()}</h3>
                <span class=format!("health-chip {health}")>{node.status.health.clone()}</span>
            </div>
            <p class="muted">{node.subtitle.clone()}</p>
            {(!node.status.summary.is_empty()).then(|| view! {
                <p class="pipeline-dag-detail-summary">{node.status.summary.clone()}</p>
            })}

            <h4>"Parameters"</h4>
            <dl class="pipeline-dag-param-list">
                {node.params.iter().map(|p| {
                    let key = p.config_key.clone();
                    view! {
                        <dt>{p.label.clone()}</dt>
                        <dd>
                            {p.value.clone()}
                            {key.map(|k| view! {
                                " "
                                <A href=format!("/settings/config#{k}") attr:class="pipeline-dag-deep-link">"⚙"</A>
                            })}
                        </dd>
                    }
                }).collect_view()}
            </dl>

            {(!node.status.metrics.is_empty()).then({
                let metrics = node.status.metrics.clone();
                move || view! {
                    <h4>"Live metrics"</h4>
                    <dl class="pipeline-dag-param-list">
                        {metrics.iter().map(|p| view! {
                            <dt>{p.label.clone()}</dt>
                            <dd>{p.value.clone()}</dd>
                        }).collect_view()}
                    </dl>
                }
            })}

            {node.link.clone().map(|link| view! {
                <A href=link attr:class="pipeline-dag-deep-link-primary">"Open →"</A>
            })}
        </div>
    }
}
