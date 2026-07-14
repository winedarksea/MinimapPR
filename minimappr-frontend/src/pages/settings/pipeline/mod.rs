//! Read-only Pipeline Flow DAG (`/settings/pipeline`): stage columns × node
//! lanes, SVG edges, deep links into node/rules/config editors. Structure
//! comes from configuration; status overlays refresh on a ~2.5 s poll while
//! the page is open (see plan `we-need-to-add-merry-dawn.md`).

mod card;
mod layout;
mod types;

use card::{DetailPanel, StageCard};
use futures::StreamExt;
use gloo_net::http::Request;
use gloo_timers::future::IntervalStream;
use layout::compute_layout;
use leptos::prelude::*;
use leptos::task::spawn_local;
use types::PipelineGraph;

const POLL_MS: u32 = 2_500;

#[component]
pub fn PipelineGraphView() -> impl IntoView {
    let graph: RwSignal<Option<PipelineGraph>> = RwSignal::new(None);
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let selected: RwSignal<Option<String>> = RwSignal::new(None);

    let fetch = move || {
        spawn_local(async move {
            match Request::get("/api/v1/pipeline/graph").send().await {
                Ok(resp) if resp.ok() => match resp.json::<PipelineGraph>().await {
                    Ok(g) => {
                        graph.set(Some(g));
                        error.set(None);
                    }
                    Err(e) => error.set(Some(format!("parse: {e}"))),
                },
                Ok(resp) => error.set(Some(format!("HTTP {}", resp.status()))),
                Err(e) => error.set(Some(e.to_string())),
            }
        });
    };

    Effect::new(move |_| fetch());
    Effect::new(move |_| {
        spawn_local(async move {
            let mut iv = IntervalStream::new(POLL_MS);
            while iv.next().await.is_some() {
                fetch();
            }
        });
    });

    let selected_node = move || {
        graph.get().and_then(|g| {
            selected
                .get()
                .and_then(|id| g.nodes.iter().find(|n| n.id == id).cloned())
        })
    };

    view! {
        <div class="pipeline-dag-page">
            <div class="pipeline-header">
                <h2>"Pipeline Flow"</h2>
                {move || error.get().map(|e| view! { <span class="daily-error">{e}</span> })}
            </div>

            {move || match graph.get() {
                None => view! { <p class="muted">"Loading…"</p> }.into_any(),
                Some(g) => {
                    let fusion_available = g.fusion_available;
                    view! {
                        <div>
                            {(!fusion_available).then(|| view! {
                                <div class="pipeline-banner-warn pipeline-health-banner">
                                    "Fusion pipeline unavailable — showing configured structure only."
                                </div>
                            })}
                            <PipelineDagCanvas graph=g selected=selected />
                            {move || selected_node().map(|n| view! { <DetailPanel node=n /> })}
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}

#[component]
fn PipelineDagCanvas(graph: PipelineGraph, selected: RwSignal<Option<String>>) -> impl IntoView {
    let layout = compute_layout(&graph);
    let width = layout.width.max(1.0);
    let height = layout.height.max(1.0);
    let container_style = format!("position:relative; width:{width}px; height:{height}px;");
    let svg_style = "position:absolute; top:0; left:0; pointer-events:none;".to_string();

    let positions_by_id: std::collections::HashMap<String, layout::CardPosition> = layout
        .positions
        .iter()
        .cloned()
        .map(|p| (p.id.clone(), p))
        .collect();

    let columns_header = graph
        .columns
        .iter()
        .map(|c| {
            let x = (c.order as f64) * (layout::COLUMN_WIDTH + layout::COLUMN_GUTTER);
            let style = format!("left:{x}px; width:{}px;", layout::COLUMN_WIDTH);
            view! { <div class="pipeline-dag-column-header" style=style>{c.title.clone()}</div> }
        })
        .collect_view();

    let edges = layout
        .edge_paths
        .iter()
        .map(|e| {
            let cls = format!(
                "pipeline-dag-edge kind-{}{}",
                e.kind,
                if e.active { "" } else { " inactive" }
            );
            view! { <path class=cls d=e.d.clone() fill="none" /> }
        })
        .collect_view();

    let cards = graph
        .nodes
        .iter()
        .filter_map(|n| {
            positions_by_id.get(&n.id).map(|pos| {
                view! { <StageCard node=n.clone() position=pos.clone() selected=selected /> }
            })
        })
        .collect_view();

    view! {
        <div class="pipeline-dag-scroll">
            <div class="pipeline-dag-column-headers" style=format!("width:{width}px;")>
                {columns_header}
            </div>
            <div class="pipeline-dag-canvas" style=container_style>
                <svg class="pipeline-dag-svg" style=svg_style width=width height=height viewBox=format!("0 0 {width} {height}")>
                    {edges}
                </svg>
                {cards}
            </div>
        </div>
    }
}
