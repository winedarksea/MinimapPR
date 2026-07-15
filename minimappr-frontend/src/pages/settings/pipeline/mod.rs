//! Read-only Pipeline Flow DAG (`/settings/pipeline`). The graph is a bounded
//! zoom/pan surface so a complete site topology remains inspectable without
//! horizontal scrolling; cards still select the sticky details drawer.

mod card;
mod layout;
mod types;

use card::{DetailPanel, StageCard};
use futures::StreamExt;
use gloo_net::http::Request;
use gloo_timers::future::IntervalStream;
use layout::compute_layout;
use leptos::task::spawn_local;
use leptos::{ev, html, prelude::*};
use types::PipelineGraph;

const POLL_MS: u32 = 2_500;
const GRAPH_HEADER_HEIGHT: f64 = 28.0;
const MIN_ZOOM: f64 = 0.25;
const MAX_ZOOM: f64 = 2.5;

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
                        <div class="pipeline-dag-workspace">
                            {(!fusion_available).then(|| view! {
                                <div class="pipeline-banner-warn pipeline-health-banner">
                                    "Fusion pipeline unavailable — showing configured structure only."
                                </div>
                            })}
                            <PipelineDagCanvas graph=g selected=selected />
                            {move || selected_node().map(|n| {
                                let close_drawer = move |_| selected.set(None);
                                view! {
                                    <section class="pipeline-dag-detail-drawer" aria-label="Selected pipeline stage">
                                        <button class="pipeline-dag-detail-close" type="button" on:click=close_drawer>
                                            "Close details"
                                        </button>
                                        <DetailPanel node=n />
                                    </section>
                                }
                            })}
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
    let board_height = height + GRAPH_HEADER_HEIGHT;
    let viewport_ref = NodeRef::<html::Div>::new();
    let zoom = RwSignal::new(1.0);
    let pan = RwSignal::new((0.0, 0.0));
    let active_pointers = RwSignal::new(Vec::<(i32, f64, f64)>::new());
    let drag_pointer = RwSignal::new(None::<(i32, f64, f64)>);
    let pinch_start = RwSignal::new(None::<(f64, f64)>);

    let fit_to_view = move || {
        if let Some(viewport) = viewport_ref.get() {
            let viewport_size = (
                viewport.client_width() as f64,
                viewport.client_height() as f64,
            );
            let scale = fit_scale(viewport_size, (width, board_height));
            zoom.set(scale);
            pan.set(center_pan(viewport_size, (width, board_height), scale));
        }
    };
    Effect::new(move |_| fit_to_view());
    let resize_listener = window_event_listener(ev::resize, move |_| fit_to_view());
    on_cleanup(move || resize_listener.remove());

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

    let edges = layout.edge_paths.iter().map(|e| {
        let cls = format!("pipeline-dag-edge kind-{}{}", e.kind, if e.active { "" } else { " inactive" });
        let label_cls = if e.active { "pipeline-dag-edge-label" } else { "pipeline-dag-edge-label inactive" };
        view! {
            <path class=cls d=e.d.clone() fill="none" />
            {(!e.label.is_empty()).then(|| view! {
                <text class=label_cls x=e.label_x y=e.label_y text-anchor="middle">{e.label.clone()}</text>
            })}
        }
    }).collect_view();

    let cards = graph
        .nodes
        .iter()
        .filter_map(|n| {
            positions_by_id.get(&n.id).map(|pos| {
                view! {
                    <StageCard node=n.clone() position=pos.clone() selected=selected />
                }
            })
        })
        .collect_view();

    let on_pointer_down = move |event: ev::PointerEvent| {
        let pointer = (
            event.pointer_id(),
            event.client_x() as f64,
            event.client_y() as f64,
        );
        active_pointers.update(|pointers| {
            pointers.retain(|(id, _, _)| *id != pointer.0);
            pointers.push(pointer);
            if pointers.len() == 2 {
                pinch_start.set(Some((
                    pointer_distance(pointers[0], pointers[1]),
                    zoom.get(),
                )));
                drag_pointer.set(None);
            } else if pointers.len() == 1 {
                drag_pointer.set(Some(pointer));
            }
        });
    };
    let on_pointer_move = move |event: ev::PointerEvent| {
        let pointer = (
            event.pointer_id(),
            event.client_x() as f64,
            event.client_y() as f64,
        );
        active_pointers.update(|pointers| {
            if let Some(current) = pointers.iter_mut().find(|(id, _, _)| *id == pointer.0) {
                *current = pointer;
            } else {
                return;
            }
            if pointers.len() == 2 {
                if let Some((start_distance, start_scale)) = pinch_start.get() {
                    if start_distance > 0.0 {
                        zoom.set(
                            (start_scale * pointer_distance(pointers[0], pointers[1])
                                / start_distance)
                                .clamp(MIN_ZOOM, MAX_ZOOM),
                        );
                    }
                }
            } else if let Some(previous) = drag_pointer.get() {
                if previous.0 == pointer.0 {
                    pan.update(|(x, y)| {
                        *x += pointer.1 - previous.1;
                        *y += pointer.2 - previous.2;
                    });
                    drag_pointer.set(Some(pointer));
                }
            }
        });
        if let Some(viewport) = viewport_ref.get() {
            let viewport_size = (
                viewport.client_width() as f64,
                viewport.client_height() as f64,
            );
            pan.update(|value| {
                *value = clamp_pan(*value, viewport_size, (width, board_height), zoom.get())
            });
        }
    };
    let on_pointer_end = move |event: ev::PointerEvent| {
        active_pointers.update(|pointers| pointers.retain(|(id, _, _)| *id != event.pointer_id()));
        drag_pointer.set(None);
        pinch_start.set(None);
    };
    let on_wheel = move |event: ev::WheelEvent| {
        if event.ctrl_key() || event.meta_key() {
            event.prevent_default();
            let factor = if event.delta_y() < 0.0 {
                1.12
            } else {
                1.0 / 1.12
            };
            zoom.update(|value| *value = (*value * factor).clamp(MIN_ZOOM, MAX_ZOOM));
            if let Some(viewport) = viewport_ref.get() {
                let viewport_size = (
                    viewport.client_width() as f64,
                    viewport.client_height() as f64,
                );
                pan.update(|value| {
                    *value = clamp_pan(*value, viewport_size, (width, board_height), zoom.get())
                });
            }
        }
    };
    let zoom_in = move |_| {
        zoom.update(|value| *value = (*value * 1.2).clamp(MIN_ZOOM, MAX_ZOOM));
        if let Some(viewport) = viewport_ref.get() {
            let viewport_size = (
                viewport.client_width() as f64,
                viewport.client_height() as f64,
            );
            pan.update(|value| {
                *value = clamp_pan(*value, viewport_size, (width, board_height), zoom.get())
            });
        }
    };
    let zoom_out = move |_| {
        zoom.update(|value| *value = (*value / 1.2).clamp(MIN_ZOOM, MAX_ZOOM));
        if let Some(viewport) = viewport_ref.get() {
            let viewport_size = (
                viewport.client_width() as f64,
                viewport.client_height() as f64,
            );
            pan.update(|value| {
                *value = clamp_pan(*value, viewport_size, (width, board_height), zoom.get())
            });
        }
    };

    view! {
        <div class="pipeline-dag-controls" aria-label="Pipeline graph controls">
            <button type="button" on:click=zoom_out aria-label="Zoom out">"−"</button>
            <button type="button" on:click=zoom_in aria-label="Zoom in">"+"</button>
            <button type="button" on:click=move |_| fit_to_view()>"Fit graph"</button>
        </div>
        <div
            class="pipeline-dag-viewport"
            node_ref=viewport_ref
            on:pointerdown=on_pointer_down
            on:pointermove=on_pointer_move
            on:pointerup=on_pointer_end
            on:pointercancel=on_pointer_end
            on:wheel=on_wheel
        >
            <div class="pipeline-dag-surface" style=move || format!(
                "width:{width}px; height:{board_height}px; transform:translate({}px, {}px) scale({});",
                pan.get().0, pan.get().1, zoom.get()
            )>
                <div class="pipeline-dag-column-headers" style=format!("width:{width}px;")>{columns_header}</div>
                <div class="pipeline-dag-canvas" style=format!("width:{width}px; height:{height}px;")>
                    <svg class="pipeline-dag-svg" width=width height=height viewBox=format!("0 0 {width} {height}")>{edges}</svg>
                    {cards}
                </div>
            </div>
        </div>
    }
}

fn fit_scale(viewport: (f64, f64), board: (f64, f64)) -> f64 {
    (viewport.0 / board.0)
        .min(viewport.1 / board.1)
        .clamp(MIN_ZOOM, 1.0)
}

fn center_pan(viewport: (f64, f64), board: (f64, f64), scale: f64) -> (f64, f64) {
    (
        (viewport.0 - board.0 * scale) / 2.0,
        (viewport.1 - board.1 * scale) / 2.0,
    )
}

fn clamp_pan(pan: (f64, f64), viewport: (f64, f64), board: (f64, f64), scale: f64) -> (f64, f64) {
    let clamp_axis = |value: f64, viewport_size: f64, board_size: f64| {
        let scaled_size = board_size * scale;
        if scaled_size <= viewport_size {
            (viewport_size - scaled_size) / 2.0
        } else {
            value.clamp(viewport_size - scaled_size, 0.0)
        }
    };
    (
        clamp_axis(pan.0, viewport.0, board.0),
        clamp_axis(pan.1, viewport.1, board.1),
    )
}

fn pointer_distance(a: (i32, f64, f64), b: (i32, f64, f64)) -> f64 {
    ((a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_scale_is_bounded_and_fits_both_dimensions() {
        assert_eq!(fit_scale((1000.0, 800.0), (500.0, 400.0)), 1.0);
        assert_eq!(fit_scale((250.0, 300.0), (1000.0, 400.0)), 0.25);
        assert_eq!(fit_scale((10.0, 10.0), (1000.0, 1000.0)), MIN_ZOOM);
    }

    #[test]
    fn center_pan_centers_scaled_board() {
        assert_eq!(
            center_pan((1000.0, 800.0), (500.0, 400.0), 1.0),
            (250.0, 200.0)
        );
    }

    #[test]
    fn clamp_pan_keeps_a_zoomed_board_covering_the_viewport() {
        assert_eq!(
            clamp_pan((100.0, -900.0), (500.0, 400.0), (500.0, 400.0), 2.0),
            (0.0, -400.0),
        );
        assert_eq!(
            clamp_pan((0.0, 0.0), (1000.0, 800.0), (500.0, 400.0), 1.0),
            (250.0, 200.0),
        );
    }
}
