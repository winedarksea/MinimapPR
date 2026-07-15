//! Pure, deterministic geometry for the pipeline DAG. No DOM measurement —
//! fixed column widths and card height so edges can be drawn as an SVG
//! underlay without `getBoundingClientRect`/`ResizeObserver` timing loops in
//! Leptos CSR (see plan Phase 5). Host-testable in isolation from rendering.

use super::types::{PipelineGraph, PipelineGraphEdge};
use std::collections::HashMap;

pub const COLUMN_WIDTH: f64 = 190.0;
pub const COLUMN_GUTTER: f64 = 48.0;
pub const CARD_HEIGHT: f64 = 64.0;
pub const CARD_V_GAP: f64 = 14.0;
pub const LANE_V_GAP: f64 = 28.0;
pub const LANE_HEADER_H: f64 = 22.0;

#[derive(Clone, Debug, PartialEq)]
pub struct CardPosition {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EdgePath {
    pub id: String,
    pub d: String,
    pub kind: String,
    pub label: String,
    pub label_x: f64,
    pub label_y: f64,
    pub active: bool,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct GraphLayout {
    pub positions: Vec<CardPosition>,
    pub width: f64,
    pub height: f64,
    pub edge_paths: Vec<EdgePath>,
}

fn column_x(order: usize) -> f64 {
    order as f64 * (COLUMN_WIDTH + COLUMN_GUTTER)
}

/// Lay out cards in a stage-columns × node-lanes grid, then compute cubic
/// Bézier edge paths (right-center of source → left-center of target).
pub fn compute_layout(graph: &PipelineGraph) -> GraphLayout {
    let mut column_order: HashMap<&str, usize> = HashMap::new();
    let mut columns_sorted: Vec<_> = graph.columns.iter().collect();
    columns_sorted.sort_by_key(|c| c.order);
    for (i, c) in columns_sorted.iter().enumerate() {
        column_order.insert(c.id.as_str(), i);
    }

    let mut lanes_sorted: Vec<_> = graph.lanes.iter().collect();
    lanes_sorted.sort_by_key(|l| l.order);
    let mut lane_order: HashMap<&str, usize> = HashMap::new();
    for (i, l) in lanes_sorted.iter().enumerate() {
        lane_order.insert(l.id.as_str(), i);
    }

    // Bucket nodes per (lane, column) to stack same-cell nodes vertically.
    let mut cell_counts: HashMap<(usize, usize), usize> = HashMap::new();
    let mut lane_row_heights: HashMap<usize, f64> = HashMap::new();

    struct Placed<'a> {
        id: &'a str,
        col: usize,
        lane: usize,
        slot: usize,
    }
    let mut placed: Vec<Placed> = Vec::new();

    for node in &graph.nodes {
        let col = *column_order.get(node.column.as_str()).unwrap_or(&0);
        let lane = *lane_order.get(node.lane.as_str()).unwrap_or(&0);
        let slot = *cell_counts.entry((lane, col)).or_insert(0);
        cell_counts.insert((lane, col), slot + 1);
        placed.push(Placed {
            id: &node.id,
            col,
            lane,
            slot,
        });
    }

    // Row height per lane = max stack depth in any column for that lane.
    for ((lane, _col), count) in &cell_counts {
        let h = lane_row_heights.entry(*lane).or_insert(0.0);
        let stack_h = (*count as f64) * CARD_HEIGHT + (*count as f64 - 1.0).max(0.0) * CARD_V_GAP;
        if stack_h > *h {
            *h = stack_h;
        }
    }

    // Compute lane y-offsets (cumulative).
    let mut lane_y_offset: HashMap<usize, f64> = HashMap::new();
    let mut cursor = 0.0;
    for i in 0..lanes_sorted.len() {
        lane_y_offset.insert(i, cursor);
        let row_h = *lane_row_heights.get(&i).unwrap_or(&CARD_HEIGHT);
        cursor += LANE_HEADER_H + row_h + LANE_V_GAP;
    }
    let total_height = cursor;

    let mut positions: Vec<CardPosition> = Vec::with_capacity(placed.len());
    let mut pos_by_id: HashMap<String, (f64, f64)> = HashMap::new();
    for p in &placed {
        let x = column_x(p.col);
        let lane_top = *lane_y_offset.get(&p.lane).unwrap_or(&0.0) + LANE_HEADER_H;
        let y = lane_top + (p.slot as f64) * (CARD_HEIGHT + CARD_V_GAP);
        positions.push(CardPosition {
            id: p.id.to_string(),
            x,
            y,
            width: COLUMN_WIDTH,
            height: CARD_HEIGHT,
        });
        pos_by_id.insert(p.id.to_string(), (x, y));
    }

    let total_width = columns_sorted.len() as f64 * (COLUMN_WIDTH + COLUMN_GUTTER);

    let mut parallel_edge_counts: HashMap<(&str, &str), usize> = HashMap::new();
    for edge in &graph.edges {
        *parallel_edge_counts
            .entry((&edge.source, &edge.target))
            .or_insert(0) += 1;
    }
    let mut parallel_edge_indices: HashMap<(&str, &str), usize> = HashMap::new();

    let edge_paths = graph
        .edges
        .iter()
        .filter_map(|e: &PipelineGraphEdge| {
            let (sx, sy) = pos_by_id.get(&e.source)?;
            let (tx, ty) = pos_by_id.get(&e.target)?;
            let pair = (e.source.as_str(), e.target.as_str());
            let index = parallel_edge_indices.entry(pair).or_insert(0);
            let count = *parallel_edge_counts.get(&pair).unwrap_or(&1);
            let offset = (*index as f64 - (count.saturating_sub(1) as f64 / 2.0)) * 12.0;
            *index += 1;
            let x1 = sx + COLUMN_WIDTH;
            let y1 = sy + CARD_HEIGHT / 2.0 + offset;
            let x2 = *tx;
            let y2 = ty + CARD_HEIGHT / 2.0 + offset;
            let d = format!(
                "M {x1:.1},{y1:.1} C {c1:.1},{y1:.1} {c2:.1},{y2:.1} {x2:.1},{y2:.1}",
                c1 = x1 + 40.0,
                c2 = x2 - 40.0,
            );
            Some(EdgePath {
                id: e.id.clone(),
                d,
                kind: e.kind.clone(),
                label: e.label.clone(),
                label_x: (x1 + x2) / 2.0,
                label_y: (y1 + y2) / 2.0 - 5.0,
                active: e.active,
            })
        })
        .collect();

    GraphLayout {
        positions,
        width: total_width,
        height: total_height,
        edge_paths,
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{
        PipelineGraph, PipelineGraphColumn, PipelineGraphEdge, PipelineGraphLane,
        PipelineGraphNode, PipelineStageStatus,
    };
    use super::*;

    fn node(id: &str, column: &str, lane: &str) -> PipelineGraphNode {
        PipelineGraphNode {
            id: id.to_string(),
            stage: "source".to_string(),
            column: column.to_string(),
            lane: lane.to_string(),
            title: id.to_string(),
            subtitle: String::new(),
            modality: "audio".to_string(),
            enabled: true,
            node_type: None,
            params: vec![],
            status: PipelineStageStatus::default(),
            link: None,
        }
    }

    fn sample_graph() -> PipelineGraph {
        PipelineGraph {
            generated_ns: 0,
            active_pipeline: "python".to_string(),
            structure_hash: "x".to_string(),
            fusion_available: true,
            columns: vec![
                PipelineGraphColumn {
                    id: "sources".into(),
                    title: "Sources".into(),
                    order: 0,
                },
                PipelineGraphColumn {
                    id: "gates".into(),
                    title: "Gates".into(),
                    order: 1,
                },
            ],
            lanes: vec![PipelineGraphLane {
                id: "n1".into(),
                title: "n1".into(),
                node_type: None,
                health: None,
                link: None,
                order: 0,
            }],
            nodes: vec![
                node("src:n1", "sources", "n1"),
                node("gate:n1", "gates", "n1"),
            ],
            edges: vec![PipelineGraphEdge {
                id: "src:n1->gate:n1".into(),
                source: "src:n1".into(),
                target: "gate:n1".into(),
                kind: "audio".into(),
                label: String::new(),
                active: true,
            }],
        }
    }

    #[test]
    fn column_order_is_monotonic_in_x() {
        let layout = compute_layout(&sample_graph());
        let src = layout.positions.iter().find(|p| p.id == "src:n1").unwrap();
        let gate = layout.positions.iter().find(|p| p.id == "gate:n1").unwrap();
        assert!(gate.x > src.x);
    }

    #[test]
    fn edge_paths_reference_known_cards() {
        let layout = compute_layout(&sample_graph());
        assert_eq!(layout.edge_paths.len(), 1);
        let path = &layout.edge_paths[0];
        assert!(path.d.starts_with("M "));
        assert!(path.d.contains('C'));
        assert!(path.label_x > 0.0);
    }

    #[test]
    fn dangling_edge_is_dropped_not_panicking() {
        let mut g = sample_graph();
        g.edges.push(PipelineGraphEdge {
            id: "ghost".into(),
            source: "does-not-exist".into(),
            target: "gate:n1".into(),
            kind: "audio".into(),
            label: String::new(),
            active: true,
        });
        let layout = compute_layout(&g);
        assert_eq!(layout.edge_paths.len(), 1);
    }

    #[test]
    fn stacked_nodes_in_same_cell_do_not_overlap() {
        let mut g = sample_graph();
        g.nodes.push(node("src:n1:extra", "sources", "n1"));
        let layout = compute_layout(&g);
        let a = layout.positions.iter().find(|p| p.id == "src:n1").unwrap();
        let b = layout
            .positions
            .iter()
            .find(|p| p.id == "src:n1:extra")
            .unwrap();
        assert!((a.y - b.y).abs() >= CARD_HEIGHT);
    }

    #[test]
    fn parallel_edges_have_distinct_paths_and_label_positions() {
        let mut g = sample_graph();
        g.edges.push(PipelineGraphEdge {
            id: "src:n1->gate:n1:second".into(),
            source: "src:n1".into(),
            target: "gate:n1".into(),
            kind: "audio".into(),
            label: "alternate route".into(),
            active: true,
        });
        let layout = compute_layout(&g);
        assert_eq!(layout.edge_paths.len(), 2);
        assert_ne!(layout.edge_paths[0].d, layout.edge_paths[1].d);
        assert_ne!(layout.edge_paths[0].label_y, layout.edge_paths[1].label_y);
    }
}
