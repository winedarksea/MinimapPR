pub mod acoustic;
pub mod detections;
pub mod heatmap;
pub mod nodes;
pub mod omni;
pub mod overlays;
pub mod tracks;
pub mod uncertainty;
pub mod zones;

use crate::state::{AppState, MapLayerVisibility};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerGroup {
    Entities,
    Context,
    Analytics,
}

impl LayerGroup {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Entities => "entities",
            Self::Context => "context",
            Self::Analytics => "analytics",
        }
    }
}

#[derive(Clone, Copy)]
pub struct LayerDef {
    pub id: &'static str,
    pub title: &'static str,
    pub group: LayerGroup,
    pub default_visible: bool,
    pub get_visible: fn(&MapLayerVisibility) -> bool,
    pub set_visible: fn(&mut MapLayerVisibility, bool),
}

pub const LAYER_DEFS: &[LayerDef] = &[
    LayerDef {
        id: "nodes",
        title: "Nodes",
        group: LayerGroup::Entities,
        default_visible: true,
        get_visible: |layers| layers.nodes,
        set_visible: |layers, visible| layers.nodes = visible,
    },
    LayerDef {
        id: "tracks",
        title: "Tracks",
        group: LayerGroup::Entities,
        default_visible: true,
        get_visible: |layers| layers.tracks,
        set_visible: |layers, visible| layers.tracks = visible,
    },
    LayerDef {
        id: "detections",
        title: "Detections",
        group: LayerGroup::Entities,
        default_visible: true,
        get_visible: |layers| layers.detections,
        set_visible: |layers, visible| layers.detections = visible,
    },
    LayerDef {
        id: "omni",
        title: "Omni",
        group: LayerGroup::Entities,
        default_visible: true,
        get_visible: |layers| layers.omni,
        set_visible: |layers, visible| layers.omni = visible,
    },
    LayerDef {
        id: "zones",
        title: "Zones",
        group: LayerGroup::Context,
        default_visible: true,
        get_visible: |layers| layers.zones,
        set_visible: |layers, visible| layers.zones = visible,
    },
    LayerDef {
        id: "overlays",
        title: "Overlays",
        group: LayerGroup::Context,
        default_visible: true,
        get_visible: |layers| layers.overlays,
        set_visible: |layers, visible| layers.overlays = visible,
    },
    LayerDef {
        id: "acoustic",
        title: "Acoustic",
        group: LayerGroup::Analytics,
        default_visible: true,
        get_visible: |layers| layers.acoustic,
        set_visible: |layers, visible| layers.acoustic = visible,
    },
];

pub fn all_layer_defs() -> &'static [LayerDef] {
    LAYER_DEFS
}

pub fn mount_core_marker_layers(state: &AppState) {
    nodes::mount(state);
    detections::mount(state);
    omni::mount(state);
    tracks::mount(state);
    zones::mount(state);
    heatmap::mount(state);
    overlays::mount(state);
    acoustic::mount(state);
    uncertainty::mount(state);
}
