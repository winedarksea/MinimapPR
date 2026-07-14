//! Serde mirrors of `minimappr.models` pipeline-graph types
//! (`GET /api/v1/pipeline/graph`). Kept intentionally permissive
//! (`#[serde(default)]` on everything optional) so backend additions never
//! break parsing on the frontend.

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
pub struct PipelineParam {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub value: String,
    #[serde(default)]
    pub config_key: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
pub struct PipelineStageStatus {
    #[serde(default = "default_unknown")]
    pub health: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub metrics: Vec<PipelineParam>,
}

fn default_unknown() -> String {
    "unknown".to_string()
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct PipelineGraphNode {
    pub id: String,
    pub stage: String,
    pub column: String,
    pub lane: String,
    pub title: String,
    #[serde(default)]
    pub subtitle: String,
    #[serde(default = "default_audio")]
    pub modality: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub node_type: Option<String>,
    #[serde(default)]
    pub params: Vec<PipelineParam>,
    #[serde(default)]
    pub status: PipelineStageStatus,
    #[serde(default)]
    pub link: Option<String>,
}

fn default_audio() -> String {
    "audio".to_string()
}
fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct PipelineGraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    #[serde(default = "default_audio")]
    pub kind: String,
    #[serde(default)]
    pub label: String,
    #[serde(default = "default_true")]
    pub active: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct PipelineGraphColumn {
    pub id: String,
    pub title: String,
    pub order: i64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct PipelineGraphLane {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub node_type: Option<String>,
    #[serde(default)]
    pub health: Option<String>,
    #[serde(default)]
    pub link: Option<String>,
    #[serde(default)]
    pub order: i64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct PipelineGraph {
    #[serde(default)]
    pub generated_ns: i64,
    #[serde(default)]
    pub active_pipeline: String,
    #[serde(default)]
    pub structure_hash: String,
    #[serde(default = "default_true")]
    pub fusion_available: bool,
    #[serde(default)]
    pub columns: Vec<PipelineGraphColumn>,
    #[serde(default)]
    pub lanes: Vec<PipelineGraphLane>,
    #[serde(default)]
    pub nodes: Vec<PipelineGraphNode>,
    #[serde(default)]
    pub edges: Vec<PipelineGraphEdge>,
}
