//! Types shared by the host call pipeline and its localizer seam.

use std::collections::BTreeMap;

use crate::protocol::detection_messages::{ClipPayload, ReadinessAnswer};
use crate::protocol::wire_codes::SpeciesScore;
use crate::protocol::NodeId;

use super::clock_mapping::NodeClockMapping;
use super::config::MeshConfig;
use super::host_membership::SessionMembership;
use super::island_sync_engine::{IslandSyncEngine, NodePositionForSolve};
use super::{DetectionKey, MeshMilliseconds};

/// Read-only host state a call step needs.
pub struct CallContext<'host> {
    pub engine: &'host IslandSyncEngine,
    pub membership: &'host SessionMembership,
    pub present: &'host [NodeId],
    pub config: &'host MeshConfig,
    /// False on a constrained link or in detection-only mode.
    pub clip_fetch_enabled: bool,
}

/// One usable clip, already placed on the island clock and map.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipLocalizationInput {
    pub node_id: NodeId,
    pub payload: ClipPayload,
    pub clock: NodeClockMapping,
    pub position: NodePositionForSolve,
    pub walked_at_call: bool,
    pub island_id: u32,
    pub island_epoch: u64,
    pub clock_epoch_at_request: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallLocalizationRequest {
    pub call_id: u64,
    pub island_id: u32,
    pub classifying_node: NodeId,
    pub centre_reference_s: f64,
    pub clips: Vec<ClipLocalizationInput>,
    /// Map chirality confidence of the island's anchor (None: unanchored).
    pub chirality_confidence: Option<f64>,
    pub chirality_locked: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallFix {
    pub position_m: [f64; 3],
    pub covariance_m2: [[f64; 3]; 3],
    pub horizontal_major_sigma_m: f64,
    pub used_nodes: Vec<NodeId>,
    pub rejected_nodes: Vec<NodeId>,
}

/// Seam between the call pipeline and a solver: clip SRP + LM on phones,
/// arrival-time abstraction in the Layer 2 simulator.
pub trait ClipSetLocalizer {
    fn localize(&self, request: &CallLocalizationRequest) -> Result<CallFix, String>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RequestedClip {
    pub request_id: u32,
    pub clock_epoch_at_request: u32,
}

#[derive(Debug, Clone)]
pub enum PendingStage {
    AwaitingReadiness {
        deadline_ms: MeshMilliseconds,
        queries: BTreeMap<NodeId, u32>,
        answers: BTreeMap<NodeId, ReadinessAnswer>,
    },
    FetchingClips {
        deadline_ms: MeshMilliseconds,
        requests: BTreeMap<NodeId, RequestedClip>,
        received: BTreeMap<NodeId, ClipPayload>,
        readiness_excluded_by_cap: usize,
    },
}

#[derive(Debug, Clone)]
pub struct PendingCall {
    pub call_id: u64,
    pub island_id: u32,
    pub island_epoch: u64,
    pub classifying_node: NodeId,
    pub centre_reference_s: f64,
    pub half_window_s: f64,
    pub received_ms: MeshMilliseconds,
    pub species_belief: Vec<SpeciesScore>,
    pub detections: Vec<DetectionKey>,
    pub stage: PendingStage,
}

/// Outcome of one call, kept for the entity update and for assertions.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LocalizedCallRecord {
    pub call_id: u64,
    pub island_id: u32,
    pub received_ms: MeshMilliseconds,
    pub completed_ms: MeshMilliseconds,
    pub centre_reference_s: f64,
    pub classifying_node: NodeId,
    pub species_belief: Vec<SpeciesScore>,
    pub detections: Vec<DetectionKey>,
    pub requested_nodes: Vec<NodeId>,
    pub clip_nodes: Vec<NodeId>,
    pub walked_nodes: Vec<NodeId>,
    pub readiness_excluded_by_cap: usize,
    /// (node, island id, island epoch, clock epoch at request, at use).
    pub clip_domains: Vec<(NodeId, u32, u64, u32, u32)>,
    pub chirality_confidence: Option<f64>,
    pub chirality_locked: bool,
    pub fix: Option<CallFix>,
    pub failure: Option<String>,
}

impl LocalizedCallRecord {
    pub fn from_call(call: &PendingCall, now_ms: MeshMilliseconds, readiness_excluded_by_cap: usize) -> Self {
        Self {
            call_id: call.call_id,
            island_id: call.island_id,
            received_ms: call.received_ms,
            completed_ms: now_ms,
            centre_reference_s: call.centre_reference_s,
            classifying_node: call.classifying_node,
            species_belief: call.species_belief.clone(),
            detections: call.detections.clone(),
            readiness_excluded_by_cap,
            ..Self::default()
        }
    }

    pub fn abandoned(call: &PendingCall, now_ms: MeshMilliseconds, reason: &str) -> Self {
        Self {
            failure: Some(reason.to_string()),
            ..Self::from_call(call, now_ms, 0)
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CallPipelineCounters {
    pub readiness_queries_sent: u64,
    pub readiness_answers_received: u64,
    pub readiness_not_ready: u64,
    pub readiness_excluded_by_cap: u64,
    pub clip_requests_sent: u64,
    pub clips_received: u64,
    pub stale_clips_discarded: u64,
    pub late_clips_discarded: u64,
    pub clips_denied_by_budget: u64,
    pub calls_skipped_for_clip_budget: u64,
    pub calls_dropped_backpressure: u64,
    pub calls_without_clip_fetch: u64,
    pub calls_localized: u64,
    pub calls_abandoned: u64,
    pub detections_merged_into_calls: u64,
}
