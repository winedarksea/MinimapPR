//! A call solved on a one-round clock model (closeout F2-H1): the clip's
//! clock σ is the filter's prediction at the call, and the call waits for
//! the round after it rather than extrapolating a rate-less model.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::layout::filtered_map_anchor::FilteredMapAnchor;
use crate::layout::gps_anchor_fit::ChiralityCue;
use crate::mesh::config::MeshConfig;
use crate::mesh::host_membership::SessionMembership;
use crate::mesh::island_sync_engine::{IslandLayout, IslandState, IslandSyncEngine};
use crate::sync::clock_filter::PhoneClockFilter;

const ISLAND_ID: u32 = 1;
const ROUND_SIGMA_S: f64 = 15e-6;
const FIRST_ROUND_S: f64 = 100.0;
const CALL_CENTRE_S: f64 = 180.0;
const RECEIVED_MS: MeshMilliseconds = 180_000;
/// A rate-less model 80 s out is off by up to milliseconds, so its σ must be.
const RATE_LESS_SIGMA_FLOOR_S: f64 = 1e-3;
/// Pinned by a round 1 s after the call, with a rate from an 81 s baseline.
const PINNED_SIGMA_CEILING_S: f64 = 50e-6;

fn node(id: u32) -> NodeId {
    NodeId(id)
}

/// Island {0, 1, 2}, reference 0, nodes 1 and 2 seen in one round only.
fn engine_after_one_round(config: &MeshConfig) -> IslandSyncEngine {
    let members: BTreeSet<NodeId> = (0..3).map(node).collect();
    let mut filters = BTreeMap::new();
    for id in 1..3 {
        let mut filter = PhoneClockFilter::new(config.island_sync.clock_filter.clone());
        filter.update(FIRST_ROUND_S, 0.01 * id as f64, ROUND_SIGMA_S);
        filters.insert(node(id), filter);
    }
    let positions = [[0.0, 0.0, 1.4], [10.0, 0.0, 1.4], [0.0, 10.0, 1.4]];
    let layout = IslandLayout {
        captured_ms: 100_000,
        local_positions_m: members.iter().copied().zip(positions).collect(),
        sigmas_m: members.iter().map(|n| (*n, 0.05)).collect(),
        anchor: None,
    };
    let mut engine = IslandSyncEngine::default();
    engine.islands.insert(ISLAND_ID, IslandState {
        island_id: ISLAND_ID,
        tracker_epoch: 1,
        members,
        reference_node: Some(node(0)),
        filters,
        last_reference_time_s: FIRST_ROUND_S,
        layout: Some(layout),
        previous_layout: None,
        layout_epoch: 1,
        chirality_cue: ChiralityCue::Unknown,
        anchor: None,
        anchor_tracker: FilteredMapAnchor::new(config.island_sync.anchor_tracking.clone()),
        group_is_still: true,
    });
    engine
}

/// Records each request's clip clock σ and returns a fix at the origin.
#[derive(Default)]
struct RecordingLocalizer {
    clock_sigmas_s: RefCell<Vec<Vec<f64>>>,
}

impl ClipSetLocalizer for RecordingLocalizer {
    fn localize(&self, request: &CallLocalizationRequest) -> Result<CallFix, String> {
        self.clock_sigmas_s.borrow_mut().push(request.clips.iter().map(|clip| clip.clock.offset_sigma_s).collect());
        Ok(CallFix {
            position_m: [0.0; 3],
            covariance_m2: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            horizontal_major_sigma_m: 1.0,
            used_nodes: request.clips.iter().map(|clip| clip.node_id).collect(),
            rejected_nodes: Vec::new(),
        })
    }
}

/// A call with every clip in, ready to localize.
fn pipeline_with_clips_in() -> CallPipeline {
    let mut requests = BTreeMap::new();
    let mut received = BTreeMap::new();
    for id in 0..3 {
        requests.insert(node(id), RequestedClip { request_id: id + 1, clock_epoch_at_request: 0 });
        received.insert(node(id), ClipPayload {
            request_id: id + 1,
            node_id: node(id),
            clip_container: Vec::new(),
            onsets: Vec::new(),
            sensor_frames: Vec::new(),
        });
    }
    let mut pipeline = CallPipeline::default();
    pipeline.pending.push_back(PendingCall {
        call_id: 7,
        island_id: ISLAND_ID,
        island_epoch: 1,
        classifying_node: node(0),
        centre_reference_s: CALL_CENTRE_S,
        half_window_s: 0.8,
        received_ms: RECEIVED_MS,
        species_belief: Vec::new(),
        detections: vec![(node(0), 1)],
        held_for_clock_model: false,
        stage: PendingStage::FetchingClips {
            deadline_ms: RECEIVED_MS + 4_000,
            requests,
            received,
            readiness_excluded_by_cap: 0,
        },
    });
    pipeline
}

fn context<'host>(engine: &'host IslandSyncEngine, membership: &'host SessionMembership, config: &'host MeshConfig) -> CallContext<'host> {
    CallContext { engine, membership, present: &[], config, clip_fetch_enabled: true }
}

#[test]
fn a_rate_less_clock_maps_with_its_prediction_sigma_at_the_call() {
    let config = MeshConfig::default();
    let engine = engine_after_one_round(&config);
    let at_round = engine.clock_mapping(node(1)).expect("model").offset_sigma_s;
    let at_call = engine.clock_mapping_at(node(1), CALL_CENTRE_S).expect("model").offset_sigma_s;
    assert!(at_round < 2.0 * ROUND_SIGMA_S, "round-time σ {at_round}");
    assert!(at_call > RATE_LESS_SIGMA_FLOOR_S, "σ 80 s past a rate-less round {at_call}");
    assert_eq!(engine.clock_mapping_at(node(0), CALL_CENTRE_S).expect("reference").offset_sigma_s, 0.0);
}

#[test]
fn a_call_on_a_rate_less_clock_waits_for_the_round_after_it() {
    let config = MeshConfig::default();
    let membership = SessionMembership::default();
    let mut engine = engine_after_one_round(&config);
    let localizer = RecordingLocalizer::default();
    let mut pipeline = pipeline_with_clips_in();

    pipeline.tick(&context(&engine, &membership, &config), &localizer, RECEIVED_MS + 1_000);
    assert!(localizer.clock_sigmas_s.borrow().is_empty(), "solved before the post-detection round");
    assert_eq!(pipeline.counters.calls_held_for_clock_model, 1);

    let post_detection_round_s = CALL_CENTRE_S + 1.0;
    let island = engine.islands.get_mut(&ISLAND_ID).expect("island");
    for (id, filter) in island.filters.iter_mut() {
        filter.update(post_detection_round_s, 0.01 * id.0 as f64, ROUND_SIGMA_S);
    }
    island.last_reference_time_s = post_detection_round_s;
    pipeline.tick(&context(&engine, &membership, &config), &localizer, RECEIVED_MS + 3_000);
    let sigmas = localizer.clock_sigmas_s.borrow();
    assert_eq!(sigmas.len(), 1);
    assert!(sigmas[0].iter().all(|sigma| *sigma < PINNED_SIGMA_CEILING_S), "{sigmas:?}");
    assert_eq!(pipeline.counters.clock_model_waits_expired, 0);
}

#[test]
fn a_round_that_never_comes_ends_the_wait_with_an_honest_sigma() {
    let config = MeshConfig::default();
    let membership = SessionMembership::default();
    let engine = engine_after_one_round(&config);
    let localizer = RecordingLocalizer::default();
    let mut pipeline = pipeline_with_clips_in();
    let expiry_ms = RECEIVED_MS + config.clip_fetch.clock_hold_maximum_wait_ms;

    pipeline.tick(&context(&engine, &membership, &config), &localizer, expiry_ms - 1);
    assert!(localizer.clock_sigmas_s.borrow().is_empty());
    pipeline.tick(&context(&engine, &membership, &config), &localizer, expiry_ms);
    let sigmas = localizer.clock_sigmas_s.borrow();
    assert_eq!(sigmas.len(), 1);
    assert!(sigmas[0].iter().filter(|sigma| **sigma > 0.0).all(|sigma| *sigma > RATE_LESS_SIGMA_FLOOR_S), "{sigmas:?}");
    assert_eq!(pipeline.counters.clock_model_waits_expired, 1);
}

/// A model with a rate from spaced rounds is solved at once, even a full
/// heartbeat interval after its last round.
#[test]
fn a_clock_with_a_known_rate_is_not_held() {
    const HEARTBEAT_S: f64 = 90.0;
    const STEADY_ROUNDS: usize = 4;
    let config = MeshConfig::default();
    let membership = SessionMembership::default();
    let mut engine = engine_after_one_round(&config);
    let island = engine.islands.get_mut(&ISLAND_ID).expect("island");
    let mut last_round_s = FIRST_ROUND_S;
    for _ in 0..STEADY_ROUNDS {
        last_round_s += HEARTBEAT_S;
        for (id, filter) in island.filters.iter_mut() {
            filter.update(last_round_s, 0.01 * id.0 as f64, ROUND_SIGMA_S);
        }
    }
    island.last_reference_time_s = last_round_s;
    let localizer = RecordingLocalizer::default();
    let mut pipeline = pipeline_with_clips_in();
    pipeline.pending[0].centre_reference_s = last_round_s + HEARTBEAT_S;
    pipeline.tick(&context(&engine, &membership, &config), &localizer, RECEIVED_MS);
    assert_eq!(localizer.clock_sigmas_s.borrow().len(), 1, "a steady model was held");
    assert_eq!(pipeline.counters.calls_held_for_clock_model, 0);
}
