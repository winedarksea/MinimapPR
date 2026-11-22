//! Sync and geometry scenarios: input route change, acoustic island split
//! and merge, walking at call time, mirrored layout, muted phone, earbuds.
//! Arrival-time abstraction: chirps are arrival times on each phone's own
//! sample clock fed through the real core ranging, clock graph, clock
//! filter, island, MDS layout and GPS anchor code.

use std::collections::BTreeSet;

use minimappr_core::mesh::node_state_machine::NodeUserNotice;
use minimappr_core::protocol::wire_codes::NodeStatusChangeCode;
use minimappr_core::protocol::{FieldMessage, NodeId};

use crate::device_spec::{DeviceSpec, Trajectory};
use crate::invariants::ensure;
use crate::scenario::{ring_of_phones, ScenarioSpec, ScriptAction};

use super::call_scenarios::{healthy_path_checks, INSIDE_LAYOUT_GATE_M};
use super::{report, run_checked, ScenarioReport};

const MILLISECONDS_PER_SECOND: f64 = 1_000.0;
const PHONE_HEIGHT_M: f64 = 1.4;

fn node(device: usize) -> NodeId {
    NodeId(device as u32)
}

/// One phone's input route changes: its clock model is invalidated at once
/// and its clips are excluded until the next good round includes it.
pub fn route_change(seed: u64) -> Result<ScenarioReport, String> {
    const CHANGED_PHONE: usize = 2;
    const CHANGE_S: f64 = 28.0;
    // An earlier invalidation spends the triggered-round allowance, so the
    // resync round for the changed phone comes a few seconds later.
    let spec = ScenarioSpec::new("route_change", 75.0, ring_of_phones(5, [0.0, 0.0], 12.0, seed))
        .at(25.0, ScriptAction::InputRouteChange(3))
        .at(CHANGE_S, ScriptAction::InputRouteChange(CHANGED_PHONE))
        .call(28.5, [2.0, 3.0, 6.0])
        .call(55.0, [-3.0, -2.0, 7.0]);
    report("route_change", seed, (|| {
        let outcome = run_checked(spec, seed)?;
        let (_, host) = outcome.active_host().ok_or("no host")?;
        let change_ms = (CHANGE_S * MILLISECONDS_PER_SECOND) as u64;
        let resync_ms = host
            .sync
            .round_completion_ms
            .iter()
            .copied()
            .find(|completed| *completed > change_ms)
            .ok_or("no round after the route change")?;
        let records = outcome.call_records();
        let in_gap: Vec<_> = records.iter().filter(|r| r.received_ms > change_ms && r.received_ms < resync_ms).collect();
        for record in &in_gap {
            ensure(!record.requested_nodes.contains(&node(CHANGED_PHONE)), || format!("call {} fetched the invalidated phone before resync", record.call_id))?;
        }
        let rejoined = records.iter().any(|r| r.received_ms > resync_ms && r.clip_nodes.contains(&node(CHANGED_PHONE)));
        ensure(rejoined, || "changed phone never contributed after its resync round".into())?;
        let worst = healthy_path_checks(&outcome, 2, Some(INSIDE_LAYOUT_GATE_M))?;
        Ok(format!("{} call(s) in the gap excluded the phone, resync at {:.1} s, worst {worst:.2} m", in_gap.len(), resync_ms as f64 / MILLISECONDS_PER_SECOND))
    })())
}

/// Eight phones split into two 4-phone clumps ~150 m apart and later walk
/// back (the strategy says 3 + 3; a 3-phone island has only two TDOAs, so with
/// unknown bird height its fix is often an honest "unobservable"): two
/// islands with separate clock domains localize independently, no TDOA
/// pair crosses them, and they merge into one fresh domain within one
/// heartbeat interval of hearing each other again.
const SPLIT_CLUMP_B: [usize; 4] = [4, 5, 6, 7];
const SPLIT_PHONE_COUNT: usize = 8;
const SPLIT_BACK_TOGETHER_S: f64 = 175.0;

pub fn island_split_merge_spec(seed: u64) -> ScenarioSpec {
    // The plan's 70 m assumes real phone speakers; this world's omnidirectional
    // 85 dB chirps still close two-way at ~90 m, so the clumps go further.
    const CLUMP_B_OFFSET_M: [f64; 2] = [150.0, 0.0];
    let mut devices = ring_of_phones(SPLIT_PHONE_COUNT, [0.0, 0.0], 8.0, seed);
    for device in SPLIT_CLUMP_B {
        let home = devices[device].trajectory.position_at(0.0);
        let away = [home[0] + CLUMP_B_OFFSET_M[0], home[1] + CLUMP_B_OFFSET_M[1], home[2]];
        devices[device].trajectory = Trajectory::stationary(home).then_move(30.0, 75.0, away).then_move(130.0, SPLIT_BACK_TOGETHER_S, home);
    }
    ScenarioSpec::new("island_split_merge", 290.0, devices)
        .call(100.0, [2.0, 3.0, 5.0])
        .call(112.0, [CLUMP_B_OFFSET_M[0] + 1.0, -2.0, 6.0])
        .call(275.0, [-2.0, 1.0, 6.0])
}

pub fn island_split_merge(seed: u64) -> Result<ScenarioReport, String> {
    const CLUMP_B: [usize; 4] = SPLIT_CLUMP_B;
    const BACK_TOGETHER_S: f64 = SPLIT_BACK_TOGETHER_S;
    let spec = island_split_merge_spec(seed);
    let heartbeat_s = spec.mesh.scheduler.heartbeat_round_interval_ms as f64 / MILLISECONDS_PER_SECOND;
    report("island_split_merge", seed, (|| {
        let outcome = run_checked(spec, seed)?;
        let memberships: Vec<_> = outcome
            .host_log_records()
            .into_iter()
            .filter_map(|record| match record {
                FieldMessage::IslandMembership(island) => Some(island),
                _ => None,
            })
            .collect();
        let clump_a: BTreeSet<NodeId> = (0..SPLIT_PHONE_COUNT - CLUMP_B.len()).map(node).collect();
        let clump_b: BTreeSet<NodeId> = CLUMP_B.iter().map(|d| node(*d)).collect();
        let as_set = |ids: &[NodeId]| ids.iter().copied().collect::<BTreeSet<_>>();
        let split_epochs: BTreeSet<u32> = memberships
            .iter()
            .filter(|m| as_set(&m.member_node_ids) == clump_a || as_set(&m.member_node_ids) == clump_b)
            .map(|m| m.membership_epoch)
            .collect();
        ensure(split_epochs.len() >= 2, || format!("never split into the two clumps: {memberships:?}"))?;
        let (_, host) = outcome.active_host().ok_or("no host")?;
        let final_islands: Vec<_> = host.sync.engine.islands.values().filter(|i| i.members.len() == SPLIT_PHONE_COUNT).collect();
        ensure(final_islands.len() == 1, || "clumps never merged back into one island".into())?;
        ensure(!split_epochs.contains(&(final_islands[0].tracker_epoch as u32)), || "merged island reused a pre-merge clock epoch".into())?;
        let merge_record_index = memberships.iter().rposition(|m| m.member_node_ids.len() == SPLIT_PHONE_COUNT).ok_or("no merged membership logged")?;
        let merged_epoch = memberships[merge_record_index].membership_epoch;
        let merge_s = host
            .sync
            .round_completion_ms
            .iter()
            .map(|ms| *ms as f64 / MILLISECONDS_PER_SECOND)
            .filter(|t| *t > BACK_TOGETHER_S)
            .find(|_| true)
            .unwrap_or(f64::INFINITY);
        ensure(merge_s <= BACK_TOGETHER_S + heartbeat_s, || format!("merge round at {merge_s:.0} s, back together at {BACK_TOGETHER_S}"))?;
        let records = outcome.call_records();
        // A 3-phone island has two TDOAs: a fix, or an explicit
        // unobservable/ambiguous failure. Never a pair across the islands.
        let fixes = outcome.scored_fixes();
        let mut split_fixes = 0;
        for (call_index, clump) in [(0usize, &clump_a), (1, &clump_b)] {
            for fix in fixes.iter().filter(|f| f.call_index == call_index) {
                ensure(fix.node_ids.iter().all(|n| clump.contains(n)), || format!("split call {call_index} used nodes {:?} outside its island", fix.node_ids))?;
                split_fixes += 1;
            }
        }
        ensure(split_fixes >= 2, || format!("islands did not both localize while split: {:?}", records.iter().map(|r| &r.failure).collect::<Vec<_>>()))?;
        let worst = healthy_path_checks(&outcome, 3, None)?;
        Ok(format!("split epochs {split_epochs:?} → merged epoch {merged_epoch}, {split_fixes} island-local fixes while split, worst {worst:.2} m"))
    })())
}

/// The group walks while a bird sings and stops 2 s later; the
/// post-detection round lands after the stop. Walking phones keep the
/// pre-call layout (never the post-call one), the clock still applies
/// backward, and no fix is worse than its reported region.
pub fn walking_at_call(seed: u64) -> Result<ScenarioReport, String> {
    const WALK_START_S: f64 = 20.0;
    const STOP_S: f64 = 42.0;
    const CALL_S: f64 = 40.0;
    const WALK_DISTANCE_M: f64 = 26.4;
    const STILL_PHONE: usize = 0;
    const RELATIVE_DRIFT_M: f64 = 1.0;
    let mut devices = ring_of_phones(5, [0.0, 0.0], 6.0, seed);
    for (device, spec) in devices.iter_mut().enumerate() {
        if device == STILL_PHONE {
            continue;
        }
        let start = spec.trajectory.position_at(0.0);
        let drift = crate::seeded_random::hashed_gaussian(seed, "walk_drift", device as u64, 0) * RELATIVE_DRIFT_M;
        let end = [start[0] + WALK_DISTANCE_M, start[1] + drift, start[2]];
        spec.trajectory = Trajectory::stationary(start).then_move(WALK_START_S, STOP_S, end);
    }
    let group_at_call = [WALK_DISTANCE_M * (CALL_S - WALK_START_S) / (STOP_S - WALK_START_S), 0.0];
    let spec = ScenarioSpec::new("walking_at_call", 70.0, devices)
        .call(10.0, [2.0, 2.0, 6.0])
        .call(CALL_S, [group_at_call[0] + 2.0, group_at_call[1] + 3.0, 6.0]);
    report("walking_at_call", seed, (|| {
        let outcome = run_checked(spec, seed)?;
        let records = outcome.call_records();
        let walking_call = records
            .iter()
            .find(|r| r.fix.is_some() && !r.walked_nodes.is_empty())
            .ok_or_else(|| format!("no fix used walking phones: {:?}", records.iter().map(|r| (&r.walked_nodes, &r.failure)).collect::<Vec<_>>()))?;
        ensure(!walking_call.walked_nodes.contains(&node(STILL_PHONE)), || "the still phone was treated as walking".into())?;
        let worst = healthy_path_checks(&outcome, 2, None)?;
        Ok(format!("{} walking phones on pre-call layout, worst {worst:.2} m inside its region", walking_call.walked_nodes.len()))
    })())
}

/// A small layout under bad GPS: the anchor fit either picks the right
/// chirality or reports low confidence; once a stereo phone's chirp bearing
/// arrives, chirality is locked and no bird is ever mirrored.
pub fn mirrored_layout_spec(seed: u64) -> ScenarioSpec {
    const STEREO_PHONE: usize = 3;
    const BAD_GPS_SIGMA_M: f64 = 6.0;
    const CUE_ENABLED_S: f64 = 35.0;
    let mut devices = ring_of_phones(4, [0.0, 0.0], 5.0, seed);
    for device in devices.iter_mut() {
        device.gps_error_sigma_m = BAD_GPS_SIGMA_M;
    }
    devices[STEREO_PHONE].reports_chirp_direction = true;
    ScenarioSpec::new("mirrored_layout", 90.0, devices)
        .call(20.0, [2.0, 1.0, 6.0])
        .call(28.0, [-1.0, 3.0, 5.0])
        .at(CUE_ENABLED_S, ScriptAction::EnableChirpDirection(STEREO_PHONE))
        .call(40.0, [1.0, -2.0, 7.0])
        .call(60.0, [-2.0, -1.0, 6.0])
        .call(75.0, [3.0, 2.0, 5.0])
}

pub fn mirrored_layout(seed: u64) -> Result<ScenarioReport, String> {
    let spec = mirrored_layout_spec(seed);
    report("mirrored_layout", seed, (|| {
        let outcome = run_checked(spec, seed)?;
        let fixes = outcome.scored_fixes();
        crate::invariants::no_confident_mirror(&fixes)?;
        let locked_fixes = fixes.iter().filter(|fix| fix.chirality_locked).count();
        ensure(locked_fixes >= 1, || "no fix after the chirality cue".into())?;
        let flagged = fixes.iter().filter(|fix| fix.layout_mirrored == Some(true)).count();
        let worst = healthy_path_checks(&outcome, 3, None)?;
        Ok(format!("{locked_fixes} fixes after lock (none mirrored), {flagged} flagged low-confidence mirrors before, worst {worst:.2} m"))
    })())
}

/// One phone's chirps are silent: the muted flag is raised on its first
/// slot and it is excluded from ranging (never mis-ranged); the other four
/// still localize.
pub fn muted_phone(seed: u64) -> Result<ScenarioReport, String> {
    const MUTED: usize = 3;
    let mut devices = ring_of_phones(5, [0.0, 0.0], 12.0, seed);
    devices[MUTED].muted_speaker = true;
    let spec = ScenarioSpec::new("muted_phone", 60.0, devices).call(20.0, [2.0, 2.0, 6.0]).call(40.0, [-3.0, 1.0, 7.0]);
    report("muted_phone", seed, (|| {
        let outcome = run_checked(spec, seed)?;
        let (_, host) = outcome.active_host().ok_or("no host")?;
        let first_slot_ms = host.sync.rounds_started.iter().find(|r| r.emitters.contains(&node(MUTED))).map(|r| r.started_ms).ok_or("muted phone never got a slot")?;
        let first_round_end_s = host
            .sync
            .round_completion_ms
            .iter()
            .find(|completed| **completed >= first_slot_ms)
            .map(|ms| *ms as f64 / MILLISECONDS_PER_SECOND)
            .ok_or("its first round never completed")?;
        let notice_s = outcome.metrics.first_notice(MUTED, NodeUserNotice::MediaVolumeMuted).ok_or("muted phone never warned")?;
        ensure(notice_s <= first_round_end_s, || format!("muted banner at {notice_s:.1} s, first round ended {first_round_end_s:.1} s"))?;
        let logged = outcome.host_log_records().iter().any(|record| {
            matches!(record, FieldMessage::NodeStatusChange(change) if change.node_id == node(MUTED) && change.change == NodeStatusChangeCode::MediaVolumeMuted)
        });
        ensure(logged, || "muted flag not logged".into())?;
        ensure(host.sync.ranged_pairs_involving_muted == 0, || format!("{} ranged pairs involved the muted phone", host.sync.ranged_pairs_involving_muted))?;
        ensure(host.sync.engine.island_of(node(MUTED)).is_none_or(|island| island.members.len() == 1), || "muted phone shares a clock domain".into())?;
        let worst = healthy_path_checks(&outcome, 2, Some(INSIDE_LAYOUT_GATE_M))?;
        Ok(format!("muted flag at {notice_s:.1} s, 0 muted ranges, worst {worst:.2} m"))
    })())
}

/// Bluetooth earbuds connect: on Android the chirp stays pinned to the
/// speaker and the clock model is kept; a native iOS phone becomes
/// listen-only (no chirp slot) and also keeps its clock model.
pub fn earbuds(seed: u64) -> Result<ScenarioReport, String> {
    const ANDROID_PHONE: usize = 2;
    const IOS_PHONE: usize = 4;
    const EARBUDS_S: f64 = 30.0;
    let mut devices = ring_of_phones(5, [0.0, 0.0], 10.0, seed);
    let ios_position = devices[IOS_PHONE].trajectory.position_at(0.0);
    devices[IOS_PHONE] = DeviceSpec::ios_native_at([ios_position[0], ios_position[1], PHONE_HEIGHT_M]);
    let spec = ScenarioSpec::new("earbuds", 85.0, devices)
        .call(20.0, [2.0, 2.0, 6.0])
        .at(EARBUDS_S, ScriptAction::OutputRouteChange { device: ANDROID_PHONE, chirp_speaker_pinned: true })
        .at(EARBUDS_S, ScriptAction::OutputRouteChange { device: IOS_PHONE, chirp_speaker_pinned: false })
        .call(50.0, [-3.0, 1.0, 7.0])
        .call(70.0, [1.0, -3.0, 5.0]);
    report("earbuds", seed, (|| {
        let outcome = run_checked(spec, seed)?;
        let (_, host) = outcome.active_host().ok_or("no host")?;
        let after: Vec<_> = host
            .sync
            .rounds_started
            .iter()
            .filter(|round| round.started_ms as f64 / MILLISECONDS_PER_SECOND > EARBUDS_S + 1.0)
            .collect();
        ensure(!after.is_empty(), || "no rounds after the earbuds connected".into())?;
        ensure(after.iter().all(|r| r.emitters.contains(&node(ANDROID_PHONE))), || "Android phone lost its chirp slot".into())?;
        ensure(after.iter().all(|r| !r.emitters.contains(&node(IOS_PHONE))), || "iOS phone still assigned a chirp slot".into())?;
        for device in [ANDROID_PHONE, IOS_PHONE] {
            ensure(host.sync.engine.clock_mapping(node(device)).is_some(), || format!("device {device} lost its clock model"))?;
        }
        ensure(outcome.metrics.first_notice(IOS_PHONE, NodeUserNotice::ClockInvalidated).is_none(), || "iOS clock invalidated by an output-only change".into())?;
        ensure(outcome.metrics.first_notice(IOS_PHONE, NodeUserNotice::ListenOnlyChirps).is_some(), || "iOS phone not told it is listen-only".into())?;
        let worst = healthy_path_checks(&outcome, 3, Some(INSIDE_LAYOUT_GATE_M))?;
        Ok(format!("Android pinned, iOS listen-only, clocks kept, worst {worst:.2} m"))
    })())
}
