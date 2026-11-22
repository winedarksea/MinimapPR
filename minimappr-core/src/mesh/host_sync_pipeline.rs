//! Host chirp-round lifecycle: pick a due island, broadcast its schedule
//! (everyone listens, so other islands' members reveal a returning clump),
//! collect arrival reports, then update hearing, islands, clocks, layouts
//! and anchors. One round is in flight at a time across the session.

use std::collections::{BTreeMap, BTreeSet};

use crate::layout::gps_anchor_fit::GpsFix;
use crate::protocol::session_messages::{
    ChirpArrivalReport, ChirpSchedule, ChirpSlotAssignment, ClockModel, IslandMembership, LayoutModel,
    NodePositionEstimate,
};
use crate::protocol::wire_codes::MotionState;
use crate::protocol::{FieldMessage, NodeId};

use super::chirp_scheduler::{IslandChirpScheduler, RoundTrigger};
use super::config::{MeshConfig, DEVICE_SAMPLE_RATE_HZ};
use super::host_membership::SessionMembership;
use super::island_round_solver::{solve_island_round, IslandRoundInput};
use super::island_sync_engine::IslandSyncEngine;
use super::local_tangent_plane::LocalTangentPlane;
use super::{MeshMilliseconds, MessageRecipient, OutboundMessage};

/// Scheduler key for present nodes not yet in any island.
const PROVISIONAL_ISLAND_KEY: u32 = 0;
const MINIMUM_ROUND_EMITTERS: usize = 2;
const SAMPLES_PER_MILLISECOND: u64 = (DEVICE_SAMPLE_RATE_HZ as u64) / 1_000;

#[derive(Debug, Clone)]
struct RoundInFlight {
    round_id: u32,
    emission_order: Vec<NodeId>,
    listeners: BTreeSet<NodeId>,
    report_deadline_ms: MeshMilliseconds,
    reports: BTreeMap<NodeId, ChirpArrivalReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoundStartRecord {
    pub started_ms: MeshMilliseconds,
    pub round_id: u32,
    pub scheduler_key: u32,
    pub trigger: RoundTrigger,
    pub emitters: Vec<NodeId>,
}

/// Records the host logs after a round.
#[derive(Debug, Clone, PartialEq)]
pub enum SyncRecord {
    Islands(Vec<IslandMembership>),
    Layout(LayoutModel),
    Clock(ClockModel),
    NodeMuted(NodeId),
}

#[derive(Debug, Clone, Default)]
pub struct HostSyncPipeline {
    pub engine: IslandSyncEngine,
    schedulers: BTreeMap<u32, IslandChirpScheduler>,
    in_flight: Option<RoundInFlight>,
    next_round_id: u32,
    pub rounds_started: Vec<RoundStartRecord>,
    pub rounds_completed: u32,
    pub round_completion_ms: Vec<MeshMilliseconds>,
    pub muted_nodes: BTreeSet<NodeId>,
    /// Must stay 0: a muted emitter never enters a ranged pair.
    pub ranged_pairs_involving_muted: u64,
    pub local_plane: Option<LocalTangentPlane>,
    last_published_islands: Vec<IslandMembership>,
    /// Provisional nodes already included in a round (a lone node that
    /// stays lone must not trigger a round every tick).
    surveyed_provisional_nodes: BTreeSet<NodeId>,
    last_new_member_round_ms: Option<MeshMilliseconds>,
}

impl HostSyncPipeline {
    pub fn round_in_flight(&self) -> bool {
        self.in_flight.is_some()
    }

    /// Nodes in a multi-member island schedule with it; lone or new nodes
    /// are provisional and chirp in every round until they join an island.
    fn scheduler_key_of(&self, node: NodeId) -> u32 {
        self.engine
            .island_of(node)
            .filter(|island| island.members.len() >= MINIMUM_ROUND_EMITTERS)
            .map_or(PROVISIONAL_ISLAND_KEY, |island| island.island_id)
    }

    pub fn note_detection(&mut self, node: NodeId, now_ms: MeshMilliseconds, config: &MeshConfig) {
        let key = self.scheduler_key_of(node);
        self.schedulers.entry(key).or_default().note_detection(now_ms, &config.scheduler);
    }

    pub fn note_clock_invalid(&mut self, node: NodeId) {
        self.engine.invalidate_node_clock(node);
        let key = self.scheduler_key_of(node);
        self.schedulers.entry(key).or_default().note_clock_invalid();
    }

    pub fn on_report(&mut self, report: ChirpArrivalReport) {
        if let Some(round) = self.in_flight.as_mut() {
            if round.round_id == report.round_id {
                round.reports.insert(report.reporting_node_id, report);
            }
        }
    }

    /// GNSS fixes of members in the session's local plane.
    pub fn gps_fixes(&mut self, membership: &SessionMembership) -> BTreeMap<NodeId, GpsFix> {
        let mut fixes = BTreeMap::new();
        for member in membership.members.values() {
            let Some(fix) = &member.latest_gnss_fix else { continue };
            let plane = *self
                .local_plane
                .get_or_insert_with(|| LocalTangentPlane::new(fix.latitude_deg, fix.longitude_deg));
            let [east_m, north_m] = plane.east_north_from_geodetic(fix.latitude_deg, fix.longitude_deg);
            fixes.insert(
                member.node_id,
                GpsFix {
                    east_m,
                    north_m,
                    sigma_m: fix.horizontal_accuracy_m as f64,
                },
            );
        }
        fixes
    }

    fn group_members(&self, key: u32, present: &[NodeId]) -> Vec<NodeId> {
        present
            .iter()
            .copied()
            .filter(|node| self.scheduler_key_of(*node) == key)
            .collect()
    }

    /// Emitters of a round for `key`: its members plus every provisional node.
    fn round_participants(&self, key: u32, present: &[NodeId]) -> Vec<NodeId> {
        present
            .iter()
            .copied()
            .filter(|node| {
                let node_key = self.scheduler_key_of(*node);
                node_key == key || node_key == PROVISIONAL_ISLAND_KEY
            })
            .collect()
    }

    fn start_round(
        &mut self,
        key: u32,
        trigger: RoundTrigger,
        present: &[NodeId],
        membership: &SessionMembership,
        now_ms: MeshMilliseconds,
        config: &MeshConfig,
    ) -> Option<OutboundMessage> {
        let emitters: Vec<NodeId> = self
            .round_participants(key, present)
            .into_iter()
            .filter(|node| membership.members.get(node).is_some_and(|member| !member.listen_only))
            .collect();
        for node in &emitters {
            self.surveyed_provisional_nodes.insert(*node);
        }
        if emitters.len() < MINIMUM_ROUND_EMITTERS {
            self.schedulers.entry(key).or_default().mark_round_started(now_ms, trigger);
            return None;
        }
        let scheduler = &config.scheduler;
        let mut emission_order = emitters.clone();
        emission_order.push(emitters[0]);
        let round_id = self.next_round_id;
        self.next_round_id += 1;
        let start_ms = now_ms + scheduler.schedule_lead_time_ms;
        let slot_samples = scheduler.slot_duration_ms * SAMPLES_PER_MILLISECOND;
        let listen_end_ms = start_ms
            + emission_order.len() as u64 * scheduler.slot_duration_ms
            + scheduler.round_listen_tail_ms;
        self.in_flight = Some(RoundInFlight {
            round_id,
            emission_order: emission_order.clone(),
            listeners: present.iter().copied().collect(),
            report_deadline_ms: listen_end_ms + scheduler.round_report_timeout_ms,
            reports: BTreeMap::new(),
        });
        self.schedulers.entry(key).or_default().mark_round_started(now_ms, trigger);
        self.rounds_started.push(RoundStartRecord {
            started_ms: now_ms,
            round_id,
            scheduler_key: key,
            trigger,
            emitters,
        });
        let schedule = ChirpSchedule {
            island_id: key,
            round_id,
            // Coarse session clock in 48 kHz ticks: slots are 200+ ms apart,
            // so the OS clock anchor's millisecond accuracy is plenty.
            round_start_host_sample_index: start_ms * SAMPLES_PER_MILLISECOND,
            slots: emission_order
                .iter()
                .enumerate()
                .map(|(slot, emitter)| ChirpSlotAssignment {
                    emitter_node_id: *emitter,
                    slot_offset_samples: (slot as u64 * slot_samples) as u32,
                    chirp_low_hz: scheduler.chirp_band_hz[0],
                    chirp_high_hz: scheduler.chirp_band_hz[1],
                    chirp_code_id: emitter.0 as u16,
                })
                .collect(),
        };
        Some(OutboundMessage::control(MessageRecipient::AllNodes, FieldMessage::ChirpSchedule(schedule)))
    }

    fn finish_round(
        &mut self,
        round: RoundInFlight,
        present: &[NodeId],
        membership: &SessionMembership,
        now_ms: MeshMilliseconds,
        config: &MeshConfig,
    ) -> Vec<SyncRecord> {
        let mut records = Vec::new();
        self.rounds_completed += 1;
        self.round_completion_ms.push(now_ms);
        let sync = &config.island_sync;
        self.engine.record_hearing(&round.reports, &round.emission_order, now_ms);
        for report in round.reports.values() {
            if report.own_chirp_inaudible {
                if self.muted_nodes.insert(report.reporting_node_id) {
                    records.push(SyncRecord::NodeMuted(report.reporting_node_id));
                }
            } else if report.own_emission_device_sample_index.is_some() {
                self.muted_nodes.remove(&report.reporting_node_id);
            }
        }
        self.engine.listen_only_nodes = membership
            .members
            .values()
            .filter(|member| member.listen_only)
            .map(|member| member.node_id)
            .collect();
        self.engine.recompute_islands(present, now_ms, sync);
        let fixes = self.gps_fixes(membership);
        let self_paths = membership.self_path_lengths();
        let island_ids: Vec<u32> = self.engine.islands.keys().copied().collect();
        for island_id in island_ids {
            let island = &self.engine.islands[&island_id];
            if island.members.len() < 2 {
                continue;
            }
            // A reported capture gap inside a node's arrival span corrupts
            // its ranges too, not only its clock: leave it out of this round.
            let gapped: BTreeSet<NodeId> = round
                .reports
                .values()
                .filter(|report| {
                    let gap = membership.members.get(&report.reporting_node_id).and_then(|m| m.last_capture_gap_device_sample);
                    let span = report.arrivals.iter().map(|a| a.arrival_device_sample_index);
                    match (gap, span.clone().min(), span.max()) {
                        (Some(gap), Some(first), Some(last)) => gap >= first && gap <= last,
                        _ => false,
                    }
                })
                .map(|report| report.reporting_node_id)
                .collect();
            let mut members: Vec<NodeId> = island.members.iter().copied().filter(|node| !gapped.contains(node)).collect();
            let preferred_reference = island.reference_node;
            let solve = |members: Vec<NodeId>| {
                solve_island_round(
                    &IslandRoundInput {
                        members,
                        reports: &round.reports,
                        emission_order: &round.emission_order,
                        self_path_lengths_m: &self_paths,
                        muted_emitters: &self.muted_nodes,
                        preferred_reference,
                    },
                    sync,
                )
            };
            let Some(mut solution) = solve(members.clone()) else { continue };
            // A silent stall shows as a clock-offset jump; that node's ranges
            // in this round are suspect, so re-solve without it and rebuild
            // its clock model from scratch (caught within one round).
            let stalled = self.engine.nodes_with_silent_gap(island_id, &solution, sync);
            if !stalled.is_empty() {
                members.retain(|node| !stalled.contains(node));
                for node in &stalled {
                    self.engine.invalidate_node_clock(*node);
                }
                match solve(members) {
                    Some(resolved) => solution = resolved,
                    None => continue,
                }
            }
            self.ranged_pairs_involving_muted += solution
                .ranged_pairs
                .iter()
                .filter(|(a, b, _)| self.muted_nodes.contains(a) || self.muted_nodes.contains(b))
                .count() as u64;
            self.engine.apply_round_solution(island_id, &solution, now_ms, sync);
            self.engine.refresh_anchor(island_id, &fixes, sync);
            records.extend(self.island_model_records(island_id, now_ms, sync));
        }
        let islands: Vec<IslandMembership> = self
            .engine
            .islands
            .values()
            .map(|island| IslandMembership {
                island_id: island.island_id,
                membership_epoch: island.tracker_epoch as u32,
                member_node_ids: island.members.iter().copied().collect(),
            })
            .collect();
        if islands != self.last_published_islands {
            self.last_published_islands = islands.clone();
            records.push(SyncRecord::Islands(islands));
        }
        records
    }

    fn island_model_records(
        &self,
        island_id: u32,
        now_ms: MeshMilliseconds,
        sync: &super::config::IslandSyncConfig,
    ) -> Vec<SyncRecord> {
        let island = &self.engine.islands[&island_id];
        let mut records = Vec::new();
        if let Some(layout) = &island.layout {
            let (origin_latitude_deg, origin_longitude_deg) = self
                .local_plane
                .map_or((0.0, 0.0), |plane| (plane.origin_latitude_deg, plane.origin_longitude_deg));
            let node_positions = layout
                .local_positions_m
                .keys()
                .filter_map(|node| {
                    let estimate = self.engine.node_position_for_solve(*node, now_ms, false, sync)?;
                    let std = estimate.std_m as f32;
                    Some(NodePositionEstimate {
                        node_id: *node,
                        east_north_up_m: estimate.position_m.map(|v| v as f32),
                        position_standard_deviation_m: [std, std, std],
                    })
                })
                .collect();
            records.push(SyncRecord::Layout(LayoutModel {
                island_id,
                layout_epoch: island.layout_epoch,
                origin_latitude_deg,
                origin_longitude_deg,
                origin_altitude_m: 0.0,
                node_positions,
            }));
        }
        if let Some(reference_node) = island.reference_node {
            records.push(SyncRecord::Clock(ClockModel {
                island_id,
                clock_epoch: island.tracker_epoch as u32,
                reference_node_id: reference_node,
                node_clocks: island
                    .members
                    .iter()
                    .filter_map(|node| Some(self.engine.clock_mapping(*node)?.to_wire(*node)))
                    .collect(),
            }));
        }
        records
    }

    /// Advance: finish a round whose reports are in, or start a due one.
    pub fn tick(
        &mut self,
        present: &[NodeId],
        membership: &SessionMembership,
        now_ms: MeshMilliseconds,
        config: &MeshConfig,
    ) -> (Vec<OutboundMessage>, Vec<SyncRecord>) {
        if let Some(round) = &self.in_flight {
            let all_reported = round.listeners.iter().all(|node| round.reports.contains_key(node));
            if all_reported || now_ms >= round.report_deadline_ms {
                let round = self.in_flight.take().expect("checked above");
                return (Vec::new(), self.finish_round(round, present, membership, now_ms, config));
            }
            return (Vec::new(), Vec::new());
        }
        let mut keys: BTreeSet<u32> = present.iter().map(|node| self.scheduler_key_of(*node)).collect();
        keys.retain(|key| *key == PROVISIONAL_ISLAND_KEY || self.engine.islands.contains_key(key));
        for key in &keys {
            let members = self.group_members(*key, present);
            let group_is_still = !members.is_empty()
                && members.iter().all(|node| {
                    membership.members.get(node).is_some_and(|m| m.motion_state == MotionState::Still)
                });
            self.schedulers.entry(*key).or_default().note_group_motion(group_is_still);
        }
        let surveyed = &self.surveyed_provisional_nodes;
        self.surveyed_provisional_nodes = surveyed
            .iter()
            .copied()
            .filter(|node| self.scheduler_key_of(*node) == PROVISIONAL_ISLAND_KEY)
            .collect();
        let has_new_member = present.iter().any(|node| {
            self.scheduler_key_of(*node) == PROVISIONAL_ISLAND_KEY && !self.surveyed_provisional_nodes.contains(node)
        });
        let new_member_gap_elapsed = self
            .last_new_member_round_ms
            .is_none_or(|last| now_ms.saturating_sub(last) >= config.scheduler.new_member_round_minimum_gap_ms);
        if has_new_member && new_member_gap_elapsed {
            // Survey joiners with the largest island so hearing is measured both ways.
            let key = self
                .engine
                .islands
                .values()
                .filter(|island| island.members.len() >= MINIMUM_ROUND_EMITTERS)
                .max_by_key(|island| island.members.len())
                .map_or(PROVISIONAL_ISLAND_KEY, |island| island.island_id);
            if self.round_participants(key, present).len() >= MINIMUM_ROUND_EMITTERS {
                self.last_new_member_round_ms = Some(now_ms);
                if let Some(message) = self.start_round(key, RoundTrigger::NewMember, present, membership, now_ms, config) {
                    return (vec![message], Vec::new());
                }
            }
        }
        for key in keys {
            if self.round_participants(key, present).len() < MINIMUM_ROUND_EMITTERS {
                continue;
            }
            let due = self.schedulers.entry(key).or_default().due(now_ms, &config.scheduler);
            if let Some(trigger) = due {
                if let Some(message) = self.start_round(key, trigger, present, membership, now_ms, config) {
                    return (vec![message], Vec::new());
                }
            }
        }
        (Vec::new(), Vec::new())
    }
}
