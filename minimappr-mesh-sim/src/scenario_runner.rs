//! Discrete-event loop on a virtual clock: device ticks, network
//! deliveries, calls reaching classifiers, chirp emissions and listen
//! windows, and scripted faults, all ordered by (time, insertion order).
//! Nothing reads the wall clock, so a seed reproduces a run exactly.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use minimappr_core::mesh::host_state_machine::FieldHostStateMachine;
use minimappr_core::mesh::local_tangent_plane::LocalTangentPlane;
use minimappr_core::mesh::node_state_machine::FieldNodeStateMachine;
use minimappr_core::protocol::{FieldMessage, NodeId, SessionId};

use crate::acoustic_world::AcousticWorld;
use crate::device_harness::SimulatedDevice;
use crate::run_metrics::{HostTenure, RunMetrics};
use crate::scenario::{ScenarioSpec, ScriptAction};
use crate::scenario_outcome::ScenarioOutcome;
use crate::simulated_network::SimulatedNetwork;
use crate::synthetic_clip::ArrivalClipLocalizer;

const MICROSECONDS_PER_SECOND: f64 = 1e6;
pub(crate) const MILLISECONDS_PER_SECOND: f64 = 1_000.0;
const SIMULATED_SESSION_ID: SessionId = SessionId([0x5a; 16]);

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum EventPayload {
    Tick,
    Deliver { to: usize, from: NodeId, frame: Vec<u8> },
    CallReachesClassifier { device: usize, call: usize },
    ChirpEmission { device: usize, round_id: u32 },
    ListenEnd { device: usize, round_id: u32 },
    Script(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EventKey {
    time_us: u64,
    sequence: u64,
}

pub struct ScenarioRun {
    pub spec: ScenarioSpec,
    pub seed: u64,
    pub world: AcousticWorld,
    pub network: SimulatedNetwork,
    pub devices: Vec<SimulatedDevice>,
    pub localizer: ArrivalClipLocalizer,
    pub metrics: RunMetrics,
    pub site_plane: LocalTangentPlane,
    pub now_s: f64,
    queue: BinaryHeap<Reverse<(EventKey, usize)>>,
    payloads: Vec<Option<EventPayload>>,
    sequence: u64,
    pub(crate) armed_host_kills: Vec<ScriptAction>,
}

impl ScenarioRun {
    pub fn new(mut spec: ScenarioSpec, seed: u64) -> Self {
        for device in &spec.devices {
            spec.mesh
                .island_sync
                .self_path_length_by_platform_m
                .insert(device.platform_name(), device.self_path_length_m);
        }
        let world = AcousticWorld::new(
            seed,
            spec.world.clone(),
            spec.devices.clone(),
            spec.calls.clone(),
            spec.noise_intervals.clone(),
            spec.duration_s,
        );
        let mut network = SimulatedNetwork::new(seed, spec.network.clone(), spec.devices.len());
        network.cell = vec![spec.initial_cell; spec.devices.len()];
        let host_node = NodeId(spec.initial_host_device as u32);
        let devices = spec
            .devices
            .iter()
            .enumerate()
            .map(|(index, device)| {
                let node = FieldNodeStateMachine::new(
                    NodeId(index as u32),
                    device.node_descriptor(),
                    SIMULATED_SESSION_ID,
                    host_node,
                    spec.mesh.clone(),
                    0,
                );
                SimulatedDevice::new(index, node)
            })
            .collect();
        let site_plane = LocalTangentPlane::new(spec.site_origin_deg.0, spec.site_origin_deg.1);
        let mut run = Self {
            seed,
            world,
            network,
            devices,
            localizer: ArrivalClipLocalizer { classifier_detection_snr_db: spec.world.classifier_detection_snr_db, ..ArrivalClipLocalizer::default() },
            metrics: RunMetrics::default(),
            site_plane,
            now_s: 0.0,
            queue: BinaryHeap::new(),
            payloads: Vec::new(),
            sequence: 0,
            armed_host_kills: Vec::new(),
            spec,
        };
        run.schedule_initial_events();
        run
    }

    pub(crate) fn schedule(&mut self, time_s: f64, payload: EventPayload) {
        let key = EventKey {
            time_us: (time_s.max(self.now_s) * MICROSECONDS_PER_SECOND).round() as u64,
            sequence: self.sequence,
        };
        self.sequence += 1;
        self.payloads.push(Some(payload));
        self.queue.push(Reverse((key, self.payloads.len() - 1)));
    }

    fn schedule_initial_events(&mut self) {
        let host_device = self.spec.initial_host_device;
        let (host, actions) = FieldHostStateMachine::new_session(
            NodeId(host_device as u32),
            self.spec.devices[host_device].node_descriptor(),
            self.spec.mesh.clone(),
            self.spec.transport_preference,
            self.spec.initial_cell,
            0,
        );
        self.devices[host_device].host = Some(host);
        self.metrics.host_tenures.push(HostTenure { device: host_device, host_term: 1, started_s: 0.0 });
        self.apply_host_actions(host_device, actions);
        self.schedule(0.0, EventPayload::Tick);
        for call in 0..self.world.calls.len() {
            for device in 0..self.devices.len() {
                let spec = &self.world.calls[call];
                let arrival = self.world.arrival_time(spec.time_s, spec.position_m, device);
                let ready = arrival + self.world.config.classification_latency_after_arrival_s;
                self.schedule(ready, EventPayload::CallReachesClassifier { device, call });
            }
        }
        for index in 0..self.spec.script.len() {
            let at = self.spec.script[index].at_s;
            self.schedule(at, EventPayload::Script(index));
        }
    }

    pub fn current_host_device(&self) -> Option<usize> {
        self.devices.iter().position(|device| device.host.is_some() && self.network.alive[device.index])
    }

    pub(crate) fn now_ms(&self) -> u64 {
        (self.now_s * MILLISECONDS_PER_SECOND).round() as u64
    }

    pub(crate) fn kill_device(&mut self, device: usize) {
        if !self.network.alive[device] {
            return;
        }
        self.network.alive[device] = false;
        self.metrics.killed_devices.push((self.now_s, device));
        if let Some(host) = self.devices[device].host.take() {
            self.metrics.retired_hosts.push((device, host));
        }
    }

    fn run_script(&mut self, index: usize) {
        let action = self.spec.script[index].action.clone();
        match action {
            ScriptAction::KillDevice(device) => self.kill_device(device),
            ScriptAction::KillHostDuringRound | ScriptAction::KillHostDuringClipFetch => self.armed_host_kills.push(action),
            ScriptAction::SetLossProbability(probability) => self.network.set_loss_override(probability),
            ScriptAction::SetCellQuality { device, quality } => {
                self.network.cell[device] = quality;
                if self.devices[device].host.is_some() {
                    self.deliver_host_input(device, minimappr_core::mesh::host_state_machine::HostInput::CellLinkChanged(quality));
                }
            }
            ScriptAction::DropFromRelay { device, until_s } => self.network.drop_from_relay(device, until_s),
            ScriptAction::Disconnect { device, until_s } => self.network.disconnect(device, until_s),
            ScriptAction::ExtraBulkDelay { device, extra_s } => self.network.set_extra_bulk_delay(device, extra_s),
            ScriptAction::InputRouteChange(device) => {
                self.deliver_node_status(device, minimappr_core::mesh::node_state_machine::DeviceStatusChange::InputRouteChanged)
            }
            ScriptAction::OutputRouteChange { device, chirp_speaker_pinned } => self.deliver_node_status(
                device,
                minimappr_core::mesh::node_state_machine::DeviceStatusChange::OutputRouteChanged { chirp_speaker_pinned },
            ),
            ScriptAction::EnableChirpDirection(device) => self.devices[device].chirp_direction_enabled = true,
            ScriptAction::ReachRelayMessageCap => {
                if !self.network.relay_capped {
                    self.network.relay_capped = true;
                    self.metrics.relay_cap_reached_s = Some(self.now_s);
                    if let Some(host_device) = self.current_host_device() {
                        self.deliver_host_input(host_device, minimappr_core::mesh::host_state_machine::HostInput::RelayRefusedSessionCap);
                    }
                }
            }
        }
    }

    fn check_armed_host_kills(&mut self) {
        let Some(host_device) = self.current_host_device() else { return };
        let host = self.devices[host_device].host.as_ref().expect("current host");
        let trigger = self.armed_host_kills.iter().position(|action| match action {
            ScriptAction::KillHostDuringRound => host.sync.round_in_flight(),
            ScriptAction::KillHostDuringClipFetch => host.calls.pending_call_count() > 0,
            _ => false,
        });
        if let Some(index) = trigger {
            self.armed_host_kills.remove(index);
            self.kill_device(host_device);
        }
    }

    fn dispatch(&mut self, payload: EventPayload) {
        match payload {
            EventPayload::Tick => {
                self.check_armed_host_kills();
                for device in 0..self.devices.len() {
                    if self.network.alive[device] {
                        self.tick_device(device);
                    }
                }
                if let Some(host_device) = self.current_host_device() {
                    let head = self.devices[host_device].host.as_ref().expect("host").event_log.head_sequence();
                    self.metrics.host_log_head_history.push((self.now_s, head));
                }
                let next = self.now_s + self.spec.tick_interval_ms as f64 / MILLISECONDS_PER_SECOND;
                self.schedule(next, EventPayload::Tick);
            }
            EventPayload::Deliver { to, from, frame } => self.deliver_frame(to, from, &frame),
            EventPayload::CallReachesClassifier { device, call } => self.call_reaches_classifier(device, call),
            EventPayload::ChirpEmission { device, round_id } => self.emit_chirp(device, round_id),
            EventPayload::ListenEnd { device, round_id } => self.end_listen(device, round_id),
            EventPayload::Script(index) => self.run_script(index),
        }
    }

    /// Run to the scenario's end and hand back everything for assertions.
    pub fn run(mut self) -> ScenarioOutcome {
        let end_us = (self.spec.duration_s * MICROSECONDS_PER_SECOND) as u64;
        while let Some(Reverse((key, index))) = self.queue.pop() {
            if key.time_us > end_us {
                break;
            }
            self.now_s = key.time_us as f64 / MICROSECONDS_PER_SECOND;
            if let Some(payload) = self.payloads[index].take() {
                self.dispatch(payload);
            }
        }
        ScenarioOutcome::from_run(self)
    }

    pub(crate) fn decode(frame: &[u8]) -> Option<FieldMessage> {
        match minimappr_core::protocol::decode_field_message_frame(frame) {
            Ok(minimappr_core::protocol::DecodedFrame::Known { message, .. }) => Some(message),
            _ => None,
        }
    }
}
