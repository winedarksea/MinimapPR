//! The shell side of each device: feeds world events into the node/host
//! state machines and carries out their actions (network sends, chirps,
//! listen windows, takeover, transport switches, step-down).

use minimappr_core::mesh::host_state_machine::{FieldHostStateMachine, HostAction, HostInput};
use minimappr_core::mesh::node_state_machine::{DeviceStatusChange, LocalClassification, NodeAction, NodeInput};
use minimappr_core::mesh::{MessageRecipient, OutboundMessage, TrafficClass};
use minimappr_core::protocol::mesh_messages::ChirpDirectionReport;
use minimappr_core::protocol::session_messages::GnssFix;
use minimappr_core::protocol::wire_codes::{FrequencyBandHz, MotionState, SessionTransportCode, SpeciesScore};
use minimappr_core::protocol::{FieldMessage, NodeId};

use crate::acoustic_world::DEVICE_SAMPLE_RATE_HZ;
use crate::device_harness::DeviceAudioView;
use crate::run_metrics::HostTenure;
use crate::scenario_runner::{EventPayload, ScenarioRun, MILLISECONDS_PER_SECOND};
use crate::seeded_random::hashed_gaussian;
use crate::simulated_network::ActiveTransport;

/// Headroom changes smaller than this are not re-reported.
const HEADROOM_REPORT_STEP: f64 = 0.02;
/// Per-fix GNSS jitter on top of the device's slowly varying bias.
const GNSS_FIX_JITTER_M: f64 = 0.5;
const CLASSIFIER_MODEL_ID: u16 = 3;
const CALL_BAND_LOW_FRACTION: f64 = 0.5;
const CALL_BAND_HIGH_FRACTION: f64 = 2.0;

fn is_host_bound(message: &FieldMessage) -> bool {
    matches!(
        message,
        FieldMessage::JoinRequest(_)
            | FieldMessage::Heartbeat(_)
            | FieldMessage::DetectionEvent(_)
            | FieldMessage::QueuedDetectionUpload(_)
            | FieldMessage::ChirpArrivalReport(_)
            | FieldMessage::ReadinessAnswer(_)
            | FieldMessage::ClipPayload(_)
            | FieldMessage::NodeStatusChange(_)
            | FieldMessage::EventLogCatchUpRequest(_)
            | FieldMessage::ChirpDirectionReport(_)
    )
}

impl ScenarioRun {
    pub(crate) fn send_message(&mut self, from: usize, outbound: OutboundMessage) {
        let Ok(frame) = outbound.message.encode_frame() else { return };
        let kind = outbound.message.message_kind_id();
        let is_clip = matches!(outbound.message, FieldMessage::ClipPayload(_));
        let bulk = outbound.traffic_class == TrafficClass::Bulk;
        let recipients: Vec<usize> = match outbound.recipient {
            MessageRecipient::Node(node) => vec![node.0 as usize],
            MessageRecipient::AllNodes => (0..self.devices.len()).collect(),
        };
        let was_capped = self.network.relay_capped;
        for to in recipients {
            if to >= self.devices.len() {
                continue;
            }
            if let Some(arrival) = self.network.send(from, to, frame.len(), kind, bulk, is_clip, &self.world, self.now_s) {
                self.schedule(arrival, EventPayload::Deliver { to, from: NodeId(from as u32), frame: frame.clone() });
            }
        }
        if !was_capped && self.network.relay_capped {
            self.metrics.relay_cap_reached_s = Some(self.now_s);
            if let Some(host_device) = self.current_host_device() {
                self.deliver_host_input(host_device, HostInput::RelayRefusedSessionCap);
            }
        }
    }

    pub(crate) fn deliver_node_input(&mut self, device: usize, input: NodeInput) {
        if !self.network.alive[device] {
            return;
        }
        let now_ms = self.now_ms();
        let sample_now = self.world.devices[device].sample_index_at(self.now_s).max(0.0) as u64;
        let view = DeviceAudioView { world: &self.world, device, now_s: self.now_s };
        let actions = self.devices[device].node.handle(now_ms, sample_now, input, &view);
        self.apply_node_actions(device, actions);
    }

    pub(crate) fn deliver_node_status(&mut self, device: usize, change: DeviceStatusChange) {
        self.deliver_node_input(device, NodeInput::DeviceStatus(change));
    }

    pub(crate) fn deliver_host_input(&mut self, device: usize, input: HostInput) {
        let now_ms = self.now_ms();
        let Some(host) = self.devices[device].host.as_mut() else { return };
        let actions = host.handle(now_ms, input, &self.localizer);
        self.apply_host_actions(device, actions);
    }

    fn apply_node_actions(&mut self, device: usize, actions: Vec<NodeAction>) {
        for action in actions {
            match action {
                NodeAction::Send(outbound) => self.send_message(device, outbound),
                NodeAction::EmitChirp { round_id, emit_at_ms, .. } => {
                    self.schedule(emit_at_ms as f64 / MILLISECONDS_PER_SECOND, EventPayload::ChirpEmission { device, round_id })
                }
                NodeAction::ListenForChirpRound { round_id, listen_until_ms } => {
                    self.devices[device].listening_rounds.insert(round_id);
                    self.schedule(listen_until_ms as f64 / MILLISECONDS_PER_SECOND, EventPayload::ListenEnd { device, round_id });
                }
                NodeAction::BecomeHost { host_term, reason } => {
                    if self.devices[device].host.is_some() {
                        continue;
                    }
                    let entries = self.devices[device].node.mirror.entries().to_vec();
                    let (host, host_actions) = FieldHostStateMachine::take_over(
                        NodeId(device as u32),
                        &entries,
                        host_term,
                        reason,
                        self.spec.mesh.clone(),
                        self.spec.transport_preference,
                        self.network.cell[device],
                        self.now_ms(),
                    );
                    self.metrics.host_tenures.push(HostTenure { device, host_term: host.host_term, started_s: self.now_s });
                    self.devices[device].host = Some(host);
                    self.apply_host_actions(device, host_actions);
                }
                NodeAction::UserNotice(notice) => self.metrics.notices.push((self.now_s, device, notice)),
            }
        }
    }

    pub(crate) fn apply_host_actions(&mut self, device: usize, actions: Vec<HostAction>) {
        for action in actions {
            match action {
                HostAction::Send(outbound) => self.send_message(device, outbound),
                HostAction::SwitchTransport(decision) => {
                    let transport = match decision.transport {
                        SessionTransportCode::CloudRelay => ActiveTransport::Relay,
                        SessionTransportCode::LocalHotspot => match self.network.transport {
                            ActiveTransport::LocalHotspot { host_device, started_s } if host_device == device => {
                                ActiveTransport::LocalHotspot { host_device, started_s }
                            }
                            _ => ActiveTransport::LocalHotspot { host_device: device, started_s: self.now_s },
                        },
                        _ => ActiveTransport::Unavailable,
                    };
                    if transport != self.network.transport {
                        self.network.transport = transport;
                        self.metrics.transport_changes.push((self.now_s, transport));
                    }
                }
                HostAction::ShedLocalClassification => self.deliver_node_status(device, DeviceStatusChange::ShedClassification),
                HostAction::StepDown { .. } => {
                    if let Some(host) = self.devices[device].host.take() {
                        self.metrics.retired_hosts.push((device, host));
                    }
                }
            }
        }
    }

    pub(crate) fn deliver_frame(&mut self, to: usize, from: NodeId, frame: &[u8]) {
        if !self.network.alive[to] {
            return;
        }
        let Some(message) = Self::decode(frame) else { return };
        let to_host = is_host_bound(&message) || matches!(message, FieldMessage::HostAnnouncement(_));
        let to_node = !is_host_bound(&message);
        if to_host && self.devices[to].host.is_some() {
            self.deliver_host_input(to, HostInput::MessageReceived { from, message: message.clone() });
        }
        if to_node {
            self.deliver_node_input(to, NodeInput::MessageReceived { from, message });
        }
    }

    pub(crate) fn tick_device(&mut self, device: usize) {
        let link_up = self.network.link_up(device, &self.world, self.now_s);
        if self.devices[device].reported_link_up != Some(link_up) {
            self.devices[device].reported_link_up = Some(link_up);
            self.deliver_node_input(device, NodeInput::TransportLinkChanged { link_up });
        }
        let motion = if self.world.is_walking(device, self.now_s) { MotionState::Walking } else { MotionState::Still };
        if self.devices[device].reported_motion != Some(motion) {
            self.devices[device].reported_motion = Some(motion);
            self.deliver_node_status(device, DeviceStatusChange::Motion(motion));
        }
        if let Some(ramp) = self.world.devices[device].spec.thermal_ramp {
            let headroom = ramp.headroom_at(self.now_s);
            let reported = self.devices[device].reported_headroom;
            if reported.is_none_or(|last| (last - headroom).abs() >= HEADROOM_REPORT_STEP) {
                self.devices[device].reported_headroom = Some(headroom);
                self.deliver_node_status(device, DeviceStatusChange::ThermalHeadroom(headroom as f32));
                if self.devices[device].host.is_some() {
                    self.deliver_host_input(device, HostInput::OwnThermalHeadroom(headroom as f32));
                }
                if self.devices[device].node.mic_only && !self.metrics.mic_only_since.contains_key(&device) {
                    self.metrics.mic_only_since.insert(device, self.now_s);
                }
            }
        }
        let stalls: Vec<_> = self.world.devices[device].stalls.iter().filter(|s| s.time_s <= self.now_s).copied().collect();
        while self.devices[device].reported_stall_count < stalls.len() {
            let stall = stalls[self.devices[device].reported_stall_count];
            self.devices[device].reported_stall_count += 1;
            self.metrics.stalls_applied.push((stall.time_s, device, stall.reported));
            if stall.reported {
                self.deliver_node_status(device, DeviceStatusChange::CaptureGapReported);
            }
        }
        let fix_due = self.devices[device]
            .last_gnss_fix_s
            .is_none_or(|last| self.now_s - last >= self.world.config.gnss_fix_interval_s);
        if fix_due {
            self.devices[device].last_gnss_fix_s = Some(self.now_s);
            let fix_index = self.devices[device].gnss_fix_count;
            self.devices[device].gnss_fix_count += 1;
            let truth = &self.world.devices[device];
            let position = self.world.position(device, self.now_s);
            let jitter = |axis: u64| GNSS_FIX_JITTER_M * hashed_gaussian(self.seed, "gnss_jitter", device as u64 * 2 + axis, fix_index);
            let east_north = [position[0] + truth.gps_bias_m[0] + jitter(0), position[1] + truth.gps_bias_m[1] + jitter(1)];
            let (latitude_deg, longitude_deg) = self.site_plane.geodetic_from_east_north(east_north);
            let fix = GnssFix {
                latitude_deg,
                longitude_deg,
                altitude_m: position[2] as f32,
                horizontal_accuracy_m: truth.spec.gps_error_sigma_m.max(GNSS_FIX_JITTER_M) as f32,
                fix_device_sample_index: truth.sample_index_at(self.now_s).max(0.0) as u64,
            };
            self.deliver_node_status(device, DeviceStatusChange::GnssFix(fix));
        }
        self.deliver_node_input(device, NodeInput::Tick);
        if self.devices[device].host.is_some() {
            self.deliver_host_input(device, HostInput::Tick);
        }
    }

    pub(crate) fn call_reaches_classifier(&mut self, device: usize, call: usize) {
        if !self.network.alive[device] {
            return;
        }
        let config = &self.world.config;
        let snr = self.world.call_snr_db(call, device);
        if snr < config.stage1_trigger_snr_db {
            return;
        }
        self.metrics.stage1_triggers.entry(call).or_default().push(device);
        let spec = self.world.calls[call].clone();
        let arrival = self.world.arrival_time(spec.time_s, spec.position_m, device);
        let truth = &self.world.devices[device];
        let trigger_sample = truth.sample_index_at(arrival).max(0.0) as u64;
        if !self.devices[device].node.has_classification_duty(trigger_sample) {
            return;
        }
        self.metrics.classifier_runs.entry(call).or_default().push(device);
        if snr < config.classifier_detection_snr_db {
            return;
        }
        let window_start = arrival - config.classification_window_s / 2.0;
        let classification = LocalClassification {
            window_first_device_sample_index: truth.sample_index_at(window_start).max(0.0) as u64,
            window_sample_count: (config.classification_window_s * DEVICE_SAMPLE_RATE_HZ) as u32,
            classifier_model_id: CLASSIFIER_MODEL_ID,
            species_scores: vec![SpeciesScore { label_id: spec.species_label, score: spec.classifier_score }],
            stage1_contrast_db: snr as f32,
            stage1_active_band: Some(FrequencyBandHz {
                low_hz: (spec.centre_frequency_hz * CALL_BAND_LOW_FRACTION) as f32,
                high_hz: (spec.centre_frequency_hz * CALL_BAND_HIGH_FRACTION) as f32,
            }),
        };
        self.metrics.detections_by_call.entry(call).or_default().push(device);
        self.deliver_node_input(device, NodeInput::ClassifierFired(classification));
        let key = (NodeId(device as u32), self.devices[device].node.counters.detections_made);
        self.metrics.generated_detections.push((key, device, self.now_s));
    }

    pub(crate) fn emit_chirp(&mut self, emitter: usize, round_id: u32) {
        if !self.network.alive[emitter] {
            return;
        }
        let emission_s = self.now_s;
        let emission_sample = self.world.devices[emitter].sample_index_at(emission_s).round();
        self.devices[emitter].own_emissions.entry(round_id).or_insert(emission_sample);
        let occurrence = self.devices[emitter]
            .round_arrivals
            .get(&round_id)
            .map_or(0, |arrivals| arrivals.iter().filter(|a| a.emitter_node_id.0 as usize == emitter).count()) as u64;
        let source = self.world.position(emitter, emission_s);
        for listener in 0..self.devices.len() {
            if !self.network.alive[listener] {
                continue;
            }
            let snr = self.world.chirp_snr_db(emitter, listener, emission_s);
            if snr < self.world.config.chirp_detection_minimum_snr_db {
                continue;
            }
            let arrival_s = if listener == emitter {
                emission_s + self.world.devices[emitter].spec.self_path_length_m / self.world.sound_speed_mps
            } else {
                self.world.arrival_time(emission_s, source, listener)
            };
            let std_s = self.world.timing_std_s(snr, self.world.config.chirp_pick_std_at_reference_s);
            let noise = hashed_gaussian(self.seed, "chirp_pick", (round_id as u64) << 20 | (emitter as u64) << 10 | listener as u64, occurrence);
            let sample = self.world.devices[listener].sample_index_at(arrival_s) + noise * std_s * DEVICE_SAMPLE_RATE_HZ;
            let index = sample.round();
            self.devices[listener].round_arrivals.entry(round_id).or_default().push(
                minimappr_core::protocol::session_messages::ChirpArrival {
                    emitter_node_id: NodeId(emitter as u32),
                    arrival_device_sample_index: index.max(0.0) as u64,
                    arrival_fractional_sample: (sample - index) as f32,
                    peak_to_noise_db: snr as f32,
                },
            );
        }
    }

    pub(crate) fn end_listen(&mut self, device: usize, round_id: u32) {
        if !self.devices[device].listening_rounds.remove(&round_id) {
            return;
        }
        let arrivals = self.devices[device].round_arrivals.remove(&round_id).unwrap_or_default();
        let own_emission = self.devices[device].own_emissions.remove(&round_id);
        let heard_self = arrivals.iter().any(|a| a.emitter_node_id.0 as usize == device);
        if self.devices[device].chirp_direction_enabled {
            self.send_direction_reports(device, round_id, &arrivals);
        }
        let round = minimappr_core::mesh::node_state_machine::LocalChirpRound {
            round_id,
            own_emission_device_sample_index: own_emission.map(|sample| sample.max(0.0) as u64),
            arrivals,
            own_chirp_inaudible: own_emission.is_some() && !heard_self,
        };
        self.deliver_node_input(device, NodeInput::ChirpRoundHeard(round));
    }

    fn send_direction_reports(&mut self, device: usize, round_id: u32, arrivals: &[minimappr_core::protocol::session_messages::ChirpArrival]) {
        let Some(host_device) = self.current_host_device() else { return };
        let mut emitters: Vec<usize> = arrivals.iter().map(|a| a.emitter_node_id.0 as usize).filter(|e| *e != device).collect();
        emitters.dedup();
        for emitter in emitters {
            let (from, to) = (self.world.position(device, self.now_s), self.world.position(emitter, self.now_s));
            let true_azimuth = (to[1] - from[1]).atan2(to[0] - from[0]).to_degrees();
            let config = &self.world.config;
            let heading_error = config.chirp_direction_heading_std_deg * hashed_gaussian(self.seed, "chirp_heading", round_id as u64, device as u64);
            let doa_noise = config.chirp_direction_noise_std_deg * hashed_gaussian(self.seed, "chirp_doa", round_id as u64, (device * 64 + emitter) as u64);
            let error = heading_error + doa_noise;
            let report = ChirpDirectionReport {
                round_id,
                reporting_node_id: NodeId(device as u32),
                emitter_node_id: NodeId(emitter as u32),
                world_azimuth_deg: (true_azimuth + error) as f32,
                azimuth_standard_deviation_deg: self.world.config.chirp_direction_heading_std_deg.hypot(self.world.config.chirp_direction_noise_std_deg) as f32,
            };
            self.send_message(
                device,
                OutboundMessage::control(MessageRecipient::Node(NodeId(host_device as u32)), FieldMessage::ChirpDirectionReport(report)),
            );
        }
    }
}
