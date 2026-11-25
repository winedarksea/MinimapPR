//! A two-phone acoustic and network world for `FieldMeshDevice` tests.
//!
//! True time is shared; each phone has its own monotonic-clock epoch (local
//! = true + offset, so the two clocks share no origin, as on real phones)
//! and its own audio crystal (capture rate 48 kHz · (1 + ppm)). Chirps the
//! devices ask to play are rendered into both capture streams at the right
//! fractional sample: the self path d_ii for the emitter, the separation
//! for the other phone. The network delivers envelopes with a base delay
//! and occasional asymmetric power-save spikes.

use std::collections::VecDeque;

use minimappr_core::sync::chirp_waveform::{generate_up_down_chirp, ChirpBand, SweepLaw};
use minimappr_ffi::mesh_device::{FieldDeviceAction, FieldMeshDevice};

pub const NOMINAL_RATE_HZ: f64 = 48_000.0;
pub const OVERSAMPLING: f64 = 16.0;
const NS_PER_S: f64 = 1e9;

pub struct PhoneWorld {
    pub device: FieldMeshDevice,
    pub monotonic_offset_ns: i64,
    pub capture_start_true_s: f64,
    pub audio_rate_ppm: f64,
    pub next_sample_index: u64,
    /// Emission ids this phone must report as not played.
    pub skip_emission_ids: Vec<u64>,
    pub skipped_emissions: u64,
}

pub struct Emission {
    pub emitter: usize,
    pub true_start_s: f64,
}

pub struct PairWorld {
    pub phones: Vec<PhoneWorld>,
    pub distance_m: f64,
    pub self_path_m: f64,
    pub sound_speed_mps: f64,
    pub true_now_ns: i64,
    pub emissions: Vec<Emission>,
    pub network: VecDeque<(i64, usize, Vec<u8>)>,
    pub playback_error_ns: i64,
    fine_template: Vec<f32>,
    noise_state: u64,
    message_count: u64,
}

impl PhoneWorld {
    pub fn local_ns(&self, true_ns: i64) -> i64 {
        true_ns + self.monotonic_offset_ns
    }

    pub fn capture_rate_hz(&self) -> f64 {
        NOMINAL_RATE_HZ * (1.0 + self.audio_rate_ppm * 1e-6)
    }

    /// Audio-clock seconds (sample index / nominal rate) at a true time.
    pub fn audio_seconds_at(&self, true_s: f64) -> f64 {
        (true_s - self.capture_start_true_s) * self.capture_rate_hz() / NOMINAL_RATE_HZ
    }

    pub fn true_s_at_audio_seconds(&self, audio_s: f64) -> f64 {
        self.capture_start_true_s + audio_s * NOMINAL_RATE_HZ / self.capture_rate_hz()
    }
}

impl PairWorld {
    pub fn new(phones: Vec<PhoneWorld>, distance_m: f64, self_path_m: f64, sound_speed_mps: f64, band: ChirpBand, duration_s: f64) -> Self {
        let fine_template = generate_up_down_chirp(band, duration_s, SweepLaw::HyperbolicFm, NOMINAL_RATE_HZ * OVERSAMPLING).samples;
        Self {
            phones,
            distance_m,
            self_path_m,
            sound_speed_mps,
            true_now_ns: 0,
            emissions: Vec::new(),
            network: VecDeque::new(),
            playback_error_ns: 1_500_000,
            fine_template,
            noise_state: 0x2545_F491_4F6C_DD1D,
            message_count: 0,
        }
    }

    fn noise(&mut self) -> f32 {
        self.noise_state ^= self.noise_state << 13;
        self.noise_state ^= self.noise_state >> 7;
        self.noise_state ^= self.noise_state << 17;
        ((self.noise_state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.002
    }

    /// One-way delay: 3 ms, with an asymmetric 120 ms spike every 7th message.
    fn network_delay_ns(&mut self) -> i64 {
        self.message_count += 1;
        let spike = if self.message_count % 7 == 0 { 120_000_000 } else { 0 };
        3_000_000 + spike + (self.message_count % 3) as i64 * 400_000
    }

    pub fn handle_actions(&mut self, phone: usize, actions: Vec<FieldDeviceAction>) {
        for action in actions {
            match action {
                FieldDeviceAction::SendEnvelope { envelope, .. } => {
                    let delay = self.network_delay_ns();
                    self.network.push_back((self.true_now_ns + delay, 1 - phone, envelope));
                }
                FieldDeviceAction::EmitChirp { emission_id, emit_at_local_ns, .. } => {
                    let world = &mut self.phones[phone];
                    if world.skip_emission_ids.contains(&emission_id) {
                        world.skipped_emissions += 1;
                        world.device.report_chirp_emission_outcome(emission_id, false, emit_at_local_ns);
                        continue;
                    }
                    world.device.report_chirp_emission_outcome(emission_id, true, emit_at_local_ns);
                    let true_start_ns = emit_at_local_ns - world.monotonic_offset_ns + self.playback_error_ns;
                    self.emissions.push(Emission { emitter: phone, true_start_s: true_start_ns as f64 / NS_PER_S });
                }
                FieldDeviceAction::UserNotice { .. } | FieldDeviceAction::HostRoleChanged { .. } => {}
            }
        }
    }

    fn render_sample(&self, phone: usize, true_s: f64) -> f32 {
        let template_s = self.fine_template.len() as f64 / (NOMINAL_RATE_HZ * OVERSAMPLING);
        let mut value = 0.0f32;
        for emission in &self.emissions {
            let (path, amplitude) = if emission.emitter == phone { (self.self_path_m, 0.4) } else { (self.distance_m, 0.08) };
            let offset_s = true_s - emission.true_start_s - path / self.sound_speed_mps;
            if offset_s < 0.0 || offset_s >= template_s {
                continue;
            }
            let fine_index = (offset_s * NOMINAL_RATE_HZ * OVERSAMPLING).round() as usize;
            value += amplitude * self.fine_template[fine_index.min(self.fine_template.len() - 1)];
        }
        value
    }

    /// Capture audio up to the current true time and report anchors.
    pub fn capture(&mut self) {
        let true_now_s = self.true_now_ns as f64 / NS_PER_S;
        for phone in 0..self.phones.len() {
            let (capture_start_s, rate_hz, first) = {
                let world = &self.phones[phone];
                (world.capture_start_true_s, world.capture_rate_hz(), world.next_sample_index)
            };
            if true_now_s < capture_start_s {
                continue;
            }
            let end_index = ((true_now_s - capture_start_s) * rate_hz) as u64;
            if end_index <= first {
                continue;
            }
            let mut block = Vec::with_capacity((end_index - first) as usize);
            for index in first..end_index {
                let true_s = capture_start_s + index as f64 / rate_hz;
                let sample = self.render_sample(phone, true_s) + self.noise();
                block.push((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16);
            }
            let bytes: Vec<u8> = block.iter().flat_map(|s| s.to_le_bytes()).collect();
            let world = &mut self.phones[phone];
            world.device.push_capture_block(first, bytes).expect("capture block");
            world.next_sample_index = end_index;
            let anchor_true_s = world.capture_start_true_s + end_index as f64 / world.capture_rate_hz();
            let anchor_local_ns = (anchor_true_s * NS_PER_S) as i64 + world.monotonic_offset_ns;
            world.device.report_capture_timestamp(end_index, anchor_local_ns);
        }
    }

    pub fn deliver_due(&mut self) {
        let mut remaining = VecDeque::new();
        let due: Vec<(i64, usize, Vec<u8>)> = self.network.drain(..).collect();
        for (deliver_at, to, envelope) in due {
            if deliver_at > self.true_now_ns {
                remaining.push_back((deliver_at, to, envelope));
                continue;
            }
            let local = self.phones[to].local_ns(self.true_now_ns);
            let actions = self.phones[to].device.receive_envelope(local, envelope).expect("envelope");
            self.handle_actions(to, actions);
        }
        self.network.extend(remaining);
    }

    pub fn tick_all(&mut self) {
        for phone in 0..self.phones.len() {
            let local = self.phones[phone].local_ns(self.true_now_ns);
            let actions = self.phones[phone].device.tick(local);
            self.handle_actions(phone, actions);
        }
    }

    /// Advance true time in `step_ns` steps up to `until_ns`.
    pub fn run_until(&mut self, until_ns: i64, step_ns: i64) {
        while self.true_now_ns < until_ns {
            self.true_now_ns += step_ns;
            self.capture();
            self.deliver_due();
            self.tick_all();
        }
    }

    /// True θ = local_node − local_reference (audio seconds) at the
    /// reference node's audio time `reference_audio_s`.
    pub fn true_offset_s(&self, node_phone: usize, reference_phone: usize, reference_audio_s: f64) -> f64 {
        let true_s = self.phones[reference_phone].true_s_at_audio_seconds(reference_audio_s);
        self.phones[node_phone].audio_seconds_at(true_s) - reference_audio_s
    }
}
