//! Scenario description: devices, calls, noise, network and a script of
//! timed faults. Builders keep each scenario file short.

use minimappr_core::mesh::config::MeshConfig;
use minimappr_core::mesh::host_transport_policy::{CellLinkQuality, SessionTransportPreference};

use crate::acoustic_world::{NoiseInterval, ScriptedCall, WorldConfig};
use crate::device_spec::DeviceSpec;
use crate::simulated_network::NetworkSpec;

/// Default bird call used by scenarios (a 4 kHz song at 85 dB SPL @ 1 m).
pub const DEFAULT_CALL_LEVEL_DB_SPL: f64 = 85.0;
pub const DEFAULT_CALL_FREQUENCY_HZ: f64 = 4_000.0;
pub const DEFAULT_CLASSIFIER_SCORE: f32 = 0.85;
pub const DEFAULT_SPECIES_LABEL: u32 = 4_211;

#[derive(Debug, Clone, PartialEq)]
pub enum ScriptAction {
    KillDevice(usize),
    /// Kill whoever hosts, as soon as a chirp round is in flight.
    KillHostDuringRound,
    /// Kill whoever hosts, as soon as a call is pending (readiness/clip fetch).
    KillHostDuringClipFetch,
    SetLossProbability(Option<f64>),
    SetCellQuality { device: usize, quality: CellLinkQuality },
    DropFromRelay { device: usize, until_s: f64 },
    Disconnect { device: usize, until_s: f64 },
    ExtraBulkDelay { device: usize, extra_s: f64 },
    InputRouteChange(usize),
    OutputRouteChange { device: usize, chirp_speaker_pinned: bool },
    EnableChirpDirection(usize),
    /// The relay refuses further messages for this session (cap reached).
    ReachRelayMessageCap,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScriptedEvent {
    pub at_s: f64,
    pub action: ScriptAction,
}

#[derive(Debug, Clone)]
pub struct ScenarioSpec {
    pub name: &'static str,
    pub duration_s: f64,
    pub tick_interval_ms: u64,
    pub devices: Vec<DeviceSpec>,
    pub initial_host_device: usize,
    pub transport_preference: SessionTransportPreference,
    pub initial_cell: CellLinkQuality,
    pub calls: Vec<ScriptedCall>,
    pub noise_intervals: Vec<NoiseInterval>,
    pub script: Vec<ScriptedEvent>,
    pub network: NetworkSpec,
    pub mesh: MeshConfig,
    pub world: WorldConfig,
    pub site_origin_deg: (f64, f64),
}

impl ScenarioSpec {
    pub fn new(name: &'static str, duration_s: f64, devices: Vec<DeviceSpec>) -> Self {
        let mut mesh = MeshConfig::default();
        for device in &devices {
            mesh.island_sync
                .self_path_length_by_platform_m
                .insert(device.platform_name(), device.self_path_length_m);
        }
        Self {
            name,
            duration_s,
            tick_interval_ms: 100,
            devices,
            initial_host_device: 0,
            transport_preference: SessionTransportPreference::Auto,
            initial_cell: CellLinkQuality::Unconstrained,
            calls: Vec::new(),
            noise_intervals: Vec::new(),
            script: Vec::new(),
            network: NetworkSpec::default(),
            mesh,
            world: WorldConfig::default(),
            site_origin_deg: (47.6, -122.3),
        }
    }

    pub fn call(mut self, time_s: f64, position_m: [f64; 3]) -> Self {
        self.calls.push(ScriptedCall {
            time_s,
            position_m,
            level_db_spl_at_1m: DEFAULT_CALL_LEVEL_DB_SPL,
            species_label: DEFAULT_SPECIES_LABEL,
            classifier_score: DEFAULT_CLASSIFIER_SCORE,
            centre_frequency_hz: DEFAULT_CALL_FREQUENCY_HZ,
        });
        self
    }

    pub fn call_with_level(mut self, time_s: f64, position_m: [f64; 3], level_db_spl_at_1m: f64) -> Self {
        self = self.call(time_s, position_m);
        self.calls.last_mut().expect("just pushed").level_db_spl_at_1m = level_db_spl_at_1m;
        self
    }

    pub fn at(mut self, at_s: f64, action: ScriptAction) -> Self {
        self.script.push(ScriptedEvent { at_s, action });
        self
    }
}

/// Phones on a jittered ring of `radius_m` around `centre`.
pub fn ring_of_phones(count: usize, centre: [f64; 2], radius_m: f64, seed: u64) -> Vec<DeviceSpec> {
    use crate::seeded_random::hashed_uniform;
    const RADIUS_JITTER_FRACTION: f64 = 0.25;
    const ANGLE_JITTER_FRACTION: f64 = 0.3;
    const PHONE_HEIGHT_M: f64 = 1.4;
    const PHONE_HEIGHT_JITTER_M: f64 = 0.2;
    (0..count)
        .map(|index| {
            let jitter = |purpose: &str| 2.0 * hashed_uniform(seed, purpose, index as u64, count as u64) - 1.0;
            let angle = std::f64::consts::TAU * (index as f64 + ANGLE_JITTER_FRACTION * jitter("ring_angle")) / count as f64;
            let radius = radius_m * (1.0 + RADIUS_JITTER_FRACTION * jitter("ring_radius"));
            DeviceSpec::android_at([
                centre[0] + radius * angle.cos(),
                centre[1] + radius * angle.sin(),
                PHONE_HEIGHT_M + PHONE_HEIGHT_JITTER_M * jitter("ring_height"),
            ])
        })
        .collect()
}
