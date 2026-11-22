//! Receiving/emitting devices: Sirith tetrahedral array nodes and phones.

use serde::{Deserialize, Serialize};

use crate::physics::geometry::Vec3;

/// Mirrors `SIRITH_MIC_POSITIONS_M` in
/// `minimappr-ingest-sidecar/src/dsp_worker.rs` (firmware board frame).
/// Duplicated because this crate must not depend on the sidecar/core yet;
/// switch to `minimappr_core::SIRITH_MIC_POSITIONS_M` once both join the
/// workspace. Bundles store these re-centred on their centroid, matching
/// `SIRITH_TETRA_SENSOR_OFFSETS_M` in `tests/helpers.py`.
pub const SIRITH_MIC_POSITIONS_M: [[f64; 3]; 4] = [
    [0.0, 0.050, 0.0],
    [0.0433, 0.025, 0.0],
    [0.0, 0.0, 0.0],
    [0.02165, 0.025, 0.04082],
];

/// Sirith offsets relative to the array centroid (node `position_m`).
pub fn sirith_offsets_about_centroid_m() -> Vec<Vec3> {
    let count = SIRITH_MIC_POSITIONS_M.len() as f64;
    let mut centroid = [0.0; 3];
    for mic in SIRITH_MIC_POSITIONS_M {
        for axis in 0..3 {
            centroid[axis] += mic[axis] / count;
        }
    }
    SIRITH_MIC_POSITIONS_M
        .iter()
        .map(|mic| {
            [
                mic[0] - centroid[0],
                mic[1] - centroid[1],
                mic[2] - centroid[2],
            ]
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    /// 4-mic PPS/GPS-disciplined array node (clock offset/drift ≈ 0).
    SirithTetra,
    /// Single-mic (optionally 2-mic) phone with its own speaker and a free-running clock.
    Phone,
}

/// Sample clock of a device. Receiver i takes sample n at true time
/// `offset_s + n / (fs * (1 + drift_ppm * 1e-6))`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum ClockConfig {
    /// GPS/PPS-locked: zero offset, zero drift.
    Disciplined,
    /// Offset uniform in ±max_offset_s, drift uniform in ±max_drift_ppm.
    Random {
        max_offset_s: f64,
        max_drift_ppm: f64,
    },
    Fixed {
        offset_s: f64,
        drift_ppm: f64,
    },
}

/// Default phone clock: NTP-grade start offset, crystal drift up to ±60 ppm.
pub const DEFAULT_PHONE_MAX_CLOCK_OFFSET_S: f64 = 0.02;
pub const DEFAULT_PHONE_MAX_CLOCK_DRIFT_PPM: f64 = 60.0;

/// Microphone + ADC chain response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MicResponseConfig {
    /// Low-cut corner (wind/handling HPF), Hz; None disables.
    pub highpass_hz: Option<f64>,
    pub highpass_order: usize,
    /// High-frequency roll-off corner, Hz; None disables. Weak phones: ~16 kHz, order 8.
    pub lowpass_hz: Option<f64>,
    pub lowpass_order: usize,
    /// Equivalent input self-noise, dB SPL (broadband, white).
    pub self_noise_db_spl: f64,
    /// SPL that maps to digital full scale (sets sensitivity).
    pub full_scale_db_spl: f64,
}

impl MicResponseConfig {
    pub fn default_for(kind: DeviceKind) -> Self {
        match kind {
            DeviceKind::SirithTetra => Self {
                highpass_hz: Some(40.0),
                highpass_order: 2,
                lowpass_hz: None,
                lowpass_order: 2,
                self_noise_db_spl: 30.0,
                full_scale_db_spl: 120.0,
            },
            DeviceKind::Phone => Self {
                highpass_hz: Some(80.0),
                highpass_order: 2,
                lowpass_hz: Some(20_000.0),
                lowpass_order: 4,
                self_noise_db_spl: 32.0,
                full_scale_db_spl: 120.0,
            },
        }
    }
}

impl Default for MicResponseConfig {
    fn default() -> Self {
        Self::default_for(DeviceKind::Phone)
    }
}

/// Optional second capsule on stereo-capable phones.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SecondMicConfig {
    /// Offset from the primary mic (ENU at the device's fixed yaw).
    pub offset_m: Vec3,
    /// Phone-body acoustic shadow on this capsule: high shelf above corner.
    pub body_shadow_corner_hz: f64,
    pub body_shadow_gain_db: f64,
}

impl Default for SecondMicConfig {
    fn default() -> Self {
        Self {
            offset_m: [0.0, 0.15, 0.0],
            body_shadow_corner_hz: 4000.0,
            body_shadow_gain_db: -3.0,
        }
    }
}

/// How the phone is held; sets the structural precursor level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Grip {
    Foam,
    #[default]
    Hand,
    HardTable,
}

/// Body-conduction precursor level relative to the airborne self path, per grip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BodyConductionLevels {
    pub foam_relative_db: f64,
    pub hand_relative_db: f64,
    pub hard_table_relative_db: f64,
}

impl Default for BodyConductionLevels {
    fn default() -> Self {
        Self {
            foam_relative_db: -40.0,
            hand_relative_db: -26.0,
            hard_table_relative_db: -14.0,
        }
    }
}

impl BodyConductionLevels {
    pub fn relative_db(&self, grip: Grip) -> f64 {
        match grip {
            Grip::Foam => self.foam_relative_db,
            Grip::Hand => self.hand_relative_db,
            Grip::HardTable => self.hard_table_relative_db,
        }
    }
}

/// Soft-knee tanh limiter applied to the whole mic stream (models the
/// phone's input limiter / ADC saturation that the self-chirp drives into).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LimiterConfig {
    pub enabled: bool,
    /// Linear below this |x| (full-scale units).
    pub knee_full_scale: f64,
    /// Asymptotic output ceiling (full-scale units).
    pub ceiling_full_scale: f64,
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            knee_full_scale: 0.5,
            ceiling_full_scale: 0.99,
        }
    }
}

/// Phone speaker geometry (self-recording) and grip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeakerConfig {
    /// d_ii: speaker → primary mic distance. None draws from the range.
    pub speaker_to_mic_distance_m: Option<f64>,
    pub random_distance_range_m: [f64; 2],
    pub grip: Grip,
    pub body_conduction: BodyConductionLevels,
}

impl Default for SpeakerConfig {
    fn default() -> Self {
        Self {
            speaker_to_mic_distance_m: None,
            random_distance_range_m: [0.01, 0.15],
            grip: Grip::Hand,
            body_conduction: BodyConductionLevels::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceConfig {
    pub device_id: String,
    pub kind: DeviceKind,
    /// Static position (or start position when `trajectory` is absent).
    pub position_m: Vec3,
    /// Optional `[t_s, x, y, z]` rows; overrides `position_m`.
    #[serde(default)]
    pub trajectory: Option<Vec<[f64; 4]>>,
    /// Defaults: tetra → disciplined; phone → random ±20 ms / ±60 ppm.
    #[serde(default)]
    pub clock: Option<ClockConfig>,
    #[serde(default)]
    pub mic_response: Option<MicResponseConfig>,
    /// Phones only: enables a 2nd capsule (2-channel stream).
    #[serde(default)]
    pub second_mic: Option<SecondMicConfig>,
    /// Phones only.
    #[serde(default)]
    pub speaker: SpeakerConfig,
    /// Defaults: enabled on phones, disabled on tetra nodes (hard clip only).
    #[serde(default)]
    pub limiter: Option<LimiterConfig>,
}

impl DeviceConfig {
    pub fn resolved_clock(&self) -> ClockConfig {
        self.clock.unwrap_or(match self.kind {
            DeviceKind::SirithTetra => ClockConfig::Disciplined,
            DeviceKind::Phone => ClockConfig::Random {
                max_offset_s: DEFAULT_PHONE_MAX_CLOCK_OFFSET_S,
                max_drift_ppm: DEFAULT_PHONE_MAX_CLOCK_DRIFT_PPM,
            },
        })
    }

    pub fn resolved_mic_response(&self) -> MicResponseConfig {
        self.mic_response
            .clone()
            .unwrap_or_else(|| MicResponseConfig::default_for(self.kind))
    }

    pub fn resolved_limiter(&self) -> LimiterConfig {
        self.limiter.clone().unwrap_or_else(|| LimiterConfig {
            enabled: self.kind == DeviceKind::Phone,
            ..LimiterConfig::default()
        })
    }
}
