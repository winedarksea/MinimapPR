//! Chaos knobs (test strategy, Layer 1 §4). Every knob is `None` by default
//! (disabled). Device selectors: an empty `device_ids` list means "every
//! phone" unless the knob says otherwise. All random draws use a stream
//! keyed by the knob's name, so toggling one knob never perturbs another.

use serde::{Deserialize, Serialize};

use crate::physics::geometry::Vec3;
use crate::signals::calls::CallGenerator;
use crate::signals::distractors::{ClapParams, SpeechBabbleParams};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChaosConfig {
    pub sample_drops: Option<SampleDropsKnob>,
    pub walking: Option<WalkingKnob>,
    pub pocket: Option<PocketKnob>,
    pub canopy_gps: Option<CanopyGpsKnob>,
    pub route_change: Option<RouteChangeKnob>,
    pub thermal_stall: Option<ThermalStallKnob>,
    pub micro_motion: Option<MicroMotionKnob>,
    pub muted_speaker: Option<MutedSpeakerKnob>,
    pub speech: Option<SpeechKnob>,
    pub clap: Option<ClapKnob>,
    pub intermittent_flock: Option<IntermittentFlockKnob>,
    pub same_band_pair: Option<SameBandPairKnob>,
    pub two_speaker_emission: Option<TwoSpeakerEmissionKnob>,
}

/// Delete `drops_per_device` runs of 256–4096 samples from each selected
/// stream. A *reported* gap is zero-filled (timing preserved, like a counted
/// overrun); a *silent* gap is removed, shifting every later sample earlier.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SampleDropsKnob {
    pub device_ids: Vec<String>,
    pub drops_per_device: usize,
    pub min_length_samples: usize,
    pub max_length_samples: usize,
    /// Probability a drop is reported (zero-filled) rather than silent.
    pub reported_probability: f64,
}

impl Default for SampleDropsKnob {
    fn default() -> Self {
        Self {
            device_ids: vec![],
            drops_per_device: 2,
            min_length_samples: 256,
            max_length_samples: 4096,
            reported_probability: 0.5,
        }
    }
}

/// Piecewise walking: walk at `speed_mps` in a random heading for a random
/// leg duration, then stand still for `stop_duration_s`, repeated from
/// `start_s`. The device's true trajectory is replaced; ground truth records it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WalkingKnob {
    pub device_ids: Vec<String>,
    pub start_s: f64,
    pub speed_mps: f64,
    pub leg_duration_range_s: [f64; 2],
    pub stop_duration_s: f64,
}

impl Default for WalkingKnob {
    fn default() -> Self {
        Self {
            device_ids: vec![],
            start_s: 0.0,
            speed_mps: 1.2,
            leg_duration_range_s: [4.0, 10.0],
            stop_duration_s: 6.0,
        }
    }
}

/// Phone goes into a pocket mid-session: everything above `corner_hz` is
/// cut by `attenuation_db` (high shelf) from `start_s` to `end_s`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PocketKnob {
    /// Exactly one device id expected; empty = first phone.
    pub device_ids: Vec<String>,
    pub start_s: f64,
    pub end_s: Option<f64>,
    pub corner_hz: f64,
    pub attenuation_db: f64,
}

impl Default for PocketKnob {
    fn default() -> Self {
        Self {
            device_ids: vec![],
            start_s: 10.0,
            end_s: None,
            corner_hz: 1500.0,
            attenuation_db: 25.0,
        }
    }
}

/// Canopy birds + bad GPS. Knob-generated birds (flock, same-band pair) sit
/// at heights drawn from `bird_height_range_m`; every device's *reported*
/// position (manifest) carries an Ornstein–Uhlenbeck error with stationary
/// σ = `gps_error_sigma_m` per horizontal axis and the given correlation time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CanopyGpsKnob {
    pub bird_height_range_m: [f64; 2],
    pub gps_error_sigma_m: f64,
    pub gps_error_correlation_time_s: f64,
    /// Sampling step of the recorded GPS error track.
    pub gps_track_step_s: f64,
}

impl Default for CanopyGpsKnob {
    fn default() -> Self {
        Self {
            bird_height_range_m: [5.0, 30.0],
            gps_error_sigma_m: 6.0,
            gps_error_correlation_time_s: 60.0,
            gps_track_step_s: 1.0,
        }
    }
}

/// Audio route switches to a Bluetooth headset: from `at_s` the stream is
/// band-limited to `band_limited_sample_rate_hz / 2`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RouteChangeKnob {
    pub device_ids: Vec<String>,
    pub at_s: f64,
    pub band_limited_sample_rate_hz: f64,
    pub filter_order: usize,
}

impl Default for RouteChangeKnob {
    fn default() -> Self {
        Self {
            device_ids: vec![],
            at_s: 10.0,
            band_limited_sample_rate_hz: 16_000.0,
            filter_order: 8,
        }
    }
}

/// Thermal headroom ramps linearly 0 → 1 over [ramp_start_s, ramp_end_s];
/// once past `stall_threshold` the capture thread stalls for 50–300 ms at
/// exponential intervals (mean `mean_stall_interval_s`). Stalls are reported
/// (zero-filled) or silent (samples removed) like sample drops.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ThermalStallKnob {
    pub device_ids: Vec<String>,
    pub ramp_start_s: f64,
    pub ramp_end_s: f64,
    pub stall_threshold: f64,
    pub stall_duration_range_s: [f64; 2],
    pub mean_stall_interval_s: f64,
    pub reported_probability: f64,
}

impl Default for ThermalStallKnob {
    fn default() -> Self {
        Self {
            device_ids: vec![],
            ramp_start_s: 0.0,
            ramp_end_s: 20.0,
            stall_threshold: 0.95,
            stall_duration_range_s: [0.05, 0.3],
            mean_stall_interval_s: 4.0,
            reported_probability: 0.5,
        }
    }
}

/// 10–30 cm hand/body shifts without steps during [start_s, end_s].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MicroMotionKnob {
    pub device_ids: Vec<String>,
    pub start_s: f64,
    pub end_s: f64,
    pub shift_count: usize,
    pub displacement_range_m: [f64; 2],
    /// Time each shift takes.
    pub shift_duration_s: f64,
}

impl Default for MicroMotionKnob {
    fn default() -> Self {
        Self {
            device_ids: vec![],
            start_s: 0.0,
            end_s: 10.0,
            shift_count: 4,
            displacement_range_m: [0.1, 0.3],
            shift_duration_s: 0.3,
        }
    }
}

/// Selected phones' chirps play at zero level (self-recording shows nothing).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MutedSpeakerKnob {
    /// Empty = first phone.
    pub device_ids: Vec<String>,
}

/// Human conversation near one or two phones, overlapping bird calls.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeechKnob {
    /// Speakers stand next to these devices (empty = first phone).
    pub near_device_ids: Vec<String>,
    pub distance_from_device_m: f64,
    pub start_s: f64,
    pub level_db_spl_at_1m: f64,
    pub babble: SpeechBabbleParams,
    /// Optional real recording replacing the synthetic babble.
    pub wav_path: Option<String>,
}

impl Default for SpeechKnob {
    fn default() -> Self {
        Self {
            near_device_ids: vec![],
            distance_from_device_m: 1.5,
            start_s: 2.0,
            level_db_spl_at_1m: 60.0,
            babble: SpeechBabbleParams::default(),
            wav_path: None,
        }
    }
}

/// Broadband clap(s) from a point inside the group (default: device centroid).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClapKnob {
    pub times_s: Vec<f64>,
    pub position_m: Option<Vec3>,
    pub level_db_spl_at_1m: f64,
    pub clap: ClapParams,
}

impl Default for ClapKnob {
    fn default() -> Self {
        Self {
            times_s: vec![3.0],
            position_m: None,
            level_db_spl_at_1m: 100.0,
            clap: ClapParams::default(),
        }
    }
}

/// N birds of one species at fixed perches singing 40–250 ms notes in bouts
/// separated by 4–12 s gaps, with occasional 10–30 m hops between bouts.
/// One ground-truth event per bout, linked by `individual_id`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IntermittentFlockKnob {
    pub species: String,
    pub bird_count: usize,
    /// Perches drawn uniformly in a disc of this radius around `center_m`.
    pub perch_radius_m: f64,
    pub center_m: Vec3,
    pub note_duration_range_s: [f64; 2],
    pub notes_per_bout_range: [usize; 2],
    pub note_gap_range_s: [f64; 2],
    pub bout_gap_range_s: [f64; 2],
    pub hop_probability: f64,
    pub hop_distance_range_m: [f64; 2],
    pub level_db_spl_at_1m: f64,
    /// Note centre frequency range (each note a short tonal whistle).
    pub note_frequency_range_hz: [f64; 2],
}

impl Default for IntermittentFlockKnob {
    fn default() -> Self {
        Self {
            species: "Black-capped Chickadee".into(),
            bird_count: 4,
            perch_radius_m: 30.0,
            center_m: [0.0, 0.0, 0.0],
            note_duration_range_s: [0.04, 0.25],
            notes_per_bout_range: [2, 6],
            note_gap_range_s: [0.05, 0.2],
            bout_gap_range_s: [4.0, 12.0],
            hop_probability: 0.15,
            hop_distance_range_m: [10.0, 30.0],
            level_db_spl_at_1m: 80.0,
            note_frequency_range_hz: [3000.0, 4200.0],
        }
    }
}

/// Two species with overlapping bands (e.g. robin / tanager, 2–5 kHz) at
/// different spots and levels, overlapping inside one classification window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SameBandPairKnob {
    pub species_a: String,
    pub species_b: String,
    pub position_a_m: Vec3,
    pub position_b_m: Vec3,
    pub level_a_db_spl_at_1m: f64,
    pub level_b_db_spl_at_1m: f64,
    pub start_s: f64,
    /// Species B starts this much after A (inside one window).
    pub b_offset_s: f64,
    pub generator_a: CallGenerator,
    pub generator_b: CallGenerator,
}

impl Default for SameBandPairKnob {
    fn default() -> Self {
        use crate::signals::calls::{TonalWhistleParams, TrillParams};
        Self {
            species_a: "American Robin".into(),
            species_b: "Scarlet Tanager".into(),
            position_a_m: [15.0, 10.0, 8.0],
            position_b_m: [-12.0, 18.0, 12.0],
            level_a_db_spl_at_1m: 90.0,
            level_b_db_spl_at_1m: 82.0,
            start_s: 4.0,
            b_offset_s: 0.6,
            generator_a: CallGenerator::TonalWhistle(TonalWhistleParams {
                start_frequency_hz: 2600.0,
                end_frequency_hz: 3300.0,
                note_duration_s: 0.3,
                note_count: 4,
                note_gap_s: 0.12,
                frequency_step_ratio: 1.05,
                vibrato_depth_hz: 120.0,
                vibrato_rate_hz: 25.0,
                ..TonalWhistleParams::default()
            }),
            generator_b: CallGenerator::Trill(TrillParams {
                carrier_frequency_hz: 3200.0,
                element_sweep_hz: 1400.0,
                element_rate_hz: 6.0,
                duration_s: 1.5,
                ..TrillParams::default()
            }),
        }
    }
}

/// A phone's chirp plays from both the bottom speaker and the earpiece,
/// `earpiece_offset_m` apart, so double arrivals can be flagged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TwoSpeakerEmissionKnob {
    /// Empty = first phone.
    pub device_ids: Vec<String>,
    pub earpiece_offset_m: Vec3,
    pub earpiece_relative_db: f64,
}

impl Default for TwoSpeakerEmissionKnob {
    fn default() -> Self {
        Self {
            device_ids: vec![],
            earpiece_offset_m: [0.0, 0.15, 0.0],
            earpiece_relative_db: -6.0,
        }
    }
}
