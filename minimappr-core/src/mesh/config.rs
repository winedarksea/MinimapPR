//! Every tunable of the mesh state machines, grouped by concern. Defaults
//! follow the plan's numbers; the Layer 2 simulator overrides per scenario.

use std::collections::BTreeMap;

use crate::layout::filtered_map_anchor::FilteredMapAnchorConfig;
use crate::layout::gps_anchor_fit::AnchorFitConfig;
use crate::layout::robust_layout_refine::LayoutRefineConfig;
use crate::sync::clock_filter::ClockFilterConfig;
use crate::sync::chirp_waveform::{ChirpWaveformKind, StereoOutputChannel, DEFAULT_CHIRP_DURATION_S};
use crate::sync::clock_graph::ClockGraphConfig;

use super::MeshMilliseconds;

pub const DEVICE_SAMPLE_RATE_HZ: f64 = 48_000.0;

/// Liveness, solo fallback and handover timers.
#[derive(Debug, Clone, PartialEq)]
pub struct LivenessConfig {
    pub node_heartbeat_interval_ms: MeshMilliseconds,
    pub host_announcement_interval_ms: MeshMilliseconds,
    /// Host silent this long on a live link: the successor considers it gone.
    pub host_silence_takeover_timeout_ms: MeshMilliseconds,
    /// Extra wait per succession rank, so rank 0 normally wins alone.
    pub takeover_stagger_per_rank_ms: MeshMilliseconds,
    /// Plan: a phone that loses the host for 15–30 s drops to solo.
    pub solo_fallback_timeout_ms: MeshMilliseconds,
    /// Host stops querying a node it has not heard from for this long.
    pub node_presence_timeout_ms: MeshMilliseconds,
}

impl Default for LivenessConfig {
    fn default() -> Self {
        Self {
            node_heartbeat_interval_ms: 5_000,
            host_announcement_interval_ms: 2_000,
            host_silence_takeover_timeout_ms: 6_000,
            takeover_stagger_per_rank_ms: 3_000,
            solo_fallback_timeout_ms: 20_000,
            node_presence_timeout_ms: 15_000,
        }
    }
}

/// Event log replication and the detection outbox.
#[derive(Debug, Clone, PartialEq)]
pub struct EventLogConfig {
    pub catch_up_batch_entries: usize,
    /// Resend an unechoed detection after this long (idempotent upload).
    pub outbox_resend_after_ms: MeshMilliseconds,
    /// Oldest queued detection is dropped (and counted) beyond this.
    pub outbox_capacity: usize,
    pub upload_batch_detections: usize,
    /// A node re-requests catch-up at most this often while behind.
    pub catch_up_retry_interval_ms: MeshMilliseconds,
}

impl Default for EventLogConfig {
    fn default() -> Self {
        Self {
            catch_up_batch_entries: 256,
            outbox_resend_after_ms: 4_000,
            outbox_capacity: 4_096,
            upload_batch_detections: 64,
            catch_up_retry_interval_ms: 1_000,
        }
    }
}

/// Event matcher: which detections start a localization.
#[derive(Debug, Clone, PartialEq)]
pub struct MatcherConfig {
    /// Plan's "localize-to-track" threshold on the top species score.
    pub localize_to_track_score: f32,
    /// Pending calls beyond this drop the oldest (counted, never silent).
    pub pending_call_capacity: usize,
    /// Completed call windows remembered so a covered detection merges.
    pub remembered_window_count: usize,
}

impl Default for MatcherConfig {
    fn default() -> Self {
        Self {
            localize_to_track_score: 0.5,
            pending_call_capacity: 8,
            remembered_window_count: 32,
        }
    }
}

/// Readiness query/answer and clip fetch.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipFetchConfig {
    /// Most clips fetched per call, classifying node included.
    pub readiness_clip_cap: usize,
    pub readiness_answer_timeout_ms: MeshMilliseconds,
    /// Ranking bonus per metre of separation from already chosen nodes.
    pub spread_bonus_db_per_m: f64,
    pub spread_bonus_cap_db: f64,
    /// Shared window half-length around the trigger (plan: 1.0–1.5 s window).
    pub clip_half_window_s: f64,
    /// Margin added on top of the island aperture's largest delay.
    pub inter_node_delay_margin_s: f64,
    pub clip_abandon_after_ms: MeshMilliseconds,
    /// Plan: three nodes with clips for a 2D fix.
    pub minimum_clips_for_fix: usize,
    /// `clip_codec::DecimationFilterId` wire byte (1 = bird band).
    pub decimation_filter_id: u8,
    /// Per-island token bucket: burst size and refill (clips per second).
    pub island_clip_budget_burst: f64,
    pub island_clip_budget_refill_per_s: f64,
    /// Band asked about in readiness queries when a detection has none.
    pub default_query_band_hz: [f32; 2],
    /// RMS linear acceleration in a clip's sensor slice above which the
    /// phone counts as walking at call time.
    pub walking_linear_acceleration_threshold_m_per_s2: f32,
    /// A call with its clips in is held for the next round while any clip's
    /// clock σ predicted at the call exceeds this multiple of one round's
    /// pick σ (`ClockGraphConfig::pick_offset_sigma_s`, 15 µs): beyond it the
    /// clock model, not the call's own arrival picks, dominates the TDOA
    /// error. That happens after a merge or rejoin, when a model rests on
    /// one round with no rate (±100 ppm prior, ms after a minute). A steady
    /// model a full 90 s heartbeat from its last round sits at 2.0× (filter
    /// covariance with the default `ClockFilterConfig`), so it is never held.
    pub clock_hold_sigma_pick_multiple: f64,
    /// Longest a held call waits after it was received. The post-detection
    /// round is due 0.5 s after the detection and an 8-phone round lands
    /// about 3–3.5 s after the call is received (measured, closeout F2-H1);
    /// 6 s leaves a lost-report margin without stalling a call whose round
    /// never comes.
    pub clock_hold_maximum_wait_ms: MeshMilliseconds,
}

impl Default for ClipFetchConfig {
    fn default() -> Self {
        Self {
            readiness_clip_cap: 6,
            readiness_answer_timeout_ms: 1_500,
            spread_bonus_db_per_m: 0.25,
            spread_bonus_cap_db: 6.0,
            clip_half_window_s: 0.75,
            inter_node_delay_margin_s: 0.1,
            clip_abandon_after_ms: 4_000,
            minimum_clips_for_fix: 3,
            decimation_filter_id: 1,
            island_clip_budget_burst: 36.0,
            island_clip_budget_refill_per_s: 4.0,
            default_query_band_hz: [1_500.0, 10_000.0],
            walking_linear_acceleration_threshold_m_per_s2: 0.8,
            clock_hold_sigma_pick_multiple: 3.0,
            clock_hold_maximum_wait_ms: 6_000,
        }
    }
}

/// Plan §Schedule: the two chirp schedules offered at session setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChirpScheduleMode {
    SilentUntilNeeded,
    Periodic,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChirpSchedulerConfig {
    pub mode: ChirpScheduleMode,
    /// Slow drift-tracking round (plan: 60–120 s).
    pub heartbeat_round_interval_ms: MeshMilliseconds,
    /// Periodic baseline (plan: 5–30 s).
    pub periodic_round_interval_ms: MeshMilliseconds,
    pub post_detection_round_delay_ms: MeshMilliseconds,
    /// Rounds triggered by stillness or invalid clocks are spaced at least this.
    pub triggered_round_minimum_gap_ms: MeshMilliseconds,
    /// Joiners arriving together share one survey round.
    pub new_member_round_minimum_gap_ms: MeshMilliseconds,
    /// Plan: 0.20–0.25 s slots.
    pub slot_duration_ms: MeshMilliseconds,
    /// Lead time between sending the schedule and the first slot.
    pub schedule_lead_time_ms: MeshMilliseconds,
    /// Reverberation tail listened to after the last slot.
    pub round_listen_tail_ms: MeshMilliseconds,
    /// Reports later than this after the listen window are ignored.
    pub round_report_timeout_ms: MeshMilliseconds,
    pub chirp_band_hz: [f32; 2],
    /// Waveform every emitter plays (the band is `chirp_band_hz`).
    pub chirp_waveform_kind: ChirpWaveformKind,
    pub chirp_duration_s: f64,
    /// How far a node's slot estimate can be off: coarse session-clock error
    /// (lowest-RTT alignment, ≈ RTT/2) plus output-latency jitter. The
    /// listener widens each slot's search window by this on both sides.
    pub coarse_schedule_uncertainty_ms: MeshMilliseconds,
}

impl Default for ChirpSchedulerConfig {
    fn default() -> Self {
        Self {
            mode: ChirpScheduleMode::SilentUntilNeeded,
            heartbeat_round_interval_ms: 90_000,
            periodic_round_interval_ms: 15_000,
            post_detection_round_delay_ms: 500,
            triggered_round_minimum_gap_ms: 8_000,
            new_member_round_minimum_gap_ms: 1_500,
            // Phase 1 prestudy: 250 ms fits a 50 ms chirp, ~35 m of flight and
            // the ±coarse-uncertainty margin on both sides.
            slot_duration_ms: 250,
            schedule_lead_time_ms: 400,
            round_listen_tail_ms: 300,
            round_report_timeout_ms: 2_000,
            chirp_band_hz: [13_000.0, 16_000.0],
            chirp_waveform_kind: ChirpWaveformKind::UpDownHyperbolic,
            chirp_duration_s: DEFAULT_CHIRP_DURATION_S,
            coarse_schedule_uncertainty_ms: 30,
        }
    }
}

/// Island clock, layout and anchor processing.
#[derive(Debug, Clone, PartialEq)]
pub struct IslandSyncConfig {
    pub sound_speed_mps: f64,
    /// Two-way closure needs this chirp SNR in both directions.
    pub hearing_minimum_snr_db: f64,
    /// A hearing edge expires when not re-observed for this long.
    pub hearing_memory_ms: MeshMilliseconds,
    pub range_sigma_m: f64,
    /// Heights are not solved; phones are assumed handheld at this height.
    pub assumed_phone_height_m: f64,
    /// Unknown true phone heights: part of every node σ. Handheld phones sit
    /// within about ±0.2 m of the assumed height, so a uniform spread gives
    /// σ = 0.4/√12 ≈ 0.115 m (the old 0.25 m overstated every node σ and,
    /// with it, the LM covariance; closeout C3).
    pub assumed_phone_height_sigma_m: f64,
    /// Growth of a carried-forward layout position's σ while unranged.
    pub relative_position_growth_mps: f64,
    /// Same, for a phone reported walking (absolute walking speed).
    pub walking_position_growth_mps: f64,
    /// A participant whose closing-chirp rate (its ticks per opener tick
    /// across the round) is further than this from 1 lost or gained samples
    /// inside the round: crystals are ±100 ppm and 10 cm of handling motion
    /// over a 1.4 s round is ~200 ppm, while a stall (≥ 5 ms dropped buffer)
    /// inside a ≤ 2.5 s round is ≥ 2000 ppm. Its intervals are biased, so
    /// its ranges and offsets are left out of that round (closeout F2-H1).
    pub maximum_plausible_round_rate_offset: f64,
    /// Below this the GPS chirality choice is reported as low confidence.
    pub confident_chirality_threshold: f64,
    /// Per-bearing DOA noise once the phone's common heading error cancels
    /// (bearing pairs from one stereo phone or array node, one round).
    pub relative_bearing_noise_std_deg: f64,
    /// Per-model d_ii table (plan: calibrated once per model), keyed by the
    /// join descriptor's `platform_name`; unknown models get the default.
    pub self_path_length_by_platform_m: BTreeMap<String, f64>,
    pub default_self_path_length_m: f64,
    /// Per-model speaker position relative to the ranging mic in the phone
    /// body frame (x right, y up the screen, z out of the screen), metres,
    /// for `correct_for_speaker_mic_offsets`. Filled from Phase 1 run A3.
    pub speaker_offset_body_frame_by_platform_m: BTreeMap<String, [f64; 3]>,
    /// Per-model stereo channel the chirp plays on, from Phase 1 run A1:
    /// the bottom speaker where it ranges cleanly, otherwise a measured
    /// earpiece (which then needs its d_ii and speaker vector above).
    /// Unknown models play on the default channel.
    pub chirp_output_channel_by_platform: BTreeMap<String, StereoOutputChannel>,
    pub default_chirp_output_channel: StereoOutputChannel,
    pub clock_graph: ClockGraphConfig,
    pub clock_filter: ClockFilterConfig,
    pub layout_refine: LayoutRefineConfig,
    pub anchor_fit: AnchorFitConfig,
    /// Per-island anchor filter and still-phone jump holding.
    pub anchor_tracking: FilteredMapAnchorConfig,
}

impl Default for IslandSyncConfig {
    fn default() -> Self {
        Self {
            sound_speed_mps: 343.0,
            hearing_minimum_snr_db: 10.0,
            hearing_memory_ms: 200_000,
            range_sigma_m: 0.05,
            assumed_phone_height_m: 1.4,
            assumed_phone_height_sigma_m: 0.115,
            relative_position_growth_mps: 0.3,
            maximum_plausible_round_rate_offset: 2e-3,
            walking_position_growth_mps: 1.5,
            confident_chirality_threshold: 0.9,
            relative_bearing_noise_std_deg: 8.0,
            // Pixel 10 Pro XL chirps from its earpiece (see the channel map
            // below): its self recording shows a body-borne precursor 22.3
            // samples before the airborne arrival, i.e. a speaker ~0.16 m up
            // the body from the bottom mic. Without this d_ii the precursor
            // sits at the edge of the 2 cm default's rejection window and the
            // self pick hops onto it (+3.9 / +7.8 cm range levels, 2026-11-26
            // soak). Lateral offset not measured.
            self_path_length_by_platform_m: BTreeMap::from([("pixel_10_pro_xl".to_string(), 0.16)]),
            default_self_path_length_m: 0.02,
            speaker_offset_body_frame_by_platform_m: BTreeMap::from([("pixel_10_pro_xl".to_string(), [0.0, 0.16, 0.0])]),
            // Run A1. Pixel 7a: right is the earpiece (by ear, 2026-11-25), so
            // Left. Pixel 10 Pro XL (2026-11-26, 1.000 m tape): Right is the
            // earpiece (body precursor above, confirmed by ear). With the d_ii and speaker vector
            // above it ranges 0.986 m, σ 0.6 mm over a 30 min soak. Left
            // (presumably the bottom speaker) gave multi-copy arrivals and a
            // level-dependent 5-9 cm short range, cause not found yet.
            chirp_output_channel_by_platform: BTreeMap::from([
                ("pixel_7a".to_string(), StereoOutputChannel::Left),
                ("pixel_10_pro_xl".to_string(), StereoOutputChannel::Right),
            ]),
            default_chirp_output_channel: StereoOutputChannel::Right,
            clock_graph: ClockGraphConfig::default(),
            clock_filter: ClockFilterConfig::default(),
            layout_refine: LayoutRefineConfig::default(),
            anchor_fit: AnchorFitConfig::default(),
            anchor_tracking: FilteredMapAnchorConfig::default(),
        }
    }
}

/// Thermal headroom thresholds (1 = cool, 0 = throttling imminent).
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalConfig {
    pub node_mic_only_below_headroom: f32,
    pub host_relief_below_headroom: f32,
    /// A handover target must be at least this cool.
    pub successor_minimum_headroom: f32,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            node_mic_only_below_headroom: 0.35,
            host_relief_below_headroom: 0.25,
            successor_minimum_headroom: 0.5,
        }
    }
}

/// Round-robin classification inside chirp-ranged clumps.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationRotaConfig {
    pub enabled: bool,
    /// Plan: "within a few metres".
    pub clump_radius_m: f64,
    pub slot_duration_ms: u32,
}

impl Default for ClassificationRotaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            clump_radius_m: 3.0,
            slot_duration_ms: 1_500,
        }
    }
}

/// The likelihood contour sent with a pinpointed entity: a polygon on the
/// published horizontal covariance (the calibrated fix covariance plus the
/// map anchor's error, `fix_contour`) at `contour_sigma_multiplier`.
/// The multiplier is the Gaussian one, not a fit: a regular 12-gon at 2.465σ
/// holds 94.5 % of a 2-D Gaussian (the calibration target inside the
/// 90–98 % gate band). What was fitted (0.52 until closeout C3) absorbed an
/// overstated LM covariance and so also shrank the honest anchor term; that
/// misstatement now lives in [`FixCovarianceCalibrationConfig`].
#[derive(Debug, Clone, PartialEq)]
pub struct EntityContourConfig {
    pub contour_sigma_multiplier: f64,
    pub contour_point_count: usize,
}

impl Default for EntityContourConfig {
    fn default() -> Self {
        Self {
            contour_sigma_multiplier: 2.465,
            contour_point_count: 12,
        }
    }
}

/// σ scale applied to every mesh `CallFix` covariance (variance × scale²),
/// fitted by `minimappr-sim-gates`' `calibrate_region_parameters` so end-to-end
/// Layer 2 fixes are covered ~94.5 % by the Gaussian contour. Split by
/// geometry because one scalar cannot do both (closeout C3: inside the
/// layout the LM σ is ~1.5× the error, outside ~3×, from conservative node
/// σ and the full source-height band on long baselines). A fix is "inside"
/// when it lies within the circumradius of its contributing nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct FixCovarianceCalibrationConfig {
    pub inside_layout_sigma_scale: f64,
    pub outside_layout_sigma_scale: f64,
}

impl Default for FixCovarianceCalibrationConfig {
    fn default() -> Self {
        Self { inside_layout_sigma_scale: 0.695, outside_layout_sigma_scale: 0.315 }
    }
}

/// Server-track entity tracking on the host (W4-port-server-track): every
/// localized or unlocalized call goes through the perched tracker and the
/// bird-entity registry, so one bird keeps one entity id across bouts.
/// Disabled = one entity update per call (the W3 behaviour).
#[derive(Debug, Clone, PartialEq)]
pub struct EntityTrackingConfig {
    pub enabled: bool,
    pub tracker: crate::wildlife_tracking::tracker_config::PerchedTrackerConfig,
    pub entity: crate::bird_entity::config::BirdEntityConfig,
    /// How often entities follow their tracks, step back and expire.
    pub maintenance_interval_ms: MeshMilliseconds,
    /// A fix whose used nodes' minor/major horizontal spread is below this is
    /// mirror-ambiguous (near-collinear nodes): it never places an entity.
    /// The ratio is of standard deviations (sqrt of the eigenvalue ratio):
    /// 0.2 flags a 5-phone single-file trail with GNSS jitter (~0.05) and a
    /// 20 m triangle whose apex is < ~3.5 m off the baseline, not a ring (~1).
    pub collinear_mirror_max_aspect_ratio: f64,
    /// Per-call entity outcomes kept for revert re-pointing and inspection
    /// (oldest dropped first; at least one is kept). Must cover the calls of
    /// one relocation ghost window.
    pub max_retained_call_outcomes: usize,
}

impl Default for EntityTrackingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tracker: Default::default(),
            entity: Default::default(),
            maintenance_interval_ms: 1_000,
            collinear_mirror_max_aspect_ratio: 0.2,
            max_retained_call_outcomes: 4_096,
        }
    }
}

/// S5 attribution as the host runs it: the cascade, plus how the host builds
/// each cluster's window context (the server's onset-stage seam,
/// `onset_srp_localization_stage._attach_attribution_context`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshAttributionConfig {
    pub cascade: crate::onset_clustering::attribution_cascade::AttributionCascadeConfig,
    /// Interferer onsets this close (same band) to a target onset are the
    /// target's own note or echo and are never masked (server: `echo_max_lag_seconds`).
    pub shared_onset_max_lag_seconds: f64,
    /// Clusters within `radius + fraction × range` of the target are the same
    /// source split in two, never interferers (server: `cluster_radius_m`,
    /// `cluster_range_fraction`; range from the window's node centroid).
    pub colocated_cluster_radius_m: f64,
    pub colocated_cluster_range_fraction: f64,
    /// Post-window label reconciliation (server: `onset_window_label_reconciliation_enabled`,
    /// default off there too): a gradient-settled cluster drops labels that are
    /// another settled cluster's top label in the same window.
    pub window_label_reconciliation_enabled: bool,
}

impl Default for MeshAttributionConfig {
    fn default() -> Self {
        let onset = crate::onset_clustering::OnsetClusteringConfig::default();
        Self {
            cascade: Default::default(),
            shared_onset_max_lag_seconds: onset.echo_max_lag_seconds,
            colocated_cluster_radius_m: onset.cluster_radius_m,
            colocated_cluster_range_fraction: onset.cluster_range_fraction,
            window_label_reconciliation_enabled: false,
        }
    }
}

/// Everything a node or host needs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MeshConfig {
    pub liveness: LivenessConfig,
    pub event_log: EventLogConfig,
    pub matcher: MatcherConfig,
    pub clip_fetch: ClipFetchConfig,
    pub scheduler: ChirpSchedulerConfig,
    pub island_sync: IslandSyncConfig,
    pub thermal: ThermalConfig,
    pub rota: ClassificationRotaConfig,
    pub entity_contour: EntityContourConfig,
    pub fix_covariance_calibration: FixCovarianceCalibrationConfig,
    pub entity_tracking: EntityTrackingConfig,
    /// S5 species attribution for windows holding several source clusters.
    pub attribution: MeshAttributionConfig,
}
