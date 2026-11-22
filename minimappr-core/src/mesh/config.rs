//! Every tunable of the mesh state machines, grouped by concern. Defaults
//! follow the plan's numbers; the Layer 2 simulator overrides per scenario.

use std::collections::BTreeMap;

use crate::layout::gps_anchor_fit::AnchorFitConfig;
use crate::layout::robust_layout_refine::LayoutRefineConfig;
use crate::sync::clock_filter::ClockFilterConfig;
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
            slot_duration_ms: 220,
            schedule_lead_time_ms: 400,
            round_listen_tail_ms: 300,
            round_report_timeout_ms: 2_000,
            chirp_band_hz: [13_000.0, 16_000.0],
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
    /// Unknown true phone heights (held 1.0–1.8 m): part of every node σ.
    pub assumed_phone_height_sigma_m: f64,
    /// Growth of a carried-forward layout position's σ while unranged.
    pub relative_position_growth_mps: f64,
    /// Same, for a phone reported walking (absolute walking speed).
    pub walking_position_growth_mps: f64,
    /// Below this the GPS chirality choice is reported as low confidence.
    pub confident_chirality_threshold: f64,
    /// Per-model d_ii table (plan: calibrated once per model), keyed by the
    /// join descriptor's `platform_name`; unknown models get the default.
    pub self_path_length_by_platform_m: BTreeMap<String, f64>,
    pub default_self_path_length_m: f64,
    pub clock_graph: ClockGraphConfig,
    pub clock_filter: ClockFilterConfig,
    pub layout_refine: LayoutRefineConfig,
    pub anchor_fit: AnchorFitConfig,
}

impl Default for IslandSyncConfig {
    fn default() -> Self {
        Self {
            sound_speed_mps: 343.0,
            hearing_minimum_snr_db: 10.0,
            hearing_memory_ms: 200_000,
            range_sigma_m: 0.05,
            assumed_phone_height_m: 1.4,
            assumed_phone_height_sigma_m: 0.25,
            relative_position_growth_mps: 0.3,
            walking_position_growth_mps: 1.5,
            confident_chirality_threshold: 0.9,
            self_path_length_by_platform_m: BTreeMap::new(),
            default_self_path_length_m: 0.02,
            clock_graph: ClockGraphConfig::default(),
            clock_filter: ClockFilterConfig::default(),
            layout_refine: LayoutRefineConfig::default(),
            anchor_fit: AnchorFitConfig::default(),
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
}
