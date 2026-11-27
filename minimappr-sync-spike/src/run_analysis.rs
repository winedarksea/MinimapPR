//! One run end to end: load both bundles, pick, pair, range, clock
//! residuals, onset stability, leakage and channel-map evidence, into a
//! serialisable [`RunReport`]. `analyse_single` does the one-phone subset
//! (self loopback) used for the single-phone milestone.

use std::path::Path;

use minimappr_core::sync::two_way_ranging::SelfPathLengths;

use crate::analysis_config::AnalysisConfig;
use crate::arrival_scan::{waveform_from_description, ArrivalScanner};
use crate::bundle_loader::PhoneBundle;
use crate::bundle_schema::{RunManifest, DEFAULT_AIR_TEMPERATURE_C};
use crate::capture_continuity::{capture_continuity, continuous_rounds, CaptureContinuity};
use crate::channel_map::{channel_aligned_self_picks, dominant_orientation, emitter_channel_evidence, ChannelMapObservation, EmitterChannelEvidence, EmitterPicks};
use crate::clock_residuals::{clock_residual_report, ClockResidualReport};
use crate::onset_stability::{onset_stability, OnsetStabilityReport};
use crate::microphone_path::{microphone_path_summary, MicrophonePathSummary};
use crate::pair_ranging_series::{
    distance_by_channel_pair, distance_statistics, implied_mean_self_path_m, median_of, range_rounds, ChannelPairDistance, CompensationMode, DistanceStatistics, MicLayout,
    RangedRound,
};
use crate::pick_structure::{pick_structure, PickStructure, PickWithWindow};
use crate::round_pairing::{cross_phone_sample_mapper, pair_rounds, pick_remote_arrivals, pick_self_arrivals, PhonePicks};
use crate::spectral_leakage::{leakage_report, LeakageReport};

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PhoneSummary {
    pub role: String,
    pub platform_name: String,
    pub sdk_int: u32,
    pub capture_source: serde_json::Value,
    pub duration_s: f64,
    pub emissions_played: usize,
    pub capture_gaps: usize,
    pub writer_overflows: usize,
    pub capture_anchor_residual_us: Option<f64>,
    pub capture_rate_offset_ppm: Option<f64>,
    /// Silent drops (timestamp steps) and self-interval breaks.
    pub capture_continuity: CaptureContinuity,
    pub blind_scanned: bool,
    pub onset: OnsetStabilityReport,
    pub self_leakage: LeakageReport,
    pub remote_leakage: Option<LeakageReport>,
    pub remote_detected: usize,
    pub median_remote_snr_db: Option<f64>,
    pub screen_off_events: usize,
    pub volume_zero_seen: bool,
    /// This phone's own chirps, all channels.
    pub self_pick_structure: PickStructure,
    /// The other phone's chirps as heard here, all channels.
    pub remote_pick_structure: PickStructure,
    pub microphone_path: MicrophonePathSummary,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RunReport {
    pub run_label: String,
    pub matrix_run: String,
    pub waveform: crate::bundle_schema::WaveformDescription,
    pub period_ms: u64,
    pub tape_distance_m: Option<f64>,
    pub temperature_c: f64,
    pub temperature_assumed: bool,
    pub speed_of_sound_mps: f64,
    pub self_paths_m: [f64; 2],
    pub speaker_correction_applied: bool,
    pub phones: Vec<PhoneSummary>,
    pub round_count: usize,
    /// Paired rounds left out because a capture/playback break overlaps them.
    pub rounds_excluded_for_capture_breaks: usize,
    pub distance: Vec<DistanceStatistics>,
    pub implied_mean_self_path_m: Option<f64>,
    pub clock: Option<ClockResidualReport>,
    pub channel_map: Vec<ChannelMapObservation>,
    /// Per emitter and channel: self/cross path offsets and pick structure.
    pub channel_evidence: Vec<EmitterChannelEvidence>,
    /// Range split by (A channel, B channel), in `distance_by_channel_mode`.
    pub distance_by_channel: Vec<ChannelPairDistance>,
    pub distance_by_channel_mode: String,
    #[serde(skip)]
    pub ranged_rounds: Vec<RangedRound>,
}

fn count_events(bundle: &PhoneBundle, predicate: impl Fn(&crate::bundle_schema::BundleEvent) -> bool) -> usize {
    bundle.events.iter().filter(|e| predicate(e)).count()
}

fn continuity_of(bundle: &PhoneBundle, picks: &PhonePicks, config: &AnalysisConfig) -> CaptureContinuity {
    let drops = bundle.capture_timeline.as_ref().map_or(&[][..], |timeline| &timeline.silent_drops[..]);
    let aligned = channel_aligned_self_picks(&picks.self_picks, config.sample_rate_hz);
    capture_continuity(&aligned, drops, config.self_interval_break_tolerance_samples, config.sample_rate_hz)
}

/// What one phone needs for its pick structure: its d_ii per channel.
struct PhoneContext<'a> {
    role: &'a str,
    self_path_for_channel: &'a dyn Fn(Option<&str>) -> f64,
    speed_of_sound_mps: f64,
}

fn phone_summary(phone: &PhoneContext, bundle: &PhoneBundle, picks: &PhonePicks, scanner: &ArrivalScanner, config: &AnalysisConfig) -> PhoneSummary {
    let role = phone.role;
    use crate::bundle_schema::BundleEvent as E;
    let self_starts: Vec<f64> = picks.self_picks.iter().filter_map(|p| p.arrival.map(|a| a.sample_index)).collect();
    let remote: Vec<_> = picks.remote_picks.iter().flatten().collect();
    let remote_starts: Vec<f64> = remote.iter().map(|a| a.sample_index).collect();
    let remote_snrs: Vec<f64> = remote.iter().map(|a| a.snr_db).collect();
    let fs = config.sample_rate_hz;
    let continuity = continuity_of(bundle, picks, config);
    PhoneSummary {
        role: role.into(),
        platform_name: bundle.session.platform_name.clone(),
        sdk_int: bundle.session.device.sdk_int,
        capture_source: bundle.session.capture.get("actual_source").cloned().unwrap_or_default(),
        duration_s: bundle.duration_s(),
        emissions_played: picks.self_picks.len(),
        capture_gaps: count_events(bundle, |e| matches!(e, E::CaptureGap { .. })),
        writer_overflows: count_events(bundle, |e| matches!(e, E::WriterOverflow { .. })),
        capture_anchor_residual_us: bundle.capture_timeline.as_ref().map(|t| t.residual_rms_ns() / 1e3),
        capture_rate_offset_ppm: bundle.capture_timeline.as_ref().and_then(|t| t.rate_offset_ppm()),
        onset: onset_stability(&channel_aligned_self_picks(&picks.self_picks, fs), &bundle.markers(), &continuity, fs),
        capture_continuity: continuity,
        blind_scanned: picks.blind_scanned,
        self_leakage: leakage_report(&bundle.audio.samples, &self_starts, scanner.waveform_length(), config.leakage_band_hz, fs),
        remote_leakage: (!remote_starts.is_empty()).then(|| leakage_report(&bundle.audio.samples, &remote_starts, scanner.waveform_length(), config.leakage_band_hz, fs)),
        remote_detected: remote.len(),
        median_remote_snr_db: median_of(&remote_snrs),
        screen_off_events: count_events(bundle, |e| matches!(e, E::Screen { on: false, .. })),
        volume_zero_seen: bundle.events.iter().any(|e| matches!(e, E::Volume { music_index: 0, .. })),
        self_pick_structure: pick_structure(
            &picks
                .self_picks
                .iter()
                .filter_map(|p| {
                    let window = scanner.precursor_lead_window_samples((phone.self_path_for_channel)(p.channel.as_deref()));
                    p.arrival.map(|arrival| PickWithWindow { arrival, precursor_window_samples: window })
                })
                .collect::<Vec<_>>(),
            phone.speed_of_sound_mps,
            fs,
        ),
        remote_pick_structure: pick_structure(
            &remote.iter().map(|arrival| PickWithWindow { arrival: **arrival, precursor_window_samples: None }).collect::<Vec<_>>(),
            phone.speed_of_sound_mps,
            fs,
        ),
        microphone_path: microphone_path_summary(bundle),
    }
}

fn scanner_for(bundle: &PhoneBundle, speed_of_sound_mps: f64, config: &AnalysisConfig) -> Result<ArrivalScanner, String> {
    let waveform = waveform_from_description(&bundle.session.schedule.waveform, config.sample_rate_hz)?;
    Ok(ArrivalScanner::new(&waveform, speed_of_sound_mps, config.sample_rate_hz))
}

pub fn load_manifest(run_directory: &Path) -> Result<RunManifest, String> {
    let text = std::fs::read_to_string(run_directory.join("run_manifest.json")).map_err(|e| format!("run_manifest.json: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("run_manifest.json: {e}"))
}

pub fn analyse_run(run_directory: &Path, config: &AnalysisConfig) -> Result<RunReport, String> {
    let manifest = load_manifest(run_directory)?;
    let role = |name: &str| manifest.roles.get(name).cloned().ok_or(format!("manifest has no role {name}"));
    let (role_a, role_b) = (role("A")?, role("B")?);
    let a = PhoneBundle::load(&run_directory.join(&role_a.dir))?;
    let b = PhoneBundle::load(&run_directory.join(&role_b.dir))?;
    if a.session.schedule.waveform != b.session.schedule.waveform {
        return Err("the two phones played different waveforms".into());
    }
    let speed_of_sound_mps = manifest.speed_of_sound_mps();
    let scanner = scanner_for(&a, speed_of_sound_mps, config)?;
    let default_self_path_m = config.default_self_path_length_m;
    let a_self_path = |channel: Option<&str>| role_a.self_path_for_channel(channel, default_self_path_m);
    let b_self_path = |channel: Option<&str>| role_b.self_path_for_channel(channel, default_self_path_m);
    let self_paths = SelfPathLengths { a_m: a_self_path(None), b_m: b_self_path(None) };
    let (a_self, a_blind) = pick_self_arrivals(&a, &scanner, a_self_path, config);
    let (b_self, b_blind) = pick_self_arrivals(&b, &scanner, b_self_path, config);
    let (a_remote, a_offset) = pick_remote_arrivals(&a, &b, &b_self, &scanner, config);
    let (b_remote, b_offset) = pick_remote_arrivals(&b, &a, &a_self, &scanner, config);
    let a_picks = PhonePicks { self_picks: a_self, remote_picks: a_remote, blind_scanned: a_blind, remote_offset_samples: a_offset };
    let b_picks = PhonePicks { self_picks: b_self, remote_picks: b_remote, blind_scanned: b_blind, remote_offset_samples: b_offset };
    let paired = pair_rounds(&a_picks, &b_picks, config.sample_rate_hz);
    let (a_continuity, b_continuity) = (continuity_of(&a, &a_picks, config), continuity_of(&b, &b_picks, config));
    let (mut rounds, rounds_excluded_for_capture_breaks) = continuous_rounds(paired, &a_continuity, &b_continuity, config.sample_rate_hz);
    for round in &mut rounds {
        let [a_channel, b_channel] = [round.channels[0].as_deref(), round.channels[1].as_deref()];
        round.self_paths = SelfPathLengths { a_m: a_self_path(a_channel), b_m: b_self_path(b_channel) };
        round.speaker_offsets_world_m = role_a.speaker_offset_for_channel(a_channel).zip(role_b.speaker_offset_for_channel(b_channel)).map(|(a, b)| [a, b]);
    }
    let mic_layout = manifest.tape_distance_m.map(|distance| MicLayout { mic_a_m: [0.0, 0.0, 0.0], mic_b_m: [distance, 0.0, 0.0] });
    let speaker_correction_applied = mic_layout.is_some() && rounds.iter().any(|r| r.speaker_offsets_world_m.is_some());
    let ranged = range_rounds(&rounds, speed_of_sound_mps, mic_layout);
    let distance = CompensationMode::ALL
        .iter()
        .filter_map(|mode| distance_statistics(&ranged, *mode, manifest.tape_distance_m, config.range_outlier_threshold_m))
        .collect();
    let clock_mode = [CompensationMode::DoubleSided, CompensationMode::TrackedRateRatio, CompensationMode::None]
        .into_iter()
        .find(|mode| ranged.iter().filter(|r| r.result(*mode).is_some()).count() >= 3);
    let clock = clock_mode.map(|mode| {
        let (points, segment_ids): (Vec<(f64, f64)>, Vec<usize>) = ranged
            .iter()
            .filter_map(|r| r.result(mode).map(|res| ((res.offset_reference_time_on_a_s, res.clock_offset_b_minus_a_s), r.round.capture_segment)))
            .unzip();
        clock_residual_report(&points, &segment_ids, mode.name(), config.heartbeat_cadence_s)
    });
    let by_channel_mode = [CompensationMode::DoubleSided, CompensationMode::TrackedRateRatio, CompensationMode::None]
        .into_iter()
        .find(|mode| ranged.iter().any(|r| r.result(*mode).is_some()))
        .unwrap_or(CompensationMode::None);
    let distance_by_channel = distance_by_channel_pair(&ranged, by_channel_mode, manifest.tape_distance_m, config.range_outlier_threshold_m);
    let phone_a = PhoneContext { role: "A", self_path_for_channel: &a_self_path, speed_of_sound_mps };
    let phone_b = PhoneContext { role: "B", self_path_for_channel: &b_self_path, speed_of_sound_mps };
    let phones = vec![phone_summary(&phone_a, &a, &a_picks, &scanner, config), phone_summary(&phone_b, &b, &b_picks, &scanner, config)];
    let (a_to_b, b_to_a) = (cross_phone_sample_mapper(&a, &b), cross_phone_sample_mapper(&b, &a));
    let (a_window, b_window) = (
        |channel: Option<&str>| scanner.precursor_lead_window_samples(a_self_path(channel)),
        |channel: Option<&str>| scanner.precursor_lead_window_samples(b_self_path(channel)),
    );
    let channel_evidence = vec![
        emitter_channel_evidence(
            &EmitterPicks {
                role: "A",
                platform: &a.session.platform_name,
                self_picks: &a_picks.self_picks,
                remote_picks: &b_picks.remote_picks,
                map_to_listener: &a_to_b,
                precursor_window_for_channel: &a_window,
                listener_self_picks: &b_picks.self_picks,
            },
            speed_of_sound_mps,
            config.sample_rate_hz,
        ),
        emitter_channel_evidence(
            &EmitterPicks {
                role: "B",
                platform: &b.session.platform_name,
                self_picks: &b_picks.self_picks,
                remote_picks: &a_picks.remote_picks,
                map_to_listener: &b_to_a,
                precursor_window_for_channel: &b_window,
                listener_self_picks: &a_picks.self_picks,
            },
            speed_of_sound_mps,
            config.sample_rate_hz,
        ),
    ];
    let channel_map = [(&a, &b, &phones[1]), (&b, &a, &phones[0])]
        .iter()
        .map(|(emitter, listener, listener_summary)| ChannelMapObservation {
            emitter_platform: emitter.session.platform_name.clone(),
            emitter_channel: emitter.session.schedule.channel.clone(),
            emitter_orientation: dominant_orientation(&emitter.events),
            listener_platform: listener.session.platform_name.clone(),
            median_remote_snr_db: listener_summary.median_remote_snr_db,
            remote_detected: listener_summary.remote_detected,
            remote_expected: emitter.emissions().iter().filter(|e| e.played).count(),
        })
        .collect();
    Ok(RunReport {
        run_label: manifest.label.clone(),
        matrix_run: manifest.matrix_run.clone(),
        waveform: a.session.schedule.waveform.clone(),
        period_ms: a.session.schedule.period_ms,
        tape_distance_m: manifest.tape_distance_m,
        temperature_c: manifest.temperature_c.unwrap_or(DEFAULT_AIR_TEMPERATURE_C),
        temperature_assumed: manifest.temperature_c.is_none(),
        speed_of_sound_mps,
        self_paths_m: [self_paths.a_m, self_paths.b_m],
        speaker_correction_applied,
        phones,
        round_count: ranged.len(),
        rounds_excluded_for_capture_breaks,
        distance,
        implied_mean_self_path_m: manifest.tape_distance_m.and_then(|tape| implied_mean_self_path_m(&ranged, tape)),
        clock,
        channel_map,
        channel_evidence,
        distance_by_channel,
        distance_by_channel_mode: by_channel_mode.name().into(),
        ranged_rounds: ranged,
    })
}

/// One phone alone: self loopback only (no ranging).
pub fn analyse_single(phone_directory: &Path, config: &AnalysisConfig) -> Result<PhoneSummary, String> {
    let bundle = PhoneBundle::load(phone_directory)?;
    let scanner = scanner_for(&bundle, RunManifest::default().speed_of_sound_mps(), config)?;
    let self_path = |_: Option<&str>| config.default_self_path_length_m;
    let (self_picks, blind) = pick_self_arrivals(&bundle, &scanner, self_path, config);
    let picks = PhonePicks { self_picks, remote_picks: Vec::new(), blind_scanned: blind, remote_offset_samples: None };
    let phone = PhoneContext { role: &bundle.session.role, self_path_for_channel: &self_path, speed_of_sound_mps: RunManifest::default().speed_of_sound_mps() };
    Ok(phone_summary(&phone, &bundle, &picks, &scanner, config))
}
