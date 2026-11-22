//! Layer 1 scenario tests for chirp sync: render rounds with the simulator
//! and check every stage against ground truth on fixed seeds.

#[path = "support/mod.rs"]
mod support;

use minimappr_core::layout::gps_anchor_fit::{
    fit_layout_to_gps, AnchorFitConfig, ChiralityCue, GpsFix,
};
use minimappr_core::layout::robust_layout_refine::{
    solve_layout, LayoutRefineConfig, RangeMeasurement,
};
use minimappr_core::sync::acoustic_islands::{acoustic_components, HearingObservation};
use minimappr_core::sync::chirp_codes::SplitMix64;
use minimappr_core::sync::chirp_round::RoundRateCompensation;
use minimappr_core::sync::chirp_waveform::ChirpWaveformKind;
use minimappr_core::sync::first_arrival::FirstArrivalConfig;
use minimappr_sim::scenario::devices::Grip;
use support::sim_sync_scoring::{absolute_percentile, observation_of, score_round, TrialScores};
use support::sim_sync_trial::{render_and_pick, SyncTrialConfig};

/// Six phones spread over ~28 m (heights 0.9–1.5 m), the core use case.
fn six_phone_layout() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 1.2],
        [12.0, 3.0, 1.0],
        [22.0, -4.0, 1.5],
        [6.0, 14.0, 1.3],
        [18.0, 16.0, 0.9],
        [-6.0, 9.0, 1.4],
    ]
}

const FIXED_SEEDS: [u64; 3] = [11, 22, 33];
const NONE: usize = 0;
const TRACKED: usize = 1;
const CLOSING: usize = 2;
/// Precursor lead for the far mic is ≥ 440 µs; a captured pick lands there.
const PRECURSOR_CAPTURE_US: f64 = 150.0;
const WALKING_SPEED_MPS: f64 = 1.2;

fn run(configure: impl Fn(&mut SyncTrialConfig)) -> TrialScores {
    let mut total = TrialScores::default();
    for seed in FIXED_SEEDS {
        let mut config = SyncTrialConfig::baseline(seed, six_phone_layout());
        configure(&mut config);
        let round = render_and_pick(&config, &FirstArrivalConfig::default());
        total.merge(score_round(&round, true));
    }
    total
}

fn walking(config: &mut SyncTrialConfig) {
    config.phone_velocities_mps = (0..config.phone_count())
        .map(|p| {
            let heading = p as f64 * 1.1;
            [
                WALKING_SPEED_MPS * heading.cos(),
                WALKING_SPEED_MPS * heading.sin(),
                0.0,
            ]
        })
        .collect();
}

#[test]
fn still_round_ranges_to_centimetres_and_syncs_clocks() {
    let scores = run(|config| config.waveform = ChirpWaveformKind::UpDownHyperbolic);
    assert_eq!(
        scores.remote_failures, 0,
        "every remote chirp detected at ≤ 28 m"
    );
    assert_eq!(scores.self_failures, 0);
    let tracked = &scores.by_compensation[TRACKED];
    let p95_range = absolute_percentile(&tracked.ranging_errors_m, 95.0);
    let p95_offset = absolute_percentile(&tracked.graph_offset_errors_us, 95.0);
    let p95_layout = absolute_percentile(&tracked.layout_errors_m, 95.0);
    assert!(p95_range < 0.03, "range p95 {p95_range}");
    assert!(
        p95_offset < 50.0,
        "clock gate: graph offset p95 {p95_offset} µs"
    );
    assert!(p95_layout < 0.05, "layout p95 {p95_layout}");
}

#[test]
fn speaker_mic_asymmetry_biases_raw_offsets_until_corrected() {
    let scores = run(|config| config.use_far_mic = true);
    let tracked = &scores.by_compensation[TRACKED];
    let raw = absolute_percentile(&tracked.raw_pair_offset_errors_us, 50.0);
    let corrected = absolute_percentile(&tracked.pair_offset_errors_us, 50.0);
    assert!(raw > 200.0, "raw offset p50 {raw} µs");
    assert!(corrected < 40.0, "corrected offset p50 {corrected} µs");
}

#[test]
fn rate_compensation_beats_none_with_drifting_clocks() {
    let scores = run(|config| config.slot_duration_s = 0.4);
    let p95 = |index: usize| {
        absolute_percentile(&scores.by_compensation[index].graph_offset_errors_us, 95.0)
    };
    assert!(p95(TRACKED) < p95(NONE) && p95(CLOSING) < p95(NONE));
    assert!(p95(TRACKED) < 30.0 && p95(CLOSING) < 30.0);
}

#[test]
fn hard_table_precursor_does_not_capture_self_picks() {
    for use_far_mic in [false, true] {
        let scores = run(|config| {
            config.use_far_mic = use_far_mic;
            config.grip = Grip::HardTable;
        });
        let worst = absolute_percentile(&scores.self_pick_errors_us, 100.0);
        let raw = absolute_percentile(&scores.self_raw_onset_errors_us, 50.0);
        assert!(
            worst < PRECURSOR_CAPTURE_US,
            "far={use_far_mic}: worst self pick {worst} µs"
        );
        assert!(
            raw > 2.0 * PRECURSOR_CAPTURE_US,
            "raw threshold locks onto the precursor"
        );
    }
}

#[test]
fn up_down_chirp_cancels_walking_doppler_bias() {
    let hfm = run(walking);
    let up_down = run(|config| {
        walking(config);
        config.waveform = ChirpWaveformKind::UpDownHyperbolic;
    });
    assert_eq!(hfm.remote_failures + up_down.remote_failures, 0);
    let hfm_p50 = absolute_percentile(&hfm.by_compensation[TRACKED].ranging_errors_m, 50.0);
    let up_down_p50 = absolute_percentile(&up_down.by_compensation[TRACKED].ranging_errors_m, 50.0);
    assert!(
        hfm_p50 > 0.15,
        "single HFM sweep should carry the coupling bias: {hfm_p50}"
    );
    assert!(up_down_p50 < 0.06, "up-down p50 {up_down_p50}");
}

#[test]
fn layout_anchors_to_noisy_gps_with_correct_chirality() {
    const GPS_SIGMA_M: f64 = 3.0;
    let config = SyncTrialConfig {
        waveform: ChirpWaveformKind::UpDownHyperbolic,
        ..SyncTrialConfig::baseline(44, six_phone_layout())
    };
    let round = render_and_pick(&config, &FirstArrivalConfig::default());
    let observation = observation_of(&round);
    let (measurements, _) = observation.pairwise_measurements(
        round.plan.speed_of_sound_mps,
        &RoundRateCompensation::ClosingChirp,
    );
    let ranges: Vec<RangeMeasurement> = measurements
        .iter()
        .map(|m| RangeMeasurement {
            node_i: m.first_participant,
            node_j: m.second_participant,
            distance_m: m.result.distance_m,
            sigma_m: 0.03,
        })
        .collect();
    let truth = six_phone_layout();
    let heights: Vec<f64> = truth.iter().map(|p| p[2]).collect();
    let layout = solve_layout(6, &heights, &ranges, &LayoutRefineConfig::default()).unwrap();
    let local: Vec<[f64; 2]> = layout.positions_m.iter().map(|p| [p[0], p[1]]).collect();
    let mut rng = SplitMix64::new(7);
    let mut gaussian = || {
        let (u1, u2) = (rng.next_unit().max(f64::MIN_POSITIVE), rng.next_unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let fixes: Vec<Option<GpsFix>> = truth
        .iter()
        .map(|p| {
            Some(GpsFix {
                east_m: p[0] + GPS_SIGMA_M * gaussian(),
                north_m: p[1] + GPS_SIGMA_M * gaussian(),
                sigma_m: GPS_SIGMA_M,
            })
        })
        .collect();
    let fit = fit_layout_to_gps(
        &local,
        &fixes,
        ChiralityCue::Unknown,
        &AnchorFitConfig::default(),
    )
    .unwrap();
    assert!(
        fit.chirality_confidence > 0.95,
        "{}",
        fit.chirality_confidence
    );
    for (estimate, true_position) in local.iter().zip(&truth) {
        let mapped = fit.transform.apply(*estimate);
        let error = ((mapped[0] - true_position[0]).powi(2)
            + (mapped[1] - true_position[1]).powi(2))
        .sqrt();
        assert!(error < 2.0 * GPS_SIGMA_M, "map error {error} m");
    }
}

#[test]
fn clumps_out_of_earshot_form_separate_islands() {
    let clump_separation_m = 400.0;
    let mut positions = six_phone_layout()[..3].to_vec();
    positions.extend(
        six_phone_layout()[..3]
            .iter()
            .map(|p| [p[0] + clump_separation_m, p[1], p[2]]),
    );
    let config = SyncTrialConfig {
        waveform: ChirpWaveformKind::UpDownHyperbolic,
        slot_duration_s: 0.4,
        ..SyncTrialConfig::baseline(55, positions)
    };
    let round = render_and_pick(&config, &FirstArrivalConfig::default());
    let snr = |receiver: usize, emitter: usize| {
        round
            .picks
            .iter()
            .find(|p| p.receiver == receiver && p.emission == emitter)
            .and_then(|p| p.snr_db)
    };
    let mut observations = Vec::new();
    for a in 0..6u32 {
        for b in (a + 1)..6u32 {
            observations.push(HearingObservation {
                node_a: a,
                node_b: b,
                snr_a_hears_b_db: snr(a as usize, b as usize),
                snr_b_hears_a_db: snr(b as usize, a as usize),
            });
        }
    }
    let minimum_snr_db = FirstArrivalConfig::default().minimum_snr_db;
    let components = acoustic_components(&[0, 1, 2, 3, 4, 5], &observations, minimum_snr_db);
    assert_eq!(components.len(), 2, "{components:?}");
    assert!(components.iter().all(|c| c.len() == 3));
}
