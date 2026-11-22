//! Phase 1 chirp-sync pre-study on the Layer 1 simulator.
//!
//! `cargo run --release -p minimappr-core --example phase1_sync_prestudy [-- --quick]`
//!
//! Sweeps chirp band, waveform (LFM / HFM / up-down HFM / coded up-down),
//! slot length, rate compensation, near vs far self-recording mic, grip,
//! level, timing mode and range, one factor at a time around a baseline,
//! with fixed seeds. Prints markdown tables (copied into
//! `docs/field_mode_progress/phase1_sim_prestudy.md`).

#[path = "../tests/support/mod.rs"]
mod support;

use minimappr_core::sync::chirp_waveform::{
    ChirpBand, ChirpWaveformKind, CHIRP_BAND_13_TO_16_KHZ, CHIRP_BAND_15_TO_18_KHZ,
    CHIRP_BAND_18_TO_20_KHZ,
};
use minimappr_core::sync::first_arrival::{ArrivalTiming, FirstArrivalConfig};
use minimappr_sim::scenario::devices::Grip;
use support::sim_sync_scoring::{
    absolute_percentile, fraction_beyond, score_round, TrialScores, COMPENSATION_NAMES,
    WRONG_LOCK_THRESHOLD_US,
};
use support::sim_sync_trial::{render_and_pick, SyncTrialConfig};

const FULL_SEED_COUNT: u64 = 8;
const QUICK_SEED_COUNT: u64 = 2;
const RANGE_LADDER_SEED_MULTIPLIER: u64 = 2;
const FIRST_SEED: u64 = 1000;
const WALKING_SPEED_MPS: f64 = 1.2;
/// Slow body sway of a "still" birder (turning, shifting stance), m/s.
const MICRO_MOTION_SPEED_MPS: f64 = 0.05;
const TRACKED: usize = 1;
/// Weak OEM capture path (sim.md: "Weak phones: ~16 kHz, order 8").
const WEAK_PHONE_MIC_LOWPASS_HZ: f64 = 16_000.0;
const WEAK_PHONE_MIC_LOWPASS_ORDER: usize = 8;

/// Six phones over ~28 m (the core use case).
fn core_layout() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 1.2],
        [12.0, 3.0, 1.0],
        [22.0, -4.0, 1.5],
        [6.0, 14.0, 1.3],
        [18.0, 16.0, 0.9],
        [-6.0, 9.0, 1.4],
    ]
}

/// Six phones over ~45 m (spread-out clumps).
fn wide_layout() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 1.2],
        [20.0, 5.0, 1.0],
        [36.0, -6.0, 1.5],
        [10.0, 24.0, 1.3],
        [30.0, 26.0, 0.9],
        [-9.0, 14.0, 1.4],
    ]
}

fn headings(count: usize, speed: f64) -> Vec<[f64; 3]> {
    (0..count)
        .map(|p| {
            let heading = p as f64 * 1.1;
            [speed * heading.cos(), speed * heading.sin(), 0.0]
        })
        .collect()
}

fn band_name(band: ChirpBand) -> String {
    format!(
        "{:.0}–{:.0} kHz",
        band.low_hz / 1000.0,
        band.high_hz / 1000.0
    )
}

struct Study {
    seeds: u64,
}

impl Study {
    fn run(
        &self,
        layout: fn() -> Vec<[f64; 3]>,
        seeds: u64,
        detector: &FirstArrivalConfig,
        configure: &dyn Fn(&mut SyncTrialConfig),
    ) -> TrialScores {
        let mut total = TrialScores::default();
        for seed in FIRST_SEED..FIRST_SEED + seeds {
            let mut config = SyncTrialConfig::baseline(seed, layout());
            configure(&mut config);
            total.merge(score_round(&render_and_pick(&config, detector), true));
        }
        total
    }
}

fn spread_about_median_p95(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let centred: Vec<f64> = values.iter().map(|v| v - median).collect();
    absolute_percentile(&centred, 95.0)
}

fn percent(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        return "–".into();
    }
    format!("{:.1}%", 100.0 * numerator as f64 / denominator as f64)
}

fn print_header() {
    println!("| Case | Remote miss | Wrong lock | Pick spread p95 (µs) | Range p50 / p95 (cm) | Graph offset p95 (µs) | Layout p50 / p95 (cm) |");
    println!("|---|---|---|---|---|---|---|");
}

fn print_row(label: &str, scores: &TrialScores, compensation: usize) {
    let c = &scores.by_compensation[compensation];
    let remote = &scores.remote_pick_errors_us;
    let median = {
        let mut sorted = remote.clone();
        sorted.sort_by(f64::total_cmp);
        sorted.get(sorted.len() / 2).copied().unwrap_or(0.0)
    };
    let centred: Vec<f64> = remote.iter().map(|v| v - median).collect();
    println!(
        "| {label} | {} | {:.1}% | {:.0} | {:.1} / {:.1} | {:.0} | {:.1} / {:.1} |",
        percent(scores.remote_failures, scores.remote_expected),
        100.0 * fraction_beyond(&centred, WRONG_LOCK_THRESHOLD_US),
        spread_about_median_p95(remote),
        100.0 * absolute_percentile(&c.ranging_errors_m, 50.0),
        100.0 * absolute_percentile(&c.ranging_errors_m, 95.0),
        absolute_percentile(&c.graph_offset_errors_us, 95.0),
        100.0 * absolute_percentile(&c.layout_errors_m, 50.0),
        100.0 * absolute_percentile(&c.layout_errors_m, 95.0),
    );
}

fn band_and_waveform(study: &Study, detector: &FirstArrivalConfig) {
    let waveforms = [
        ChirpWaveformKind::LinearFm,
        ChirpWaveformKind::HyperbolicFm,
        ChirpWaveformKind::UpDownHyperbolic,
    ];
    for (layout_name, layout) in [
        ("core ~28 m", core_layout as fn() -> Vec<[f64; 3]>),
        ("wide ~45 m", wide_layout),
    ] {
        println!("\n### A. Band × waveform, still phones, {layout_name} (tracked rates)\n");
        print_header();
        for band in [
            CHIRP_BAND_13_TO_16_KHZ,
            CHIRP_BAND_15_TO_18_KHZ,
            CHIRP_BAND_18_TO_20_KHZ,
        ] {
            for waveform in waveforms {
                let scores = study.run(layout, study.seeds, detector, &|c| {
                    c.band = band;
                    c.waveform = waveform;
                });
                print_row(
                    &format!("{} {}", band_name(band), waveform.name()),
                    &scores,
                    TRACKED,
                );
            }
        }
    }
}

fn walking_doppler(study: &Study, detector: &FirstArrivalConfig) {
    println!(
        "\n### B. Walking 1.2 m/s (random headings), core layout, 15–18 kHz (tracked rates)\n"
    );
    print_header();
    for waveform in [
        ChirpWaveformKind::LinearFm,
        ChirpWaveformKind::HyperbolicFm,
        ChirpWaveformKind::UpDownHyperbolic,
        ChirpWaveformKind::SessionCodedUpDown,
    ] {
        for band in [CHIRP_BAND_13_TO_16_KHZ, CHIRP_BAND_15_TO_18_KHZ] {
            let scores = study.run(core_layout, study.seeds, detector, &|c| {
                c.waveform = waveform;
                c.band = band;
                c.phone_velocities_mps = headings(c.phone_count(), WALKING_SPEED_MPS);
            });
            print_row(
                &format!("{} {}", band_name(band), waveform.name()),
                &scores,
                TRACKED,
            );
        }
    }
}

fn slots_and_rate_compensation(study: &Study, detector: &FirstArrivalConfig) {
    println!("\n### C. Slot length × rate compensation, core layout, 15–18 kHz up-down HFM\n");
    println!("| Slot | Motion | Compensation | Range p95 (cm) | Pair offset p95 (µs) | Graph offset p95 (µs) | Layout p95 (cm) |");
    println!("|---|---|---|---|---|---|---|");
    for slot in [0.20, 0.25, 0.40] {
        for (motion_name, speed) in [("still", 0.0), ("sway 5 cm/s", MICRO_MOTION_SPEED_MPS)] {
            let scores = study.run(core_layout, study.seeds, detector, &|c| {
                c.slot_duration_s = slot;
                c.waveform = ChirpWaveformKind::UpDownHyperbolic;
                c.phone_velocities_mps = headings(c.phone_count(), speed);
            });
            for (index, name) in COMPENSATION_NAMES.iter().enumerate() {
                let c = &scores.by_compensation[index];
                println!(
                    "| {slot:.2} s | {motion_name} | {name} | {:.1} | {:.0} | {:.0} | {:.1} |",
                    100.0 * absolute_percentile(&c.ranging_errors_m, 95.0),
                    absolute_percentile(&c.pair_offset_errors_us, 95.0),
                    absolute_percentile(&c.graph_offset_errors_us, 95.0),
                    100.0 * absolute_percentile(&c.layout_errors_m, 95.0),
                );
            }
        }
    }
}

fn self_recording(study: &Study, detector: &FirstArrivalConfig) {
    println!("\n### D. Self-recording: near vs far mic × grip × level, core layout, 15–18 kHz up-down HFM\n");
    println!("| Mic | Grip | Level (dB SPL @1 m) | Self miss | Self pick |error| p50 / p95 / max (µs) | Raw-threshold onset |error| p50 (µs) | Raw pair offset p95 (µs) | Corrected pair offset p95 (µs) |");
    println!("|---|---|---|---|---|---|---|---|");
    for far in [false, true] {
        for grip in [Grip::Foam, Grip::Hand, Grip::HardTable] {
            for level in [85.0, 95.0] {
                let scores = study.run(core_layout, study.seeds, detector, &|c| {
                    c.use_far_mic = far;
                    c.grip = grip;
                    c.chirp_level_db_spl_at_1m = level;
                    c.waveform = ChirpWaveformKind::UpDownHyperbolic;
                });
                let c = &scores.by_compensation[TRACKED];
                println!(
                    "| {} | {grip:?} | {level:.0} | {} | {:.0} / {:.0} / {:.0} | {:.0} | {:.0} | {:.0} |",
                    if far { "far (+15 cm)" } else { "near" },
                    percent(scores.self_failures, scores.self_expected),
                    absolute_percentile(&scores.self_pick_errors_us, 50.0),
                    absolute_percentile(&scores.self_pick_errors_us, 95.0),
                    absolute_percentile(&scores.self_pick_errors_us, 100.0),
                    absolute_percentile(&scores.self_raw_onset_errors_us, 50.0),
                    absolute_percentile(&c.raw_pair_offset_errors_us, 95.0),
                    absolute_percentile(&c.pair_offset_errors_us, 95.0),
                );
            }
        }
    }
}

fn timing_mode(study: &Study) {
    println!("\n### E. Arrival timing: envelope peak vs leading edge, core layout, 15–18 kHz up-down HFM\n");
    print_header();
    for (name, timing) in [
        ("envelope peak", ArrivalTiming::EnvelopePeak),
        (
            "leading edge 50%",
            ArrivalTiming::LeadingEdge { fraction: 0.5 },
        ),
    ] {
        let detector = FirstArrivalConfig {
            timing,
            ..FirstArrivalConfig::default()
        };
        let scores = study.run(core_layout, study.seeds, &detector, &|c| {
            c.waveform = ChirpWaveformKind::UpDownHyperbolic
        });
        print_row(name, &scores, TRACKED);
    }
}

fn range_ladder(study: &Study, detector: &FirstArrivalConfig) {
    println!("\n### F. Range ladder: one pair, up-down HFM, 85 dB SPL @1 m, 35 dB SPL ambient\n");
    println!("| Band | Distance | Detection miss | Range |error| p50 / p95 (cm) | Miss with weak mic (16 kHz, order-8 roll-off) |");
    println!("|---|---|---|---|---|");
    for band in [
        CHIRP_BAND_13_TO_16_KHZ,
        CHIRP_BAND_15_TO_18_KHZ,
        CHIRP_BAND_18_TO_20_KHZ,
    ] {
        for distance in [10.0, 20.0, 30.0, 45.0, 60.0, 80.0] {
            let pair = move || vec![[0.0, 0.0, 1.2], [distance, 0.0, 1.2]];
            let seeds = study.seeds * RANGE_LADDER_SEED_MULTIPLIER;
            let configure = |weak_mic: bool| {
                move |c: &mut SyncTrialConfig| {
                    c.phone_positions_m = pair();
                    c.phone_velocities_mps = vec![[0.0; 3]; 2];
                    c.band = band;
                    c.waveform = ChirpWaveformKind::UpDownHyperbolic;
                    c.slot_duration_s = 0.4;
                    if weak_mic {
                        c.mic_lowpass_hz = Some(WEAK_PHONE_MIC_LOWPASS_HZ);
                        c.mic_lowpass_order = WEAK_PHONE_MIC_LOWPASS_ORDER;
                    }
                }
            };
            let normal = study.run(core_layout, seeds, detector, &configure(false));
            let weak = study.run(core_layout, seeds, detector, &configure(true));
            let ranges = &normal.by_compensation[TRACKED].ranging_errors_m;
            println!(
                "| {} | {distance:.0} m | {} | {:.1} / {:.1} | {} |",
                band_name(band),
                percent(normal.remote_failures, normal.remote_expected),
                100.0 * absolute_percentile(ranges, 50.0),
                100.0 * absolute_percentile(ranges, 95.0),
                percent(weak.remote_failures, weak.remote_expected),
            );
        }
    }
}

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    let study = Study {
        seeds: if quick {
            QUICK_SEED_COUNT
        } else {
            FULL_SEED_COUNT
        },
    };
    let detector = FirstArrivalConfig::default();
    println!(
        "# Phase 1 sync pre-study ({} seeds per case, first seed {FIRST_SEED})",
        study.seeds
    );
    println!("\nPick spread = p95 of |pick − median pick| over remote arrivals (the common receive-chain group delay cancels in ranging and offsets). Wrong lock = |pick − median| > {WRONG_LOCK_THRESHOLD_US} µs.");
    band_and_waveform(&study, &detector);
    walking_doppler(&study, &detector);
    slots_and_rate_compensation(&study, &detector);
    self_recording(&study, &detector);
    timing_mode(&study);
    range_ladder(&study, &detector);
}
