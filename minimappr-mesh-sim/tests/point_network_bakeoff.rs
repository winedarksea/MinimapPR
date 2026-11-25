//! E8 point-only localizer bake-off (`docs/point_network_bakeoff.md`).
//!
//! The default suite keeps the harness from rotting: one easy scene both
//! localizers must solve, and a two-seed run of the whole protocol. The full
//! bake-off is nightly / opt-in:
//! `cargo test --release -p minimappr-mesh-sim --test point_network_bakeoff -- --ignored --nocapture`
//! (or the `point_network_bakeoff` example, which also writes the markdown).

use minimappr_mesh_sim::bakeoff_candidate_fix::BakeoffCandidate;
use minimappr_mesh_sim::bakeoff_config::{BirdWaveform, PointBakeoffConfig, SeedSplit, HELD_OUT_SEED_BASE};
use minimappr_mesh_sim::bakeoff_localizers::{run_bakeoff_scene, run_point_bakeoff};
use minimappr_mesh_sim::bakeoff_report::render_markdown;

/// Seeds of the full nightly run (the counts recorded in the doc).
const NIGHTLY_TUNING_SEED_COUNT: u64 = 40;
const NIGHTLY_HELD_OUT_SEED_COUNT: u64 = 200;
const SMOKE_THREAD_COUNT: usize = 2;

/// Six surveyed GPS-locked nodes on a 60 m site, one loud FM-swept bird inside it.
fn easy_scene_config() -> PointBakeoffConfig {
    let mut config = PointBakeoffConfig::default();
    config.scene.bird_waveforms = vec![BirdWaveform::FmSweep];
    config.scene.node_count_range = [6, 6];
    config.scene.site_spread_range_m = [60.0, 60.0];
    config.scene.single_bird_probability = 1.0;
    config.scene.all_gps_probability = 1.0;
    config.scene.hand_measured_site_probability = 0.0;
    config.scene.bird_level_db_spl_at_1m = [100.0, 100.0];
    config.scene.source_radius_spread_fraction = 0.3;
    config
}

/// Seed whose easy scene has a call longer than production's event window.
const EASY_SCENE_SEED: u64 = 7;

fn assert_localizes_easy_scene(candidate: BakeoffCandidate) {
    let run = run_bakeoff_scene(EASY_SCENE_SEED, &easy_scene_config(), &[0.0]).expect("easy scene renders");
    let score = &run.score.by_candidate[&candidate];
    let primary = score.primary_by_call[0];
    assert!(primary.is_some_and(|fix| fix.localized), "{}: {primary:?} ({:?})", candidate.label(), score.failure_reasons);
}

/// Harness wiring: the Cartesian re-solve (behind a repaired front-end) and
/// onset clustering both find an easy bird.
#[test]
fn easy_scene_is_localized_by_repaired_production_and_onset_clustering() {
    assert_localizes_easy_scene(BakeoffCandidate::ProductionRepairedFrontEnd);
    assert_localizes_easy_scene(BakeoffCandidate::OnsetClusteringShadow);
}

/// The production front-end finding (`docs/point_network_bakeoff.md`): the
/// old 50–3430 Hz GCC-PHAT band missed a 3.4–7 kHz sweep, and a rectangular
/// event window cut through a loud call biased every cross-node pair toward
/// zero lag. The per-pair band split (wide, tapered cross-node pairs) fixed it.
#[test]
fn easy_scene_is_localized_by_production() {
    assert_localizes_easy_scene(BakeoffCandidate::ProductionGccPhatLsq);
}

#[test]
fn two_seed_protocol_runs_end_to_end_and_reports_a_verdict() {
    let report =
        run_point_bakeoff(SeedSplit { tuning_seeds: vec![2], held_out_seeds: vec![HELD_OUT_SEED_BASE + 1] }, PointBakeoffConfig::default(), SMOKE_THREAD_COUNT);
    assert!(report.failed_seeds.is_empty(), "{:?}", report.failed_seeds);
    assert_eq!(report.held_out_scores.len(), 1);
    assert_eq!(report.decision.criteria.len(), 5);
    let markdown = render_markdown(&report);
    assert!(markdown.contains("Verdict:"), "{markdown}");
    for candidate in BakeoffCandidate::ALL {
        assert!(markdown.contains(candidate.label()), "{markdown}");
    }
}

#[test]
#[ignore = "nightly: full point-network bake-off, minutes in release"]
fn full_point_network_bakeoff() {
    let threads = std::thread::available_parallelism().map_or(SMOKE_THREAD_COUNT, |count| count.get());
    let report = run_point_bakeoff(SeedSplit::counts(NIGHTLY_TUNING_SEED_COUNT, NIGHTLY_HELD_OUT_SEED_COUNT), PointBakeoffConfig::default(), threads);
    println!("{}", render_markdown(&report));
    assert!(report.failed_seeds.is_empty(), "{:?}", report.failed_seeds);
}
