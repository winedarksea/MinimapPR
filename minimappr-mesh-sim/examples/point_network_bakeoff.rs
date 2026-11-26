//! Nightly point-only localizer bake-off (E8, `docs/point_network_bakeoff.md`).
//!
//! cargo run --release -p minimappr-mesh-sim --example point_network_bakeoff -- \
//!     [--tuning-seeds 40] [--held-out-seeds 200] [--held-out-base B] [--threads T] [--markdown PATH] \
//!     [--settings current_production|historical_pre_km_range]
//! cargo run --release -p minimappr-mesh-sim --example point_network_bakeoff -- --seed S
//!
//! The full run tunes the onset grid reach on seeds 1..=N, then scores the
//! held-out seeds HELD_OUT_SEED_BASE+1..=HELD_OUT_SEED_BASE+M once and applies the pre-registered
//! rule; it prints the markdown report (and writes it to `--markdown`).
//! `--held-out-base` draws fresh held-out seeds (a nightly can pass the date). Exit
//! code 0 either way: the verdict is a finding, not a failure. `--seed S`
//! scores one scene and prints every candidate's per-call result.
//! `--settings` picks the named solver/gate settings (default: the app's current
//! defaults; `historical_pre_km_range` reproduces the recorded 0.491 run).

use minimappr_mesh_sim::bakeoff_candidate_fix::BakeoffCandidate;
use minimappr_mesh_sim::bakeoff_config::{PointBakeoffConfig, SeedSplit, HELD_OUT_SEED_BASE, ONSET_GRID_RANGE_MULTIPLES_TRIED};
use minimappr_mesh_sim::bakeoff_localizers::{run_bakeoff_scene, run_point_bakeoff};
use minimappr_mesh_sim::bakeoff_report::render_markdown;
use minimappr_mesh_sim::bakeoff_settings_profile::BakeoffSettingsProfile;

const DEFAULT_TUNING_SEED_COUNT: u64 = 40;
const DEFAULT_HELD_OUT_SEED_COUNT: u64 = 200;
const DEFAULT_THREAD_COUNT: usize = 8;

struct Options {
    tuning_seed_count: u64,
    held_out_seed_count: u64,
    threads: usize,
    markdown_path: Option<String>,
    single_seed: Option<u64>,
    held_out_base: u64,
    settings_profile: BakeoffSettingsProfile,
}

fn parse_options() -> Options {
    let mut options = Options {
        tuning_seed_count: DEFAULT_TUNING_SEED_COUNT,
        held_out_seed_count: DEFAULT_HELD_OUT_SEED_COUNT,
        threads: std::thread::available_parallelism().map_or(DEFAULT_THREAD_COUNT, |count| count.get()),
        markdown_path: None,
        single_seed: None,
        held_out_base: HELD_OUT_SEED_BASE,
        settings_profile: BakeoffSettingsProfile::default(),
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = || arguments.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--tuning-seeds" => options.tuning_seed_count = value().parse().expect("--tuning-seeds takes a count"),
            "--held-out-seeds" => options.held_out_seed_count = value().parse().expect("--held-out-seeds takes a count"),
            "--threads" => options.threads = value().parse().expect("--threads takes a count"),
            "--markdown" => options.markdown_path = Some(value()),
            "--held-out-base" => options.held_out_base = value().parse().expect("--held-out-base takes an integer"),
            "--seed" => options.single_seed = Some(value().parse().expect("--seed takes an integer")),
            "--settings" => {
                let label = value();
                options.settings_profile = BakeoffSettingsProfile::from_label(&label).unwrap_or_else(|| {
                    let known: Vec<&str> = BakeoffSettingsProfile::ALL.iter().map(|profile| profile.label()).collect();
                    panic!("--settings {label}: expected one of {known:?}")
                });
            }
            other => panic!("unknown flag {other}"),
        }
    }
    options
}

fn print_single_seed(seed: u64, config: &PointBakeoffConfig) {
    let run = run_bakeoff_scene(seed, config, &ONSET_GRID_RANGE_MULTIPLES_TRIED).unwrap_or_else(|error| panic!("seed {seed}: {error}"));
    println!("{:?}", run.score.class);
    for candidate in BakeoffCandidate::ALL {
        let score = &run.score.by_candidate[&candidate];
        println!(
            "{:28} fixes {} misattributed {} solves {:?} ms primary {:?} failures {:?}",
            candidate.label(),
            score.fix_count,
            score.misattributed_fix_count,
            score.solve_wall_times_s.iter().map(|seconds| (seconds * 1e3).round()).collect::<Vec<_>>(),
            score
                .primary_by_call
                .iter()
                .map(|fix| fix.map(|fix| (fix.horizontal_error_m.round(), fix.localized, fix.covariance_covers_truth)))
                .collect::<Vec<_>>(),
            score.failure_reasons
        );
    }
    for (multiple, score) in &run.onset_variant_scores {
        println!(
            "onset grid x{multiple:.1}: primary {:?}",
            score.primary_by_call.iter().map(|fix| fix.map(|fix| (fix.horizontal_error_m.round(), fix.localized))).collect::<Vec<_>>()
        );
    }
}

fn main() {
    let options = parse_options();
    let config = PointBakeoffConfig::for_settings_profile(options.settings_profile);
    if let Some(seed) = options.single_seed {
        print_single_seed(seed, &config);
        return;
    }
    let seeds = SeedSplit::with_held_out_base(options.tuning_seed_count, options.held_out_seed_count, options.held_out_base);
    let report = run_point_bakeoff(seeds, config, options.threads);
    let markdown = render_markdown(&report);
    println!("{markdown}");
    if let Some(path) = options.markdown_path {
        std::fs::write(&path, &markdown).unwrap_or_else(|error| panic!("write {path}: {error}"));
    }
}
