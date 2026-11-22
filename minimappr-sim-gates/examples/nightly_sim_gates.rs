//! Nightly W4-sim-gates runner: every statistical gate on many fresh seeds.
//!
//! cargo run --release -p minimappr-sim-gates --example nightly_sim_gates -- \
//!     [--seeds 200] [--start-seed S] [--gate NAME] [--threads T]
//!
//! `--seeds` is a base unit: each gate runs `seeds × nightly_seed_fraction`
//! seeds (expensive audio gates run fewer, cheap ones more). The start seed
//! defaults to Unix time folded so that even the largest gate stays below
//! the training-seed start (nightly never reuses calibration seeds), and is
//! printed so a run can be repeated. Each per-seed failure prints its
//! `cargo test` repro; each aggregate failure prints the nightly command
//! that reproduces it plus contributing seeds. Exit code 1 if any gate failed.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use minimappr_sim_gates::gate_config::SimGateConfig;
use minimappr_sim_gates::gate_seeds::{default_thread_count, NIGHTLY_SEED_MODULUS};
use minimappr_sim_gates::gates::{GateRun, GATES};
use minimappr_sim_gates::heap_accounting::CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const DEFAULT_SEED_UNITS: u64 = 200;

struct Options {
    seed_units: u64,
    start_seed: Option<u64>,
    gate: Option<String>,
    threads: usize,
}

fn parse_options() -> Options {
    let mut options = Options { seed_units: DEFAULT_SEED_UNITS, start_seed: None, gate: None, threads: default_thread_count() };
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = || arguments.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--seeds" => options.seed_units = value().parse().expect("--seeds takes a count"),
            "--start-seed" => options.start_seed = Some(value().parse().expect("--start-seed takes an integer")),
            "--gate" => options.gate = Some(value()),
            "--threads" => options.threads = value().parse().expect("--threads takes a count"),
            other => panic!("unknown flag {other}"),
        }
    }
    options
}

fn gate_seed_count(seed_units: u64, nightly_seed_fraction: f64) -> u64 {
    ((seed_units as f64 * nightly_seed_fraction).round() as u64).max(1)
}

/// The explicit start, else Unix time folded into the range where the
/// largest gate's last seed stays below the calibration training seeds (a
/// plain `time % modulus` can land a few thousand seeds short of training
/// and then overlap it or abort the run).
fn resolved_start_seed(options: &Options) -> u64 {
    let span = GATES.iter().map(|gate| gate_seed_count(options.seed_units, gate.nightly_seed_fraction)).max().unwrap_or(1);
    let room = NIGHTLY_SEED_MODULUS.saturating_sub(span);
    assert!(room > 0, "{span} seeds per gate do not fit below the calibration training range ({NIGHTLY_SEED_MODULUS})");
    let start = options
        .start_seed
        .unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_secs()) % room);
    assert!(
        start + span <= NIGHTLY_SEED_MODULUS,
        "start seed {start} + {span} seeds would reach the calibration training range ({NIGHTLY_SEED_MODULUS})"
    );
    start
}

fn main() {
    let options = parse_options();
    let start_seed = resolved_start_seed(&options);
    println!("nightly sim gates: {} seed units from start seed {start_seed} on {} threads", options.seed_units, options.threads);
    let started = Instant::now();
    let mut failed = Vec::new();
    for gate in GATES.iter().filter(|gate| options.gate.as_deref().is_none_or(|wanted| wanted == gate.name)) {
        let count = gate_seed_count(options.seed_units, gate.nightly_seed_fraction);
        let run = GateRun { seeds: (start_seed..start_seed + count).collect(), threads: options.threads, config: SimGateConfig::default() };
        let gate_started = Instant::now();
        let outcome = (gate.function)(&run);
        println!("{}\n  ({:.1} s)", outcome.report(gate.test_target), gate_started.elapsed().as_secs_f64());
        if !outcome.aggregate_failures.is_empty() {
            println!(
                "  reproduce aggregate: cargo run --release -p minimappr-sim-gates --example nightly_sim_gates -- --gate {} --seeds {} --start-seed {start_seed}",
                gate.name, options.seed_units
            );
        }
        if !outcome.passed() {
            failed.push(gate.name);
        }
    }
    if let Some(wanted) = options.gate.as_deref().filter(|wanted| GATES.iter().all(|gate| gate.name != *wanted)) {
        println!("unknown gate {wanted}");
        std::process::exit(2);
    }
    println!("nightly sim gates finished in {:.1} s: {} failed {:?}", started.elapsed().as_secs_f64(), failed.len(), failed);
    if !failed.is_empty() {
        std::process::exit(1);
    }
}
