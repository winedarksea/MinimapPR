//! Nightly W4-sim-gates runner: every statistical gate on many fresh seeds.
//!
//! cargo run --release -p minimappr-sim-gates --example nightly_sim_gates -- \
//!     [--seeds 200] [--start-seed S] [--gate NAME] [--threads T]
//!
//! `--seeds` is a base unit: each gate runs `seeds × nightly_seed_fraction`
//! seeds (expensive audio gates run fewer, cheap ones more). The start seed
//! defaults to Unix time modulo the training-seed start, so nightly never
//! reuses calibration seeds, and is printed so a run can be repeated. Each
//! per-seed failure prints its `cargo test` repro; each aggregate failure
//! prints the nightly command that reproduces it plus contributing seeds.
//! Exit code 1 if any gate failed.

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
    start_seed: u64,
    gate: Option<String>,
    threads: usize,
}

fn parse_options() -> Options {
    let mut options = Options {
        seed_units: DEFAULT_SEED_UNITS,
        start_seed: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_secs()) % NIGHTLY_SEED_MODULUS,
        gate: None,
        threads: default_thread_count(),
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = || arguments.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--seeds" => options.seed_units = value().parse().expect("--seeds takes a count"),
            "--start-seed" => options.start_seed = value().parse().expect("--start-seed takes an integer"),
            "--gate" => options.gate = Some(value()),
            "--threads" => options.threads = value().parse().expect("--threads takes a count"),
            other => panic!("unknown flag {other}"),
        }
    }
    assert!(
        options.start_seed + options.seed_units * 20 < NIGHTLY_SEED_MODULUS,
        "nightly seeds must stay below the calibration training range ({NIGHTLY_SEED_MODULUS})"
    );
    options
}

fn main() {
    let options = parse_options();
    println!(
        "nightly sim gates: {} seed units from start seed {} on {} threads",
        options.seed_units, options.start_seed, options.threads
    );
    let started = Instant::now();
    let mut failed = Vec::new();
    for gate in GATES.iter().filter(|gate| options.gate.as_deref().is_none_or(|wanted| wanted == gate.name)) {
        let count = ((options.seed_units as f64 * gate.nightly_seed_fraction).round() as u64).max(1);
        let run = GateRun {
            seeds: (options.start_seed..options.start_seed + count).collect(),
            threads: options.threads,
            config: SimGateConfig::default(),
        };
        let gate_started = Instant::now();
        let outcome = (gate.function)(&run);
        println!("{}\n  ({:.1} s)", outcome.report(gate.test_target), gate_started.elapsed().as_secs_f64());
        if !outcome.aggregate_failures.is_empty() {
            println!(
                "  reproduce aggregate: cargo run --release -p minimappr-sim-gates --example nightly_sim_gates -- --gate {} --seeds {} --start-seed {}",
                gate.name, options.seed_units, options.start_seed
            );
        }
        if !outcome.passed() {
            failed.push(gate.name);
        }
    }
    println!("nightly sim gates finished in {:.1} s: {} failed {:?}", started.elapsed().as_secs_f64(), failed.len(), failed);
    if !failed.is_empty() {
        std::process::exit(1);
    }
}
