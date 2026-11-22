//! Seed sets, gate outcomes and the parallel trial runner.
//!
//! One seed = one trial (layout, source, noise draws). Three disjoint seed
//! ranges keep calibration honest: parameters are fitted on the *training*
//! range, `cargo test` gates on the fixed *pull-request* range, and nightly
//! draws fresh seeds from a start it prints. `MINIMAPPR_SIM_GATES_SEED`
//! narrows every gate to one trial for a repro.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub const GATE_SEED_ENVIRONMENT_VARIABLE: &str = "MINIMAPPR_SIM_GATES_SEED";
/// Calibration fits only ever see seeds in this range.
pub const CALIBRATION_TRAINING_SEED_START: u64 = 5_000_000;
/// Held-out pull-request seeds start here (disjoint from training).
pub const PULL_REQUEST_SEED_START: u64 = 1;
/// Nightly starts should stay below training (the script uses Unix time
/// modulo this, so it never draws a training seed).
pub const NIGHTLY_SEED_MODULUS: u64 = CALIBRATION_TRAINING_SEED_START;
const DEFAULT_THREAD_COUNT: usize = 8;

/// Seeds a `cargo test` gate runs: the env override alone, else `count`
/// fixed seeds from the PR start.
pub fn pull_request_seeds(count: usize) -> Vec<u64> {
    match std::env::var(GATE_SEED_ENVIRONMENT_VARIABLE) {
        Ok(value) => vec![value.trim().parse().expect("MINIMAPPR_SIM_GATES_SEED must be an unsigned integer")],
        Err(_) => (PULL_REQUEST_SEED_START..PULL_REQUEST_SEED_START + count as u64).collect(),
    }
}

pub fn training_seeds(count: usize) -> Vec<u64> {
    (CALIBRATION_TRAINING_SEED_START..CALIBRATION_TRAINING_SEED_START + count as u64).collect()
}

/// One seed that broke a per-trial rule.
#[derive(Debug, Clone, PartialEq)]
pub struct SeedFailure {
    pub seed: u64,
    pub message: String,
}

/// What one gate found over a seed set.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GateOutcome {
    pub gate_name: &'static str,
    pub trial_count: usize,
    /// Human-readable statistics (always printed).
    pub summary_lines: Vec<String>,
    /// Per-trial rule violations (enforced at any seed count).
    pub seed_failures: Vec<SeedFailure>,
    /// Statistical rules over the whole set (enforced only when the set is
    /// large enough; see `CoverageGateConfig::minimum_trials_for_aggregate_gate`).
    pub aggregate_failures: Vec<String>,
    /// Seeds that contributed to an aggregate failure (e.g. uncovered truth),
    /// printed so a nightly failure can be inspected trial by trial.
    pub contributing_seeds: Vec<u64>,
}

impl GateOutcome {
    pub fn new(gate_name: &'static str, trial_count: usize) -> Self {
        Self { gate_name, trial_count, ..Self::default() }
    }

    pub fn passed(&self) -> bool {
        self.seed_failures.is_empty() && self.aggregate_failures.is_empty()
    }

    pub fn fail_seed(&mut self, seed: u64, message: impl Into<String>) {
        self.seed_failures.push(SeedFailure { seed, message: message.into() });
    }

    /// Enforce an aggregate rule only when there are enough trials for it
    /// to mean something; otherwise record it as a note.
    pub fn check_aggregate(&mut self, holds: bool, enough_trials: bool, message: impl FnOnce() -> String) {
        if holds {
            return;
        }
        let text = message();
        if enough_trials {
            self.aggregate_failures.push(text);
        } else {
            self.summary_lines.push(format!("(not gated, too few trials) {text}"));
        }
    }

    pub fn summary(&mut self, line: impl Into<String>) {
        self.summary_lines.push(line.into());
    }

    /// Full report, including the exact command reproducing each failing seed.
    /// `--include-ignored` because known-failure gates are `#[ignore]`d in
    /// the PR subset, and a plain `cargo test` would silently skip them.
    pub fn report(&self, repro_test_target: &str) -> String {
        let mut lines = vec![format!(
            "gate {} over {} trials: {}",
            self.gate_name,
            self.trial_count,
            if self.passed() { "PASS" } else { "FAIL" }
        )];
        lines.extend(self.summary_lines.iter().map(|line| format!("  {line}")));
        for failure in &self.aggregate_failures {
            lines.push(format!("  AGGREGATE FAILURE: {failure}"));
        }
        if !self.contributing_seeds.is_empty() && !self.aggregate_failures.is_empty() {
            lines.push(format!("  contributing seeds: {:?}", self.contributing_seeds));
        }
        for failure in &self.seed_failures {
            lines.push(format!(
                "  SEED FAILURE {}: {}\n    reproduce: {GATE_SEED_ENVIRONMENT_VARIABLE}={} cargo test --release -p minimappr-sim-gates --test {repro_test_target} {} -- --exact --include-ignored --nocapture",
                failure.seed, failure.message, failure.seed, self.gate_name
            ));
        }
        lines.join("\n")
    }
}

pub fn default_thread_count() -> usize {
    std::thread::available_parallelism().map_or(DEFAULT_THREAD_COUNT, |count| count.get())
}

/// Run `trial` for every seed on `threads` workers; results keep seed order.
pub fn run_trials_in_parallel<Result, Trial>(seeds: &[u64], threads: usize, trial: Trial) -> Vec<(u64, Result)>
where
    Result: Send,
    Trial: Fn(u64) -> Result + Sync,
{
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Result>>> = Mutex::new((0..seeds.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..threads.clamp(1, seeds.len().max(1)) {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(&seed) = seeds.get(index) else { break };
                let value = trial(seed);
                results.lock().expect("trial results lock")[index] = Some(value);
            });
        }
    });
    let collected = results.into_inner().expect("trial results lock");
    seeds.iter().copied().zip(collected.into_iter().map(|value| value.expect("every trial ran"))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-failure gates are `#[ignore]`d in the PR subset; a nightly
    /// repro that omitted `--include-ignored` would run zero tests and pass.
    #[test]
    fn seed_repro_command_runs_ignored_gates() {
        let mut outcome = GateOutcome::new("graceful_degradation", 1);
        outcome.fail_seed(42, "example");
        let report = outcome.report("statistical_accuracy");
        let expected = "MINIMAPPR_SIM_GATES_SEED=42 cargo test --release -p minimappr-sim-gates --test statistical_accuracy graceful_degradation -- --exact --include-ignored";
        assert!(report.contains(expected), "{report}");
    }
}
