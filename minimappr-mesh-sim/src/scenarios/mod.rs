//! The Layer 2 scenario catalogue. Each scenario builds a `ScenarioSpec`,
//! runs it for one seed, and checks the common invariants plus its own
//! assertions from the test strategy's Layer 2 table.

pub mod call_scenarios;
pub mod host_scenarios;
pub mod network_scenarios;
pub mod sync_scenarios;

use crate::invariants::common_invariants;
use crate::scenario::ScenarioSpec;
use crate::scenario_outcome::ScenarioOutcome;
use crate::scenario_runner::ScenarioRun;

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioReport {
    pub name: &'static str,
    pub seed: u64,
    pub summary: String,
}

pub type ScenarioFunction = fn(u64) -> Result<ScenarioReport, String>;

pub const SCENARIOS: &[(&str, ScenarioFunction)] = &[
    ("baseline_localization", call_scenarios::baseline_localization),
    ("host_death", host_scenarios::host_death),
    ("network_faults", network_scenarios::network_faults),
    ("route_change", sync_scenarios::route_change),
    ("island_split_merge", sync_scenarios::island_split_merge),
    ("offline_host_loss", network_scenarios::offline_host_loss),
    ("walking_at_call", sync_scenarios::walking_at_call),
    ("mirrored_layout", sync_scenarios::mirrored_layout),
    ("readiness_cap", call_scenarios::readiness_cap),
    ("mic_only_round_robin", call_scenarios::mic_only_round_robin),
    ("muted_phone", sync_scenarios::muted_phone),
    ("transport_fallback", network_scenarios::transport_fallback),
    ("thermal_shedding", host_scenarios::thermal_shedding),
    ("earbuds", sync_scenarios::earbuds),
    ("constrained_link", network_scenarios::constrained_link),
    ("faint_bird_one_classifier", call_scenarios::faint_bird_one_classifier),
];

pub const SCENARIO_NAMES: [&str; 16] = {
    let mut names = [""; 16];
    let mut index = 0;
    while index < SCENARIOS.len() {
        names[index] = SCENARIOS[index].0;
        index += 1;
    }
    names
};

pub fn run_scenario_by_name(name: &str, seed: u64) -> Result<ScenarioReport, String> {
    let (_, function) = SCENARIOS
        .iter()
        .find(|(candidate, _)| *candidate == name)
        .ok_or_else(|| format!("unknown scenario {name}"))?;
    function(seed)
}

/// Run a spec and apply the common invariants.
pub(crate) fn run_checked(spec: ScenarioSpec, seed: u64) -> Result<ScenarioOutcome, String> {
    let outcome = ScenarioRun::new(spec, seed).run();
    common_invariants(&outcome)?;
    Ok(outcome)
}

/// Prefix failures with the scenario context.
pub(crate) fn report(name: &'static str, seed: u64, result: Result<String, String>) -> Result<ScenarioReport, String> {
    match result {
        Ok(summary) => Ok(ScenarioReport { name, seed, summary }),
        Err(message) => Err(format!("{name} seed {seed}: {message}")),
    }
}
