//! The Layer 2 scenario catalogue. Each scenario builds a `ScenarioSpec`,
//! runs it for one seed, and checks the common invariants plus its own
//! assertions from the test strategy's Layer 2 table.

pub mod call_scenarios;
pub mod entity_scenarios;
pub mod host_scenarios;
pub mod network_scenarios;
pub mod sync_scenarios;

use std::cell::Cell;

use crate::audio_world::{AcousticMode, AudioModeConfig};
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
    ("successor_restores_entities", host_scenarios::successor_restores_entities),
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
    ("countersinging_pair", entity_scenarios::countersinging_pair),
    ("flock_echo", entity_scenarios::flock_echo),
    ("asymmetric_bouts", entity_scenarios::asymmetric_bouts),
    ("same_band_two_species", entity_scenarios::same_band_two_species),
    ("omni_to_localized_dedup", entity_scenarios::omni_to_localized_dedup),
    ("collinear_trail", entity_scenarios::collinear_trail),
    ("couples", entity_scenarios::couples),
];

pub const SCENARIO_NAMES: [&str; 24] = {
    let mut names = [""; 24];
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

thread_local! {
    static ACOUSTIC_MODE_OVERRIDE: Cell<Option<AcousticMode>> = const { Cell::new(None) };
    static AUDIO_CONFIG_OVERRIDE: std::cell::RefCell<Option<AudioModeConfig>> = const { std::cell::RefCell::new(None) };
}

/// [`run_scenario_by_name_in_mode`] in rendered-audio mode with `audio` settings.
pub fn run_scenario_by_name_with_audio(name: &str, seed: u64, audio: AudioModeConfig) -> Result<ScenarioReport, String> {
    let previous = AUDIO_CONFIG_OVERRIDE.with(|cell| cell.replace(Some(audio)));
    let result = run_scenario_by_name_in_mode(name, seed, AcousticMode::RenderedAudio);
    AUDIO_CONFIG_OVERRIDE.with(|cell| *cell.borrow_mut() = previous);
    result
}

/// Run a catalogued scenario with every run on this thread in `mode`
/// (closeout D2: the entity scenarios through rendered audio and the real
/// localizer). Scenarios build their specs internally; one override at the
/// single `run_checked` funnel avoids threading a mode through all 24.
pub fn run_scenario_by_name_in_mode(name: &str, seed: u64, mode: AcousticMode) -> Result<ScenarioReport, String> {
    let previous = ACOUSTIC_MODE_OVERRIDE.with(|cell| cell.replace(Some(mode)));
    let result = run_scenario_by_name(name, seed);
    ACOUSTIC_MODE_OVERRIDE.with(|cell| cell.set(previous));
    result
}

/// Run a spec and apply the common invariants.
pub(crate) fn run_checked(mut spec: ScenarioSpec, seed: u64) -> Result<ScenarioOutcome, String> {
    if let Some(mode) = ACOUSTIC_MODE_OVERRIDE.with(Cell::get) {
        spec.acoustic_mode = mode;
    }
    if let Some(audio) = AUDIO_CONFIG_OVERRIDE.with(|cell| cell.borrow().clone()) {
        spec.audio = audio;
    }
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
