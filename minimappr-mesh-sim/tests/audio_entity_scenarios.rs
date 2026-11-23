//! Closeout D2, nightly-only: the entity scenarios in rendered-audio mode
//! (Layer 1 audio clips, the real `OnsetClipSetLocalizer`). Seconds per
//! run, so ignored in `cargo test`; run with
//! `cargo test --release -p minimappr-mesh-sim --test audio_entity_scenarios -- --ignored`
//! (PR seeds, or `MINIMAPPR_SIM_SEED=<seed>`), or sweep seeds with
//! `nightly_mesh_seeds --audio`. Results and known failures:
//! `docs/field_mode_progress/closeout_D.md`.

use minimappr_mesh_sim::audio_world::AcousticMode;
use minimappr_mesh_sim::run_scenario_by_name_in_mode;
use minimappr_mesh_sim::seeded_random::seeds_for_test;

fn run_all_seeds_with_audio(name: &str) {
    let mut failures = Vec::new();
    for seed in seeds_for_test() {
        match run_scenario_by_name_in_mode(name, seed, AcousticMode::RenderedAudio) {
            Ok(report) => println!("{name} seed {seed}: {}", report.summary),
            Err(message) => failures.push(format!(
                "{message}\n  reproduce: MINIMAPPR_SIM_SEED={seed} cargo test --release -p minimappr-mesh-sim --test audio_entity_scenarios {name} -- --ignored"
            )),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

macro_rules! audio_scenario_tests {
    ($($name:ident),* $(,)?) => {
        $( #[test] #[ignore = "nightly: rendered-audio mode, seconds per run"] fn $name() { run_all_seeds_with_audio(stringify!($name)); } )*
    };
}

audio_scenario_tests!(
    countersinging_pair,
    flock_echo,
    asymmetric_bouts,
    same_band_two_species,
    omni_to_localized_dedup,
    collinear_trail,
    couples,
);
