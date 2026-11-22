//! Scenario parsing, determinism and chaos-knob behaviour.

mod common;

use std::path::{Path, PathBuf};

use common::*;
use minimappr_sim::physics::geometry::Trajectory;
use minimappr_sim::render::render_world;
use minimappr_sim::scenario::chaos::{PocketKnob, RouteChangeKnob};
use minimappr_sim::scenario::devices::ClockConfig;
use minimappr_sim::scenario::Scenario;
use minimappr_sim::world::{plan_world, DropoutCause, EmissionRecord};
use minimappr_sim::{render_scenario, RenderedDevice};

fn scenario_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("scenarios")
        .join(name)
}

#[test]
fn every_example_scenario_parses_and_plans() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("scenarios");
    let mut count = 0;
    for entry in std::fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "toml") {
            let scenario =
                Scenario::from_path(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            plan_world(&scenario).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            count += 1;
        }
    }
    assert!(count >= 4);
}

#[test]
fn unknown_keys_are_rejected() {
    let text = "duration_s = 1.0\nduratoin_s = 2.0\n[[devices]]\ndevice_id = \"a\"\nkind = \"phone\"\nposition_m = [0.0, 0.0, 1.0]\n";
    assert!(Scenario::from_toml_str(text)
        .unwrap_err()
        .contains("duratoin_s"));
}

fn total_samples_hash(rendered: &[RenderedDevice]) -> u64 {
    rendered
        .iter()
        .flat_map(|d| d.channels.iter().flatten())
        .fold(0u64, |hash, s| {
            hash.wrapping_mul(1_099_511_628_211)
                .wrapping_add(s.to_bits() as u64)
        })
}

#[test]
fn rendering_is_deterministic_per_seed() {
    let mut scenario = Scenario::from_path(&scenario_path("roundtrip_small.toml")).unwrap();
    let (_, first, _) = render_scenario(&scenario).unwrap();
    let (_, second, _) = render_scenario(&scenario).unwrap();
    assert_eq!(total_samples_hash(&first), total_samples_hash(&second));
    scenario.seed += 1;
    let (_, other, _) = render_scenario(&scenario).unwrap();
    assert_ne!(total_samples_hash(&first), total_samples_hash(&other));
}

#[test]
fn all_knobs_scenario_expands_every_knob() {
    let scenario = Scenario::from_path(&scenario_path("all_chaos_knobs.toml")).unwrap();
    let plan = plan_world(&scenario).unwrap();
    let device = |id: &str| plan.devices.iter().find(|d| d.device_id == id).unwrap();
    let individuals: Vec<&str> = plan
        .emissions
        .iter()
        .filter_map(|e| match &e.record {
            EmissionRecord::BirdCall(r) => Some(r.individual_id.as_str()),
            _ => None,
        })
        .collect();
    for expected in [
        "flock-1",
        "flock-2",
        "flock-3",
        "same-band-a",
        "same-band-b",
        "wtsp-1",
        "flyover-1",
    ] {
        assert!(individuals.contains(&expected), "missing {expected}");
    }
    let distractor_kinds: Vec<&str> = plan
        .emissions
        .iter()
        .filter_map(|e| match &e.record {
            EmissionRecord::Distractor(d) => Some(d.kind),
            _ => None,
        })
        .collect();
    assert_eq!(
        distractor_kinds.iter().filter(|k| **k == "speech").count(),
        2
    );
    assert_eq!(distractor_kinds.iter().filter(|k| **k == "clap").count(), 1);
    assert!(plan.emissions.iter().any(
        |e| matches!(&e.record, EmissionRecord::Chirp(c) if c.device_id == "phone-4" && c.muted)
    ));
    assert!(plan.emissions.iter().any(|e| matches!(&e.record, EmissionRecord::Chirp(c) if c.device_id == "phone-1" && c.speaker == "earpiece")));
    assert!(!device("phone-2").trajectory.is_stationary(), "walking");
    assert!(
        !device("phone-1").trajectory.is_stationary(),
        "micro-motion"
    );
    assert!(
        plan.devices.iter().all(|d| !d.gps_error_track.is_empty()),
        "gps error"
    );
    assert!(device("phone-2")
        .dropouts
        .iter()
        .any(|d| d.cause == DropoutCause::SampleDrop));
    assert!(device("phone-1")
        .dropouts
        .iter()
        .any(|d| d.cause == DropoutCause::ThermalStall));
    assert!(device("phone-3")
        .gated_filters
        .iter()
        .any(|g| g.kind == "pocket"));
    assert!(device("phone-4")
        .gated_filters
        .iter()
        .any(|g| g.kind == "route_change"));
    assert_eq!(device("phone-1").mic_offsets_m.len(), 2, "second mic");

    let rendered = render_world(&plan);
    for (planned, output) in plan.devices.iter().zip(&rendered) {
        let removed: usize = planned
            .dropouts
            .iter()
            .filter(|d| !d.reported)
            .map(|d| d.length_samples)
            .sum();
        for channel in &output.channels {
            assert_eq!(
                channel.len(),
                planned.sample_count - removed,
                "{} silent drops shorten the stream",
                planned.device_id
            );
        }
        for dropout in planned.dropouts.iter().filter(|d| d.reported) {
            let removed_before: usize = planned
                .dropouts
                .iter()
                .filter(|d| !d.reported && d.start_sample < dropout.start_sample)
                .map(|d| d.length_samples)
                .sum();
            let start = dropout.start_sample - removed_before;
            assert!(
                output.channels[0][start..start + dropout.length_samples]
                    .iter()
                    .all(|s| *s == 0.0),
                "reported drop zero-filled"
            );
        }
    }
}

#[test]
fn toggling_one_knob_does_not_perturb_other_draws() {
    let base = Scenario::from_path(&scenario_path("all_chaos_knobs.toml")).unwrap();
    let mut without_speech = base.clone();
    without_speech.chaos.speech = None;
    let (a, b) = (
        plan_world(&base).unwrap(),
        plan_world(&without_speech).unwrap(),
    );
    for (x, y) in a.devices.iter().zip(&b.devices) {
        assert_eq!(x.clock, y.clock);
        assert_eq!(x.dropouts, y.dropouts);
        assert_eq!(x.trajectory, y.trajectory);
    }
}

/// Render a steady tone to one ideal receiver with an extra knob applied;
/// return the tone level (dB) in [0.5, 1.5) s and [2.5, 3.5) s.
fn tone_levels_before_and_after(tone_hz: f64, configure: impl Fn(&mut Scenario)) -> (f64, f64) {
    let clock = ClockConfig::Fixed {
        offset_s: 0.0,
        drift_ppm: 0.0,
    };
    let mut scenario = clean_scenario(4.0, vec![ideal_receiver("phone-1", [5.0, 0.0, 1.0], clock)]);
    configure(&mut scenario);
    let mut plan = plan_world(&scenario).unwrap();
    push_emission(
        &mut plan,
        tone(tone_hz, 4.0, SAMPLE_RATE_HZ),
        0.0,
        Trajectory::stationary([0.0, 0.0, 1.0]),
    );
    let samples = &render_world(&plan)[0].channels[0];
    let level = |start_s: f64| {
        let window = &samples
            [(start_s * SAMPLE_RATE_HZ) as usize..((start_s + 1.0) * SAMPLE_RATE_HZ) as usize];
        20.0 * windowed_dtft_magnitude(window, tone_hz, SAMPLE_RATE_HZ).log10()
    };
    (level(0.5), level(2.5))
}

#[test]
fn pocket_knob_muffles_high_frequencies_from_its_start() {
    let (before, after) = tone_levels_before_and_after(6000.0, |s| {
        s.chaos.pocket = Some(PocketKnob {
            start_s: 2.0,
            ..PocketKnob::default()
        });
    });
    eprintln!("pocket: {before:.1} dB → {after:.1} dB");
    assert!((before - after - PocketKnob::default().attenuation_db).abs() < 1.0);
}

#[test]
fn route_change_band_limits_to_headset_rate() {
    let (before, after) = tone_levels_before_and_after(12_000.0, |s| {
        s.chaos.route_change = Some(RouteChangeKnob {
            at_s: 2.0,
            ..RouteChangeKnob::default()
        });
    });
    eprintln!("route change at 12 kHz: {before:.1} dB → {after:.1} dB");
    assert!(before - after > 40.0);
}
