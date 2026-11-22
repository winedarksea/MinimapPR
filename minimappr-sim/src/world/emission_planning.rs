//! Turns scripted sources and the chirp schedule into `PlannedEmission`s,
//! then appends knob-generated sources (see `knob_sources`).

use std::sync::Arc;

use super::knob_sources::plan_knob_emissions;
use super::{BirdCallRecord, ChirpRecord, EmissionRecord, PlannedDevice, PlannedEmission};
use crate::physics::constants::pascals_from_db_spl;
use crate::physics::geometry::Trajectory;
use crate::scenario::devices::DeviceKind;
use crate::scenario::sources::SourceConfig;
use crate::scenario::Scenario;
use crate::signals::calls::{normalize_active_rms, CallGenerator};
use crate::signals::chirps::generate_chirp;
use crate::signals::noise::derive_stream_rng;

/// Scale a unit-RMS buffer to the requested RMS level at 1 m (Pa).
pub(super) fn to_pascals_at_1m(
    mut unit_rms_samples: Vec<f32>,
    level_db_spl_at_1m: f64,
) -> Arc<Vec<f32>> {
    let gain = pascals_from_db_spl(level_db_spl_at_1m) as f32;
    unit_rms_samples.iter_mut().for_each(|s| *s *= gain);
    Arc::new(unit_rms_samples)
}

pub(super) struct GeneratorSeedStream<'a> {
    pub purpose: &'a str,
    pub index: u64,
}

pub(super) fn bird_call_emission(
    seed_stream: GeneratorSeedStream,
    scenario: &Scenario,
    generator: &CallGenerator,
    level_db_spl_at_1m: f64,
    start_time_s: f64,
    trajectory: Trajectory,
    record: BirdCallRecord,
) -> Result<PlannedEmission, String> {
    let mut rng = derive_stream_rng(scenario.seed, seed_stream.purpose, seed_stream.index);
    let samples = generator.generate(scenario.sample_rate(), &scenario.base_directory, &mut rng)?;
    Ok(PlannedEmission {
        emission_id: record.event_id.clone(),
        samples_pa_at_1m: to_pascals_at_1m(samples, level_db_spl_at_1m),
        start_time_s,
        trajectory,
        emitting_device_index: None,
        record: EmissionRecord::BirdCall(record),
    })
}

fn source_trajectory(source: &SourceConfig) -> Trajectory {
    match &source.trajectory {
        Some(rows) if !rows.is_empty() => Trajectory::from_rows(rows),
        _ => Trajectory::stationary(source.position_m),
    }
}

fn plan_scripted_calls(
    scenario: &Scenario,
    emissions: &mut Vec<PlannedEmission>,
) -> Result<(), String> {
    for (source_index, source) in scenario.sources.iter().enumerate() {
        let trajectory = source_trajectory(source);
        for (call_index, call) in source.calls.iter().enumerate() {
            let event_id = call
                .event_id
                .clone()
                .unwrap_or_else(|| format!("sim-{}-{:03}", source.individual_id, call_index + 1));
            let record = BirdCallRecord {
                event_id,
                individual_id: source.individual_id.clone(),
                species: vec![source.species.clone()],
                label: source
                    .label
                    .clone()
                    .unwrap_or_else(|| source.species.clone()),
                label_category: source.label_category.clone(),
                generator_kind: call.generator.kind_name().to_string(),
                level_db_spl_at_1m: call.level_db_spl_at_1m,
                origin: "scripted",
            };
            let seed_stream = GeneratorSeedStream {
                purpose: "scripted_call",
                index: ((source_index as u64) << 32) | call_index as u64,
            };
            emissions.push(bird_call_emission(
                seed_stream,
                scenario,
                &call.generator,
                call.level_db_spl_at_1m,
                call.start_s,
                trajectory.clone(),
                record,
            )?);
        }
    }
    Ok(())
}

fn plan_chirps(
    scenario: &Scenario,
    devices: &[PlannedDevice],
    emissions: &mut Vec<PlannedEmission>,
) {
    let Some(schedule) = &scenario.chirps else {
        return;
    };
    let mut unit_chirp = generate_chirp(&schedule.waveform, scenario.sample_rate());
    normalize_active_rms(&mut unit_chirp);
    let chirp = to_pascals_at_1m(unit_chirp, schedule.level_db_spl_at_1m);
    let phones: Vec<usize> = devices
        .iter()
        .enumerate()
        .filter(|(_, d)| d.kind == DeviceKind::Phone)
        .map(|(i, _)| i)
        .collect();
    for (round_index, round_start) in schedule.round_start_times_s.iter().enumerate() {
        for (slot, &device_index) in phones.iter().enumerate() {
            let device = &devices[device_index];
            let start_time_s = round_start + slot as f64 * schedule.slot_duration_s;
            for speaker in &device.speakers {
                let gain = if device.speaker_muted {
                    0.0
                } else {
                    speaker.relative_gain as f32
                };
                let samples = if (gain - 1.0).abs() < f32::EPSILON {
                    chirp.clone()
                } else {
                    Arc::new(chirp.iter().map(|s| s * gain).collect())
                };
                emissions.push(PlannedEmission {
                    emission_id: format!(
                        "chirp-r{}-{}-{}",
                        round_index + 1,
                        device.device_id,
                        speaker.name
                    ),
                    samples_pa_at_1m: samples,
                    start_time_s,
                    trajectory: device.trajectory.translated(speaker.offset_m),
                    emitting_device_index: Some(device_index),
                    record: EmissionRecord::Chirp(ChirpRecord {
                        device_id: device.device_id.clone(),
                        round_index,
                        speaker: speaker.name,
                        muted: device.speaker_muted,
                        waveform: schedule.waveform.clone(),
                        level_db_spl_at_1m: schedule.level_db_spl_at_1m,
                    }),
                });
            }
        }
    }
}

pub(super) fn plan_emissions(
    scenario: &Scenario,
    devices: &[PlannedDevice],
) -> Result<Vec<PlannedEmission>, String> {
    let mut emissions = Vec::new();
    plan_scripted_calls(scenario, &mut emissions)?;
    plan_chirps(scenario, devices, &mut emissions);
    plan_knob_emissions(scenario, devices, &mut emissions)?;
    Ok(emissions)
}
