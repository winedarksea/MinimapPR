//! Resolves device configs + device-side chaos knobs (clocks, motion,
//! GPS error, pocket, route change, muted/two speakers, drops, stalls).

use std::f64::consts::TAU;

use rand_chacha::ChaCha8Rng;

use super::motion::{
    compose_with_offsets, gps_error_track, micro_motion_offsets, walking_trajectory,
};
use super::{
    ClockTruth, DeviceSpeaker, Dropout, DropoutCause, GatedReceiverFilter, PlannedDevice,
    PlannedReflector,
};
use crate::physics::biquad::{Biquad, BiquadCascade};
use crate::physics::constants::{amplitude_ratio_from_db, pascals_from_db_spl};
use crate::physics::geometry::{add, Trajectory, Vec3};
use crate::scenario::devices::{
    sirith_offsets_about_centroid_m, ClockConfig, DeviceConfig, DeviceKind,
};
use crate::scenario::Scenario;
use crate::signals::noise::{derive_stream_rng, uniform_in_range};

/// Bluetooth headset codecs roll off a little below their Nyquist rate.
const ROUTE_CHANGE_CUTOFF_FRACTION_OF_NYQUIST: f64 = 0.9;

/// How an empty `device_ids` selector resolves for a given knob.
#[derive(Clone, Copy)]
pub(super) enum EmptySelection {
    AllPhones,
    FirstPhone,
}

pub(super) fn select_devices(
    ids: &[String],
    scenario: &Scenario,
    empty: EmptySelection,
) -> Result<Vec<usize>, String> {
    if ids.is_empty() {
        let phones = scenario
            .devices
            .iter()
            .enumerate()
            .filter(|(_, d)| d.kind == DeviceKind::Phone)
            .map(|(i, _)| i);
        return Ok(match empty {
            EmptySelection::AllPhones => phones.collect(),
            EmptySelection::FirstPhone => phones.take(1).collect(),
        });
    }
    ids.iter()
        .map(|id| {
            scenario
                .devices
                .iter()
                .position(|d| &d.device_id == id)
                .ok_or_else(|| format!("chaos knob references unknown device {id}"))
        })
        .collect()
}

fn base_trajectory(config: &DeviceConfig) -> Trajectory {
    match &config.trajectory {
        Some(rows) if !rows.is_empty() => Trajectory::from_rows(rows),
        _ => Trajectory::stationary(config.position_m),
    }
}

fn draw_clock(config: &DeviceConfig, rng: &mut ChaCha8Rng) -> ClockTruth {
    match config.resolved_clock() {
        ClockConfig::Disciplined => ClockTruth {
            offset_s: 0.0,
            drift_ppm: 0.0,
        },
        ClockConfig::Fixed {
            offset_s,
            drift_ppm,
        } => ClockTruth {
            offset_s,
            drift_ppm,
        },
        ClockConfig::Random {
            max_offset_s,
            max_drift_ppm,
        } => ClockTruth {
            offset_s: uniform_in_range(rng, -max_offset_s, max_offset_s),
            drift_ppm: uniform_in_range(rng, -max_drift_ppm, max_drift_ppm),
        },
    }
}

fn mic_response_cascade(config: &DeviceConfig, fs: f64) -> BiquadCascade {
    let response = config.resolved_mic_response();
    let mut cascade = BiquadCascade::default();
    if let Some(corner) = response.highpass_hz {
        cascade.extend(BiquadCascade::butterworth_highpass(
            response.highpass_order,
            corner,
            fs,
        ));
    }
    if let Some(corner) = response.lowpass_hz.filter(|c| *c < fs / 2.0) {
        cascade.extend(BiquadCascade::butterworth_lowpass(
            response.lowpass_order,
            corner,
            fs,
        ));
    }
    cascade
}

fn plan_single_device(index: usize, config: &DeviceConfig, scenario: &Scenario) -> PlannedDevice {
    let fs = scenario.sample_rate();
    let seed = scenario.seed;
    let response = config.resolved_mic_response();
    let full_scale_pa = pascals_from_db_spl(response.full_scale_db_spl);
    let trajectory = base_trajectory(config);
    let (mic_offsets_m, per_mic_filters) = match config.kind {
        DeviceKind::SirithTetra => {
            let offsets = sirith_offsets_about_centroid_m();
            let filters = vec![BiquadCascade::default(); offsets.len()];
            (offsets, filters)
        }
        DeviceKind::Phone => {
            let mut offsets = vec![[0.0; 3]];
            let mut filters = vec![BiquadCascade::default()];
            if let Some(second) = &config.second_mic {
                offsets.push(second.offset_m);
                filters.push(BiquadCascade::new(vec![Biquad::high_shelf(
                    second.body_shadow_corner_hz,
                    second.body_shadow_gain_db,
                    fs,
                )]));
            }
            (offsets, filters)
        }
    };
    let (speakers, speaker_to_mic_distance_m) = if config.kind == DeviceKind::Phone {
        let mut rng = derive_stream_rng(seed, "speaker_distance", index as u64);
        let range = config.speaker.random_distance_range_m;
        let distance = config
            .speaker
            .speaker_to_mic_distance_m
            .unwrap_or_else(|| uniform_in_range(&mut rng, range[0], range[1]));
        (
            vec![DeviceSpeaker {
                name: "bottom",
                offset_m: [0.0, -distance, 0.0],
                relative_gain: 1.0,
            }],
            Some(distance),
        )
    } else {
        (vec![], None)
    };
    let reported_position_m = trajectory.position_at(0.0);
    PlannedDevice {
        device_id: config.device_id.clone(),
        kind: config.kind,
        trajectory,
        mic_offsets_m,
        per_mic_filters,
        mic_response: mic_response_cascade(config, fs),
        full_scale_pa,
        self_noise_rms_pa: pascals_from_db_spl(response.self_noise_db_spl),
        limiter: config.resolved_limiter(),
        clock: draw_clock(config, &mut derive_stream_rng(seed, "clock", index as u64)),
        sample_count: (scenario.duration_s * fs).round() as usize,
        gated_filters: vec![],
        dropouts: vec![],
        speakers,
        speaker_to_mic_distance_m,
        body_conduction_relative_gain: amplitude_ratio_from_db(
            config
                .speaker
                .body_conduction
                .relative_db(config.speaker.grip),
        ),
        speaker_muted: false,
        reported_position_m,
        gps_error_track: vec![],
    }
}

fn device_sample_at(device: &PlannedDevice, true_time_s: f64, fs: f64) -> usize {
    device
        .clock
        .sample_of_true_time(true_time_s, fs)
        .max(0.0)
        .round() as usize
}

fn apply_motion_knobs(scenario: &Scenario, devices: &mut [PlannedDevice]) -> Result<(), String> {
    let chaos = &scenario.chaos;
    if let Some(knob) = &chaos.walking {
        for index in select_devices(&knob.device_ids, scenario, EmptySelection::AllPhones)? {
            let start = devices[index].trajectory.position_at(0.0);
            let mut rng = derive_stream_rng(scenario.seed, "walking", index as u64);
            devices[index].trajectory =
                walking_trajectory(start, knob, scenario.duration_s, &mut rng);
        }
    }
    if let Some(knob) = &chaos.micro_motion {
        for index in select_devices(&knob.device_ids, scenario, EmptySelection::AllPhones)? {
            let mut rng = derive_stream_rng(scenario.seed, "micro_motion", index as u64);
            let offsets = micro_motion_offsets(knob, &mut rng);
            devices[index].trajectory = compose_with_offsets(&devices[index].trajectory, &offsets);
        }
    }
    if let Some(knob) = &chaos.canopy_gps {
        for (index, device) in devices.iter_mut().enumerate() {
            let mut rng = derive_stream_rng(scenario.seed, "gps_error", index as u64);
            device.gps_error_track = gps_error_track(knob, scenario.duration_s, &mut rng);
            let first = device.gps_error_track[0];
            device.reported_position_m = add(
                device.trajectory.position_at(0.0),
                [first[1], first[2], first[3]],
            );
        }
    }
    Ok(())
}

fn apply_receiver_knobs(scenario: &Scenario, devices: &mut [PlannedDevice]) -> Result<(), String> {
    let chaos = &scenario.chaos;
    let fs = scenario.sample_rate();
    if let Some(knob) = &chaos.pocket {
        for index in select_devices(&knob.device_ids, scenario, EmptySelection::FirstPhone)? {
            let device = &mut devices[index];
            device.gated_filters.push(GatedReceiverFilter {
                kind: "pocket",
                start_sample: device_sample_at(device, knob.start_s, fs),
                end_sample: knob.end_s.map(|end| device_sample_at(device, end, fs)),
                start_s: knob.start_s,
                end_s: knob.end_s,
                filter: BiquadCascade::new(vec![Biquad::high_shelf(
                    knob.corner_hz,
                    -knob.attenuation_db,
                    fs,
                )]),
            });
        }
    }
    if let Some(knob) = &chaos.route_change {
        let cutoff =
            knob.band_limited_sample_rate_hz / 2.0 * ROUTE_CHANGE_CUTOFF_FRACTION_OF_NYQUIST;
        for index in select_devices(&knob.device_ids, scenario, EmptySelection::AllPhones)? {
            let device = &mut devices[index];
            device.gated_filters.push(GatedReceiverFilter {
                kind: "route_change",
                start_sample: device_sample_at(device, knob.at_s, fs),
                end_sample: None,
                start_s: knob.at_s,
                end_s: None,
                filter: BiquadCascade::butterworth_lowpass(
                    knob.filter_order,
                    cutoff.min(fs / 2.0 * ROUTE_CHANGE_CUTOFF_FRACTION_OF_NYQUIST),
                    fs,
                ),
            });
        }
    }
    if let Some(knob) = &chaos.muted_speaker {
        for index in select_devices(&knob.device_ids, scenario, EmptySelection::FirstPhone)? {
            devices[index].speaker_muted = true;
        }
    }
    if let Some(knob) = &chaos.two_speaker_emission {
        for index in select_devices(&knob.device_ids, scenario, EmptySelection::FirstPhone)? {
            let device = &mut devices[index];
            if let Some(bottom) = device.speakers.first().cloned() {
                device.speakers.push(DeviceSpeaker {
                    name: "earpiece",
                    offset_m: add(bottom.offset_m, knob.earpiece_offset_m),
                    relative_gain: amplitude_ratio_from_db(knob.earpiece_relative_db),
                });
            }
        }
    }
    Ok(())
}

fn apply_dropout_knobs(scenario: &Scenario, devices: &mut [PlannedDevice]) -> Result<(), String> {
    let chaos = &scenario.chaos;
    let fs = scenario.sample_rate();
    if let Some(knob) = &chaos.sample_drops {
        for index in select_devices(&knob.device_ids, scenario, EmptySelection::AllPhones)? {
            let mut rng = derive_stream_rng(scenario.seed, "sample_drops", index as u64);
            let device = &mut devices[index];
            for _ in 0..knob.drops_per_device {
                let length = uniform_in_range(
                    &mut rng,
                    knob.min_length_samples as f64,
                    knob.max_length_samples as f64,
                )
                .round() as usize;
                let latest_start = device.sample_count.saturating_sub(length) as f64;
                device.dropouts.push(Dropout {
                    start_sample: uniform_in_range(&mut rng, 0.0, latest_start) as usize,
                    length_samples: length,
                    reported: uniform_in_range(&mut rng, 0.0, 1.0) < knob.reported_probability,
                    cause: DropoutCause::SampleDrop,
                });
            }
        }
    }
    if let Some(knob) = &chaos.thermal_stall {
        let threshold_time =
            knob.ramp_start_s + knob.stall_threshold * (knob.ramp_end_s - knob.ramp_start_s);
        for index in select_devices(&knob.device_ids, scenario, EmptySelection::AllPhones)? {
            let mut rng = derive_stream_rng(scenario.seed, "thermal_stall", index as u64);
            let device = &mut devices[index];
            let mut time_s = threshold_time;
            loop {
                // Exponential inter-stall interval (Poisson stalls once hot).
                time_s += -knob.mean_stall_interval_s
                    * uniform_in_range(&mut rng, f64::MIN_POSITIVE, 1.0).ln();
                if time_s >= scenario.duration_s {
                    break;
                }
                let stall_s = uniform_in_range(
                    &mut rng,
                    knob.stall_duration_range_s[0],
                    knob.stall_duration_range_s[1],
                );
                device.dropouts.push(Dropout {
                    start_sample: device_sample_at(device, time_s, fs),
                    length_samples: (stall_s * fs).round() as usize,
                    reported: uniform_in_range(&mut rng, 0.0, 1.0) < knob.reported_probability,
                    cause: DropoutCause::ThermalStall,
                });
            }
        }
    }
    for device in devices.iter_mut() {
        device.dropouts.sort_by_key(|d| d.start_sample);
        let mut kept: Vec<Dropout> = Vec::with_capacity(device.dropouts.len());
        for dropout in device.dropouts.drain(..) {
            let overlaps = kept
                .last()
                .is_some_and(|last| dropout.start_sample < last.start_sample + last.length_samples);
            let in_range = dropout.start_sample + dropout.length_samples <= device.sample_count;
            if !overlaps && in_range {
                kept.push(dropout);
            }
        }
        device.dropouts = kept;
    }
    Ok(())
}

pub(super) fn plan_devices(scenario: &Scenario) -> Result<Vec<PlannedDevice>, String> {
    let mut devices: Vec<PlannedDevice> = scenario
        .devices
        .iter()
        .enumerate()
        .map(|(index, config)| plan_single_device(index, config, scenario))
        .collect();
    apply_motion_knobs(scenario, &mut devices)?;
    apply_receiver_knobs(scenario, &mut devices)?;
    apply_dropout_knobs(scenario, &mut devices)?;
    Ok(devices)
}

pub(super) fn device_centroid_m(devices: &[PlannedDevice]) -> Vec3 {
    let count = devices.len().max(1) as f64;
    devices.iter().fold([0.0; 3], |acc, d| {
        let p = d.trajectory.position_at(0.0);
        [
            acc[0] + p[0] / count,
            acc[1] + p[1] / count,
            acc[2] + p[2] / count,
        ]
    })
}

pub(super) fn plan_reflectors(
    scenario: &Scenario,
    devices: &[PlannedDevice],
) -> Vec<PlannedReflector> {
    let mut reflectors: Vec<PlannedReflector> = scenario
        .reflectors
        .explicit
        .iter()
        .enumerate()
        .map(|(i, r)| PlannedReflector {
            reflector_id: r
                .reflector_id
                .clone()
                .unwrap_or_else(|| format!("reflector-{}", i + 1)),
            position_m: r.position_m,
            reflection_coefficient: r.reflection_coefficient,
        })
        .collect();
    if let Some(trunks) = &scenario.reflectors.random_trunks {
        let mut rng = derive_stream_rng(scenario.seed, "random_trunks", 0);
        let centroid = device_centroid_m(devices);
        let count = uniform_in_range(
            &mut rng,
            trunks.count_range[0] as f64,
            trunks.count_range[1] as f64 + 1.0,
        )
        .floor() as usize;
        for i in 0..count.min(trunks.count_range[1]) {
            let azimuth = uniform_in_range(&mut rng, 0.0, TAU);
            let distance = uniform_in_range(
                &mut rng,
                trunks.distance_range_m[0],
                trunks.distance_range_m[1],
            );
            let coefficient_db = uniform_in_range(
                &mut rng,
                trunks.coefficient_db_range[0],
                trunks.coefficient_db_range[1],
            );
            reflectors.push(PlannedReflector {
                reflector_id: format!("trunk-{}", i + 1),
                position_m: [
                    centroid[0] + distance * azimuth.cos(),
                    centroid[1] + distance * azimuth.sin(),
                    trunks.height_m,
                ],
                reflection_coefficient: amplitude_ratio_from_db(coefficient_db),
            });
        }
    }
    reflectors
}
