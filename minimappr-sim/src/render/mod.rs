//! Renders a `WorldPlan` into per-device multichannel streams (full-scale
//! f32, device's own sample clock). Paths are rendered only around each
//! emission's receive window, so cost scales with sound content, not
//! session length; the whole-stream work (noise, IIR colouring, limiter)
//! is one fused pass per channel. Emissions are processed one at a time
//! with every channel in parallel (rayon), so per-emission filter caches
//! stay small; every random draw is keyed per stream and accumulation order
//! is fixed, so output is bit-identical regardless of thread scheduling.

pub mod absorption_filter;
pub mod propagation;
pub mod receiver_chain;

use rayon::prelude::*;

use crate::physics::constants::SOUND_SPEED_IN_PHONE_BODY_MPS;
use crate::physics::fractional_delay::FractionalDelayKernel;
use crate::physics::geometry::{add, distance, Vec3};
use crate::world::{PlannedEmission, WorldPlan};
use absorption_filter::{AbsorptionFilterBank, PreparedEmission};
use propagation::{accumulate_path, PathKind, PropagationSettings, ReceiverGeometry};
use receiver_chain::{apply_dropouts, finish_channel};

/// One device's captured audio: `channels[i]` is mic i, full-scale f32.
#[derive(Clone, Debug)]
pub struct RenderedDevice {
    pub device_id: String,
    pub channels: Vec<Vec<f32>>,
}

/// Every propagation path from `emission` to a device mic. Nothing is
/// special-cased: a phone's own chirp reaches its own mics through the same
/// direct path as any other sound (at d_ii), plus the body-conduction
/// precursor that only same-device pairs have.
pub fn paths_for(
    plan: &WorldPlan,
    emission: &PlannedEmission,
    device_index: usize,
    mic_offset_m: Vec3,
) -> Vec<PathKind> {
    let mut paths = vec![PathKind::Direct];
    if let Some((height_m, reflection_coefficient)) = plan.ground {
        paths.push(PathKind::Ground {
            height_m,
            reflection_coefficient,
        });
    }
    paths.extend(plan.reflectors.iter().map(|r| PathKind::Reflector {
        position_m: r.position_m,
        reflection_coefficient: r.reflection_coefficient,
    }));
    let device = &plan.devices[device_index];
    if emission.emitting_device_index == Some(device_index)
        && device.body_conduction_relative_gain > 0.0
    {
        let speaker_position = emission.trajectory.position_at(emission.start_time_s);
        let mic_position = add(
            device.trajectory.position_at(emission.start_time_s),
            mic_offset_m,
        );
        paths.push(PathKind::BodyConduction {
            length_m: distance(speaker_position, mic_position),
            relative_gain: device.body_conduction_relative_gain,
            body_sound_speed_mps: SOUND_SPEED_IN_PHONE_BODY_MPS,
        });
    }
    paths
}

/// Path length used to pick the absorption filter: emitter at the emission
/// midpoint to the mic at that time. Motion during one call changes air
/// absorption negligibly (~0.03 dB/m at 4 kHz).
fn representative_path_length_m(
    plan: &WorldPlan,
    emission: &PlannedEmission,
    receiver: &ReceiverGeometry,
    path: PathKind,
) -> f64 {
    let midpoint_time = emission.start_time_s + 0.5 * emission.duration_s(plan.sample_rate_hz);
    let source = emission.trajectory.position_at(midpoint_time);
    let mic = add(
        receiver.trajectory.position_at(midpoint_time),
        receiver.mic_offset_m,
    );
    match path {
        PathKind::BodyConduction { .. } => 0.0,
        _ => path.length_m(source, mic),
    }
}

/// A channel buffer tagged with its device/mic for flat parallel iteration.
struct ChannelBuffer {
    device_index: usize,
    mic_index: usize,
    samples: Vec<f32>,
}

fn accumulate_emission(
    plan: &WorldPlan,
    emission: &PlannedEmission,
    prepared: &PreparedEmission,
    filters: &AbsorptionFilterBank,
    settings: &PropagationSettings,
    channel: &mut ChannelBuffer,
) {
    let device = &plan.devices[channel.device_index];
    let receiver = ReceiverGeometry {
        trajectory: &device.trajectory,
        mic_offset_m: device.mic_offsets_m[channel.mic_index],
        clock: device.clock,
        sample_count: device.sample_count,
    };
    let output_gain = 1.0 / device.full_scale_pa;
    for path in paths_for(plan, emission, channel.device_index, receiver.mic_offset_m) {
        let filtered = filters.filtered_for_path(
            prepared,
            representative_path_length_m(plan, emission, &receiver, path),
        );
        accumulate_path(
            &mut channel.samples,
            &filtered,
            prepared.offset,
            emission,
            path,
            &receiver,
            output_gain,
            settings,
        );
    }
}

/// Render all devices of `plan`.
pub fn render_world(plan: &WorldPlan) -> Vec<RenderedDevice> {
    let kernel = FractionalDelayKernel::new(plan.render.kernel_config());
    let filters = AbsorptionFilterBank::new(
        plan.atmosphere,
        plan.sample_rate_hz,
        plan.absorption_enabled,
        plan.render.absorption_sharing_tolerance_db,
    );
    let settings = PropagationSettings {
        sample_rate_hz: plan.sample_rate_hz,
        speed_of_sound_mps: plan.speed_of_sound_mps,
        kernel: &kernel,
        control_point_spacing_samples: plan.render.control_point_spacing_samples,
        minimum_path_length_m: plan.render.minimum_path_length_m,
    };
    let guard = plan
        .render
        .absorption_guard_samples
        .max(kernel.half_width_taps() + 2);
    let prepared: Vec<PreparedEmission> = plan
        .emissions
        .par_iter()
        .map(|emission| filters.prepare(emission, guard))
        .collect();

    let mut channels: Vec<ChannelBuffer> = plan
        .devices
        .iter()
        .enumerate()
        .flat_map(|(device_index, device)| {
            (0..device.mic_offsets_m.len()).map(move |mic_index| ChannelBuffer {
                device_index,
                mic_index,
                samples: vec![0f32; device.sample_count],
            })
        })
        .collect();
    for (emission, prepared_emission) in plan.emissions.iter().zip(&prepared) {
        channels.par_iter_mut().for_each(|channel| {
            accumulate_emission(
                plan,
                emission,
                prepared_emission,
                &filters,
                &settings,
                channel,
            )
        });
        prepared_emission.release_filters();
    }
    channels.par_iter_mut().for_each(|channel| {
        finish_channel(
            &mut channel.samples,
            plan,
            channel.device_index,
            channel.mic_index,
        )
    });

    let mut rendered: Vec<RenderedDevice> = plan
        .devices
        .iter()
        .map(|d| RenderedDevice {
            device_id: d.device_id.clone(),
            channels: Vec::new(),
        })
        .collect();
    for channel in channels {
        let dropouts = &plan.devices[channel.device_index].dropouts;
        rendered[channel.device_index]
            .channels
            .push(apply_dropouts(channel.samples, dropouts));
    }
    rendered
}
