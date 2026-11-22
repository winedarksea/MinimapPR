//! Exact time-varying propagation of one emission along one path into one
//! receiver channel.
//!
//! Receiver sample n is taken at true time t = θ + n / (fs (1 + α)). The
//! path contributes g(t) · s(t_e) where the emission time t_e solves
//! t − t_e = L(x_src(t_e), x_rx(t)) / c (retarded time, fixed-point
//! iteration). Because L is recomputed as emitter and receiver move,
//! Doppler falls out with no separate model; because t is the receiver's
//! own clock law, offset and in-clip drift stretch fall out too.

use crate::physics::fractional_delay::FractionalDelayKernel;
use crate::physics::geometry::{distance, Trajectory, Vec3};
use crate::world::{ClockTruth, PlannedEmission};

/// Retarded-time fixed point: v/c ≤ 0.1 contracts error 10× per iteration.
const RETARDED_TIME_MAX_ITERATIONS: usize = 12;
const RETARDED_TIME_TOLERANCE_S: f64 = 1e-13;
/// Probes across the emission used to bound the receive window.
const RECEIVE_WINDOW_PROBE_COUNT: usize = 9;
/// Slack on the receive window for motion between probes.
const RECEIVE_WINDOW_MARGIN_S: f64 = 0.01;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathKind {
    Direct,
    /// Image source mirrored in the plane z = height.
    Ground {
        height_m: f64,
        reflection_coefficient: f64,
    },
    /// Point scatterer (tree trunk).
    Reflector {
        position_m: Vec3,
        reflection_coefficient: f64,
    },
    /// Structural path through the emitting phone's own body.
    BodyConduction {
        length_m: f64,
        relative_gain: f64,
        body_sound_speed_mps: f64,
    },
}

impl PathKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Ground { .. } => "ground",
            Self::Reflector { .. } => "reflector",
            Self::BodyConduction { .. } => "body_conduction",
        }
    }

    /// Path length (m) for emitter at `source` and receiver at `receiver`.
    #[inline]
    pub fn length_m(&self, source: Vec3, receiver: Vec3) -> f64 {
        match *self {
            Self::Direct => distance(source, receiver),
            Self::Ground { height_m, .. } => {
                let image = [source[0], source[1], 2.0 * height_m - source[2]];
                distance(image, receiver)
            }
            Self::Reflector { position_m, .. } => {
                distance(source, position_m) + distance(position_m, receiver)
            }
            Self::BodyConduction { length_m, .. } => length_m,
        }
    }

    #[inline]
    fn gain(&self, length_m: f64, minimum_length_m: f64) -> f64 {
        let spreading = 1.0 / length_m.max(minimum_length_m);
        match *self {
            Self::Direct => spreading,
            Self::Ground {
                reflection_coefficient,
                ..
            }
            | Self::Reflector {
                reflection_coefficient,
                ..
            } => reflection_coefficient * spreading,
            Self::BodyConduction { relative_gain, .. } => relative_gain * spreading,
        }
    }

    #[inline]
    fn propagation_speed(&self, air_speed_mps: f64) -> f64 {
        match *self {
            Self::BodyConduction {
                body_sound_speed_mps,
                ..
            } => body_sound_speed_mps,
            _ => air_speed_mps,
        }
    }
}

/// Receiver channel geometry + clock.
pub struct ReceiverGeometry<'a> {
    pub trajectory: &'a Trajectory,
    pub mic_offset_m: Vec3,
    pub clock: ClockTruth,
    pub sample_count: usize,
}

impl ReceiverGeometry<'_> {
    #[inline]
    fn position_at(&self, true_time_s: f64) -> Vec3 {
        let base = self.trajectory.position_at(true_time_s);
        [
            base[0] + self.mic_offset_m[0],
            base[1] + self.mic_offset_m[1],
            base[2] + self.mic_offset_m[2],
        ]
    }
}

pub struct PropagationSettings<'a> {
    pub sample_rate_hz: f64,
    pub speed_of_sound_mps: f64,
    pub kernel: &'a FractionalDelayKernel,
    pub control_point_spacing_samples: usize,
    pub minimum_path_length_m: f64,
}

/// Emission true time whose sound reaches `receiver_position` at `reception_time_s`.
#[inline]
fn solve_emission_time(
    path: PathKind,
    emitter: &Trajectory,
    receiver_position: Vec3,
    reception_time_s: f64,
    speed_mps: f64,
) -> (f64, f64) {
    let mut length = path.length_m(emitter.position_at(reception_time_s), receiver_position);
    let mut emission_time = reception_time_s - length / speed_mps;
    for _ in 0..RETARDED_TIME_MAX_ITERATIONS {
        length = path.length_m(emitter.position_at(emission_time), receiver_position);
        let next = reception_time_s - length / speed_mps;
        let converged = (next - emission_time).abs() < RETARDED_TIME_TOLERANCE_S;
        emission_time = next;
        if converged {
            break;
        }
    }
    (emission_time, length)
}

/// Receiver-sample window [start, end) that can hear this emission on this path.
fn receive_window(
    path: PathKind,
    emission: &PlannedEmission,
    receiver: &ReceiverGeometry,
    settings: &PropagationSettings,
) -> Option<(usize, usize)> {
    let fs = settings.sample_rate_hz;
    let speed = path.propagation_speed(settings.speed_of_sound_mps);
    let duration = emission.duration_s(fs);
    let (mut earliest, mut latest) = (f64::INFINITY, f64::NEG_INFINITY);
    for probe in 0..RECEIVE_WINDOW_PROBE_COUNT {
        let emission_time = emission.start_time_s
            + duration * probe as f64 / (RECEIVE_WINDOW_PROBE_COUNT - 1) as f64;
        let source = emission.trajectory.position_at(emission_time);
        // Two fixed-point steps on the receiver's position at arrival.
        let mut arrival =
            emission_time + path.length_m(source, receiver.position_at(emission_time)) / speed;
        arrival = emission_time + path.length_m(source, receiver.position_at(arrival)) / speed;
        earliest = earliest.min(arrival);
        latest = latest.max(arrival);
    }
    let kernel_margin_s = (settings.kernel.half_width_taps() as f64 + 2.0) / fs;
    let margin = RECEIVE_WINDOW_MARGIN_S + kernel_margin_s;
    let first = receiver
        .clock
        .sample_of_true_time(earliest - margin, fs)
        .floor()
        .max(0.0);
    let last = receiver
        .clock
        .sample_of_true_time(latest + margin, fs)
        .ceil()
        .min(receiver.sample_count as f64);
    (last > first).then_some((first as usize, last as usize))
}

/// Add `emission` (pre-filtered into `filtered`, whose index `filtered_offset`
/// holds emission sample 0) along `path` into `output`, scaled by `output_gain`.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_path(
    output: &mut [f32],
    filtered: &[f32],
    filtered_offset: usize,
    emission: &PlannedEmission,
    path: PathKind,
    receiver: &ReceiverGeometry,
    output_gain: f64,
    settings: &PropagationSettings,
) {
    let Some((first, last)) = receive_window(path, emission, receiver, settings) else {
        return;
    };
    let fs = settings.sample_rate_hz;
    let speed = path.propagation_speed(settings.speed_of_sound_mps);
    let spacing = settings.control_point_spacing_samples;
    let half_width = settings.kernel.half_width_taps() as f64;
    let emission_length = emission.samples_pa_at_1m.len() as f64;

    // Control points at first, first+spacing, …, ≥ last (inclusive end).
    let control_count = (last - first).div_ceil(spacing) + 1;
    let mut source_positions = Vec::with_capacity(control_count);
    let mut gains = Vec::with_capacity(control_count);
    for k in 0..control_count {
        let sample = (first + k * spacing) as f64;
        let reception_time = receiver.clock.true_time_of_sample(sample, fs);
        let receiver_position = receiver.position_at(reception_time);
        let (emission_time, length) = solve_emission_time(
            path,
            &emission.trajectory,
            receiver_position,
            reception_time,
            speed,
        );
        source_positions.push((emission_time - emission.start_time_s) * fs);
        gains.push(path.gain(length, settings.minimum_path_length_m) * output_gain);
    }

    let inverse_spacing = 1.0 / spacing as f64;
    for (segment, (positions, segment_gains)) in source_positions
        .windows(2)
        .zip(gains.windows(2))
        .enumerate()
    {
        let segment_start = first + segment * spacing;
        let segment_end = (segment_start + spacing).min(last);
        let position_step = (positions[1] - positions[0]) * inverse_spacing;
        let gain_step = (segment_gains[1] - segment_gains[0]) * inverse_spacing;
        for (offset, sample_index) in (segment_start..segment_end).enumerate() {
            let source_position = positions[0] + position_step * offset as f64;
            if source_position < -half_width || source_position > emission_length + half_width {
                continue;
            }
            let gain = segment_gains[0] + gain_step * offset as f64;
            let value = settings
                .kernel
                .evaluate(filtered, source_position + filtered_offset as f64);
            output[sample_index] += (gain as f32) * value;
        }
    }
}
