//! Receiver-side chain applied to a rendered acoustic channel (full-scale
//! units): ambient + self noise → pocket muffling → capsule shadow + mic
//! response → route-change band-limit → limiter/clip; then per-device
//! capture dropouts (shared by all channels of a device).
//!
//! All stages run fused in ONE pass over the stream (streaming filter
//! state): a 60 s channel is ~11 MB and per-stage sweeps were memory-bound.

use crate::physics::biquad::{BiquadCascade, StreamingCascade};
use crate::scenario::devices::LimiterConfig;
use crate::signals::noise::{derive_stream_rng, BackgroundNoiseSource};
use crate::world::{Dropout, GatedReceiverFilter, WorldPlan};

/// Digital full scale; PCM16 conversion clips here.
const FULL_SCALE_PEAK: f32 = 1.0;
/// Gates of this kind act acoustically (before the mic response).
const POCKET_GATE_KIND: &str = "pocket";

/// Soft-knee tanh limiter: identity below the knee, C¹-continuous tanh
/// compression approaching the ceiling above it.
pub fn soft_knee_limit(sample: f32, limiter: &LimiterConfig) -> f32 {
    let knee = limiter.knee_full_scale as f32;
    let ceiling = limiter.ceiling_full_scale as f32;
    let magnitude = sample.abs();
    if magnitude <= knee || ceiling <= knee {
        return sample.clamp(-FULL_SCALE_PEAK, FULL_SCALE_PEAK);
    }
    let headroom = ceiling - knee;
    sample.signum() * (knee + headroom * ((magnitude - knee) / headroom).tanh())
}

/// A time-gated filter in streaming form: always filtering (continuous
/// state, so no start-up transient), output crossfaded dry↔wet at the edges.
struct StreamingGate<'a> {
    state: StreamingCascade<'a>,
    start: f32,
    end: Option<f32>,
    fade: f32,
}

impl<'a> StreamingGate<'a> {
    fn new(gate: &'a GatedReceiverFilter, crossfade_samples: usize) -> Self {
        Self {
            state: gate.filter.streaming_state(),
            start: gate.start_sample as f32,
            end: gate.end_sample.map(|e| e as f32),
            fade: crossfade_samples.max(1) as f32,
        }
    }

    #[inline]
    fn process(&mut self, index: usize, dry: f64) -> f64 {
        let wet = self.state.process_sample(dry);
        let position = index as f32;
        let rise = ((position - self.start) / self.fade + 1.0).clamp(0.0, 1.0);
        let fall = self
            .end
            .map_or(1.0, |end| ((end - position) / self.fade).clamp(0.0, 1.0));
        dry + rise.min(fall) as f64 * (wet - dry)
    }
}

/// Full per-sample receive chain state for one stream segment.
struct ChainState<'a> {
    noise: BackgroundNoiseSource,
    acoustic_gates: Vec<StreamingGate<'a>>,
    capsule: StreamingCascade<'a>,
    electrical_gates: Vec<StreamingGate<'a>>,
    limiter: &'a LimiterConfig,
}

impl ChainState<'_> {
    #[inline]
    fn process(&mut self, index: usize, acoustic: f32) -> f32 {
        let mut value = (acoustic + self.noise.next_sample()) as f64;
        for gate in self.acoustic_gates.iter_mut() {
            value = gate.process(index, value);
        }
        value = self.capsule.process_sample(value);
        for gate in self.electrical_gates.iter_mut() {
            value = gate.process(index, value);
        }
        let value = value as f32;
        if self.limiter.enabled {
            soft_knee_limit(value, self.limiter)
        } else {
            value.clamp(-FULL_SCALE_PEAK, FULL_SCALE_PEAK)
        }
    }
}

/// Independent segments processed in lock-step: IIR filters are
/// latency-bound (each sample waits on the previous), so running several
/// independent recursions side by side lets the core overlap them.
const INTERLEAVED_CHAIN_SEGMENTS: usize = 4;

/// One segment: warm-up starts `warmup` samples before its output range so
/// its filter states converge (residual ∝ e^(−decay·warmup), < 1e-7 for the
/// default 0.1 s at a 40 Hz high-pass) before it starts writing.
struct SegmentPlan {
    warm_start: usize,
    output_start: usize,
    end: usize,
}

fn plan_segments(length: usize, warmup: usize) -> Vec<SegmentPlan> {
    let count = if length > INTERLEAVED_CHAIN_SEGMENTS * 2 * warmup.max(1) {
        INTERLEAVED_CHAIN_SEGMENTS
    } else {
        1
    };
    let segment_length = length.div_ceil(count);
    (0..count)
        .map(|k| {
            let output_start = k * segment_length;
            SegmentPlan {
                warm_start: output_start.saturating_sub(if k == 0 { 0 } else { warmup }),
                output_start,
                end: ((k + 1) * segment_length).min(length),
            }
        })
        .collect()
}

pub fn finish_channel(
    samples: &mut [f32],
    plan: &WorldPlan,
    device_index: usize,
    mic_index: usize,
) {
    let device = &plan.devices[device_index];
    let fs = plan.sample_rate_hz;
    let crossfade = (plan.render.gate_crossfade_s * fs) as usize;
    let warmup = (plan.render.chain_segment_warmup_s * fs) as usize;
    let (acoustic, electrical): (Vec<&GatedReceiverFilter>, Vec<&GatedReceiverFilter>) = device
        .gated_filters
        .iter()
        .partition(|g| g.kind == POCKET_GATE_KIND);
    let mut capsule_cascade: BiquadCascade = device
        .per_mic_filters
        .get(mic_index)
        .cloned()
        .unwrap_or_default();
    capsule_cascade.extend(device.mic_response.clone());
    let segments = plan_segments(samples.len(), warmup);
    let mut states: Vec<ChainState> = (0..segments.len())
        .map(|segment_index| {
            let stream_index =
                ((device_index as u64) << 32) | ((mic_index as u64) << 16) | segment_index as u64;
            ChainState {
                noise: BackgroundNoiseSource::new(
                    &mut derive_stream_rng(plan.seed, "background_noise", stream_index),
                    (device.self_noise_rms_pa / device.full_scale_pa) as f32,
                    (plan.ambient_noise_rms_pa / device.full_scale_pa) as f32,
                    plan.ambient_tilt_corner_hz,
                    fs,
                ),
                acoustic_gates: acoustic
                    .iter()
                    .map(|g| StreamingGate::new(g, crossfade))
                    .collect(),
                capsule: capsule_cascade.streaming_state(),
                electrical_gates: electrical
                    .iter()
                    .map(|g| StreamingGate::new(g, crossfade))
                    .collect(),
                limiter: &device.limiter,
            }
        })
        .collect();
    // In place is safe: segment k reads its warm-up samples at an earlier
    // step than segment k−1 overwrites them (segments are ≥ 2 warm-ups long).
    let longest = segments
        .iter()
        .map(|s| s.end - s.warm_start)
        .max()
        .unwrap_or(0);
    for step in 0..longest {
        for (segment, state) in segments.iter().zip(states.iter_mut()) {
            let index = segment.warm_start + step;
            if index >= segment.end {
                continue;
            }
            let processed = state.process(index, samples[index]);
            if index >= segment.output_start {
                samples[index] = processed;
            }
        }
    }
}

/// Reported dropouts are zero-filled in place; silent ones are removed
/// (later samples shift earlier). `dropouts` must be sorted, non-overlapping.
pub fn apply_dropouts(channel: Vec<f32>, dropouts: &[Dropout]) -> Vec<f32> {
    if dropouts.is_empty() {
        return channel;
    }
    let mut output = Vec::with_capacity(channel.len());
    let mut cursor = 0;
    for dropout in dropouts {
        let start = dropout.start_sample.min(channel.len());
        let end = (dropout.start_sample + dropout.length_samples).min(channel.len());
        output.extend_from_slice(&channel[cursor..start]);
        if dropout.reported {
            output.extend(std::iter::repeat_n(0.0, end - start));
        }
        cursor = end;
    }
    output.extend_from_slice(&channel[cursor..]);
    output
}
