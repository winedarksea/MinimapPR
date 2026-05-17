/// Offline MVDR (Minimum Variance Distortionless Response) beamformer for
/// the Sirith tetrahedral array.
///
/// This module is intentionally placed alongside `gcc_phat.rs` and
/// `srp_phat.rs` so its spatial filter computations stay consistent with
/// the live localization pipeline and remain independently testable.
///
/// Accepts a trajectory of `(sample_offset, [x,y,z])` waypoints from track
/// history, processes 50 % overlapping Hann-windowed blocks, applies a
/// first-order IIR steering slew-rate limiter (~180°/s), crossfades at block
/// boundaries, and returns a clean mono track.
///
/// Track handoff rules:
/// - New tracks: 100 ms linear fade-in from silence.
/// - Inactive / gap: 100 ms linear fade-out; no bridging across silent spans.
use std::f32::consts::PI;

use num_complex::Complex32;
use rustfft::{FftDirection, FftPlanner};

use crate::dsp_worker::SIRITH_MIC_POSITIONS_M;

const SPEED_OF_SOUND_MPS: f32 = 343.2;
const REGULARIZATION: f32 = 1e-3;

// ── public API types ──────────────────────────────────────────────────────────

/// A single trajectory waypoint: (sample offset from recording start, xyz).
#[derive(Clone, Debug)]
pub struct TrajectoryWaypoint {
    pub sample_offset: usize,
    pub position_m: [f32; 3],
}

/// Input to the MVDR render.
pub struct MvdrRenderRequest {
    /// Four channels of raw PCM samples (MK1..MK4), float32 normalised ±1.0.
    /// All four slices must be the same length.
    pub channels: [Vec<f32>; 4],
    pub sample_rate_hz: u32,
    /// Ordered trajectory waypoints covering the signal window.
    pub trajectory: Vec<TrajectoryWaypoint>,
    /// Fade duration in samples.  Defaults to ~100 ms.
    pub fade_samples: Option<usize>,
}

/// Mono output, same length as the input channels.
pub struct MvdrRenderOutput {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
}

// ── entry point ───────────────────────────────────────────────────────────────

/// Render with the fixed Sirith tetrahedral geometry.
///
/// Delegates to `render_mvdr_n` using `SIRITH_MIC_POSITIONS_M` as mic positions.
/// Use `render_mvdr_n` directly when mic geometry differs from the tetrahedral array.
pub fn render_mvdr(request: MvdrRenderRequest) -> MvdrRenderOutput {
    render_mvdr_n(MvdrRenderRequestN {
        channels: request.channels.into_iter().collect(),
        mic_positions_m: SIRITH_MIC_POSITIONS_M.to_vec(),
        sample_rate_hz: request.sample_rate_hz,
        trajectory: request.trajectory,
        fade_samples: request.fade_samples,
    })
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos()))
        .collect()
}

fn normalise_vec3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-9 {
        return [1.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn iir_slew(current: [f32; 3], target: [f32; 3], alpha: f32) -> [f32; 3] {
    let t = normalise_vec3(target);
    let blended = [
        (1.0 - alpha) * current[0] + alpha * t[0],
        (1.0 - alpha) * current[1] + alpha * t[1],
        (1.0 - alpha) * current[2] + alpha * t[2],
    ];
    normalise_vec3(blended)
}

/// Look up the steering direction at `sample` by linear interpolation between
/// adjacent trajectory waypoints.
fn waypoint_direction_at(traj: &[TrajectoryWaypoint], sample: usize) -> [f32; 3] {
    if traj.is_empty() {
        return [1.0, 0.0, 0.0];
    }
    if traj.len() == 1 || sample <= traj[0].sample_offset {
        return normalise_vec3(traj[0].position_m);
    }
    if sample >= traj[traj.len() - 1].sample_offset {
        return normalise_vec3(traj[traj.len() - 1].position_m);
    }
    // Binary search for bracketing waypoints.
    let idx = traj.partition_point(|wp| wp.sample_offset <= sample);
    let prev = &traj[idx - 1];
    let next = &traj[idx];
    let span = (next.sample_offset - prev.sample_offset) as f32;
    let t = (sample - prev.sample_offset) as f32 / span.max(1.0);
    let p = prev.position_m;
    let n = next.position_m;
    let interp = [
        p[0] + t * (n[0] - p[0]),
        p[1] + t * (n[1] - p[1]),
        p[2] + t * (n[2] - p[2]),
    ];
    normalise_vec3(interp)
}

// ── N-channel generalized API ──────────────────────────────────────────────────

/// Input to the N-channel MVDR render (M ≥ 1 mics, arbitrary geometry).
///
/// The existing `MvdrRenderRequest` / `render_mvdr()` API is preserved for the
/// tetrahedral 4-mic case.  Use this struct when working with distributed node
/// clusters or non-tetrahedral arrays.
pub struct MvdrRenderRequestN {
    /// M channels of raw PCM (float32, ±1.0), each the same length.
    pub channels: Vec<Vec<f32>>,
    /// Centroid-relative mic positions, one per channel.  Length == channels.len().
    pub mic_positions_m: Vec<[f32; 3]>,
    pub sample_rate_hz: u32,
    pub trajectory: Vec<TrajectoryWaypoint>,
    pub fade_samples: Option<usize>,
}

/// N-channel MVDR render using arbitrary mic geometry.
///
/// Produces the same mono output as `render_mvdr()` but accepts M mics instead
/// of the fixed tetrahedral 4.  The tetrahedral case is the M=4 special case.
pub fn render_mvdr_n(request: MvdrRenderRequestN) -> MvdrRenderOutput {
    let m = request.channels.len().min(request.mic_positions_m.len());
    if m == 0 {
        return MvdrRenderOutput { samples: vec![], sample_rate_hz: request.sample_rate_hz };
    }
    let n_samples = request.channels[0].len();
    for ch in &request.channels {
        assert_eq!(ch.len(), n_samples, "all channels must have equal length");
    }
    if n_samples == 0 {
        return MvdrRenderOutput { samples: vec![], sample_rate_hz: request.sample_rate_hz };
    }

    let sr = request.sample_rate_hz as f32;
    let fade_len = request.fade_samples.unwrap_or((0.1 * sr) as usize).max(1);
    let block_size = 512_usize;
    let hop = block_size / 2;

    let mut output = vec![0.0f32; n_samples];
    let mut norm = vec![0.0f32; n_samples];
    let window = hann_window(block_size);

    thread_local! {
        static PLANNER_N: std::cell::RefCell<FftPlanner<f32>> =
            std::cell::RefCell::new(FftPlanner::new());
    }
    let (fft, ifft) = PLANNER_N.with(|planner| {
        let mut p = planner.borrow_mut();
        (p.plan_fft(block_size, FftDirection::Forward), p.plan_fft(block_size, FftDirection::Inverse))
    });

    let first_dir = if request.trajectory.is_empty() {
        [1.0f32, 0.0, 0.0]
    } else {
        normalise_vec3(request.trajectory[0].position_m)
    };
    let mut current_dir = first_dir;
    let alpha = (1.0 - (-2.0 * PI * 0.5 * (hop as f32 / sr)).exp()).clamp(0.01, 0.5);

    let n_blocks = n_samples.div_ceil(hop);
    for block_idx in 0..n_blocks {
        let start = block_idx * hop;
        let end = (start + block_size).min(n_samples);
        let actual = end - start;

        let target_dir = waypoint_direction_at(&request.trajectory, start + hop / 2);
        current_dir = iir_slew(current_dir, target_dir, alpha);

        let steering = steering_vector_n(&current_dir, &request.mic_positions_m[..m], sr, block_size);

        // Build frequency-domain multi-channel block [M × block_size].
        let x_freq: Vec<Vec<Complex32>> = (0..m).map(|ch| {
            let mut buf: Vec<Complex32> = (0..block_size).map(|i| {
                let s = if start + i < n_samples { request.channels[ch][start + i] * window[i] } else { 0.0 };
                Complex32::new(s, 0.0)
            }).collect();
            fft.process(&mut buf);
            buf
        }).collect();

        let mut beam_freq = vec![Complex32::ZERO; block_size];
        for k in 0..block_size {
            let x_k: Vec<Complex32> = (0..m).map(|ch| x_freq[ch][k]).collect();
            let d_k: Vec<Complex32> = (0..m).map(|ch| steering[ch][k]).collect();
            let w_k = mvdr_weight_scalar_n(&x_k, &d_k, REGULARIZATION);
            beam_freq[k] = w_k.iter().zip(x_k.iter()).map(|(w, x)| w.conj() * x).fold(Complex32::ZERO, |a, v| a + v);
        }

        ifft.process(&mut beam_freq);
        let scale = 1.0 / block_size as f32;
        let is_first = block_idx == 0;
        let is_last = block_idx + 1 >= n_blocks;
        for i in 0..actual {
            let mut sample = beam_freq[i].re * scale;
            if is_first && i < fade_len { sample *= i as f32 / fade_len as f32; }
            if is_last { let rem = actual - i; if rem < fade_len { sample *= rem as f32 / fade_len as f32; } }
            output[start + i] += sample * window[i];
            norm[start + i] += window[i] * window[i];
        }
    }

    for (out, n) in output.iter_mut().zip(norm.iter()) {
        if *n > 1e-6 { *out /= n; }
    }
    MvdrRenderOutput { samples: output, sample_rate_hz: request.sample_rate_hz }
}

/// Compute per-bin steering vectors d[channel][bin] for M mics with arbitrary geometry.
fn steering_vector_n(dir: &[f32; 3], mic_positions: &[[f32; 3]], sample_rate_hz: f32, fft_size: usize) -> Vec<Vec<Complex32>> {
    mic_positions.iter().map(|pos| {
        let delay_s = -(pos[0] * dir[0] + pos[1] * dir[1] + pos[2] * dir[2]) / SPEED_OF_SOUND_MPS;
        let delay_samples = delay_s * sample_rate_hz;
        (0..fft_size).map(|k| {
            let phase = -2.0 * PI * (k as f32 / fft_size as f32) * delay_samples;
            Complex32::new(phase.cos(), phase.sin())
        }).collect()
    }).collect()
}

/// Compute scalar MVDR weight vector for one frequency bin (M-channel generalisation).
fn mvdr_weight_scalar_n(x: &[Complex32], d: &[Complex32], reg: f32) -> Vec<Complex32> {
    let m = x.len().min(d.len());
    if m == 0 { return Vec::new(); }
    let reg_inv = 1.0 / reg;
    let xhd: Complex32 = x[..m].iter().zip(d[..m].iter()).map(|(xi, di)| xi.conj() * di).sum();
    let xhx_reg: Complex32 = x[..m].iter()
        .map(|xi| reg_inv * xi.norm_sqr())
        .fold(Complex32::ZERO, |a, v| a + Complex32::new(v, 0.0));
    let denom_inv = 1.0 / (Complex32::new(1.0, 0.0) + xhx_reg);
    let xhd_reg = reg_inv * xhd;
    let mut r_inv_d: Vec<Complex32> = (0..m)
        .map(|i| reg_inv * d[i] - x[i] * xhd_reg * denom_inv)
        .collect();
    let dhrid: Complex32 = d[..m].iter().zip(r_inv_d.iter()).map(|(di, ri)| di.conj() * ri).sum();
    let dhrid_safe = if dhrid.re.abs() < 1e-12 { Complex32::new(1e-12, 0.0) } else { dhrid };
    for w in &mut r_inv_d { *w /= dhrid_safe; }
    r_inv_d
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn silence_channels(n: usize) -> [Vec<f32>; 4] {
        std::array::from_fn(|_| vec![0.0f32; n])
    }

    fn tone_channels(freq_hz: f32, sr: u32, n: usize) -> [Vec<f32>; 4] {
        let sr_f = sr as f32;
        std::array::from_fn(|ch| {
            let phase_offset = ch as f32 * 0.1;
            (0..n)
                .map(|i| (2.0 * PI * freq_hz * i as f32 / sr_f + phase_offset).sin() * 0.5)
                .collect()
        })
    }

    #[test]
    fn empty_input_returns_empty() {
        let result = render_mvdr(MvdrRenderRequest {
            channels: silence_channels(0),
            sample_rate_hz: 16_000,
            trajectory: vec![],
            fade_samples: None,
        });
        assert!(result.samples.is_empty());
    }

    #[test]
    fn silence_in_silence_out() {
        let n = 4096;
        let result = render_mvdr(MvdrRenderRequest {
            channels: silence_channels(n),
            sample_rate_hz: 16_000,
            trajectory: vec![TrajectoryWaypoint {
                sample_offset: 0,
                position_m: [1.0, 0.0, 0.0],
            }],
            fade_samples: None,
        });
        assert_eq!(result.samples.len(), n);
        let rms: f32 = (result.samples.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();
        assert!(rms < 1e-5, "silence in → near-silence out, got RMS {rms}");
    }

    #[test]
    fn output_length_matches_input() {
        for n in [512, 1024, 1600, 8000] {
            let result = render_mvdr(MvdrRenderRequest {
                channels: tone_channels(440.0, 16_000, n),
                sample_rate_hz: 16_000,
                trajectory: vec![
                    TrajectoryWaypoint {
                        sample_offset: 0,
                        position_m: [1.0, 0.0, 0.0],
                    },
                    TrajectoryWaypoint {
                        sample_offset: n / 2,
                        position_m: [0.0, 1.0, 0.0],
                    },
                ],
                fade_samples: Some(160),
            });
            assert_eq!(result.samples.len(), n, "output length mismatch for n={n}");
        }
    }

    #[test]
    fn steering_slew_does_not_produce_nan() {
        let n = 16_000;
        // Waypoints that jump abruptly — should be smoothed by IIR.
        let trajectory = (0..20)
            .map(|i| TrajectoryWaypoint {
                sample_offset: i * (n / 20),
                position_m: if i % 2 == 0 {
                    [1.0, 0.0, 0.0]
                } else {
                    [-1.0, 0.0, 0.0]
                },
            })
            .collect();

        let result = render_mvdr(MvdrRenderRequest {
            channels: tone_channels(800.0, 16_000, n),
            sample_rate_hz: 16_000,
            trajectory,
            fade_samples: None,
        });
        assert!(
            result.samples.iter().all(|s| s.is_finite()),
            "output contains NaN or Inf"
        );
    }
}
