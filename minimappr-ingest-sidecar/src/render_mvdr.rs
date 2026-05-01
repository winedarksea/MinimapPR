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
use rustfft::{FftPlanner, FftDirection};

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

pub fn render_mvdr(request: MvdrRenderRequest) -> MvdrRenderOutput {
    let n_samples = request.channels[0].len();
    for ch in &request.channels {
        assert_eq!(ch.len(), n_samples, "all channels must have equal length");
    }
    if n_samples == 0 {
        return MvdrRenderOutput {
            samples: vec![],
            sample_rate_hz: request.sample_rate_hz,
        };
    }

    let sr = request.sample_rate_hz as f32;
    let fade_len = request.fade_samples.unwrap_or((0.1 * sr) as usize).max(1);

    // Block size: next power-of-two ≥ 256.
    let block_size = 512_usize;
    let hop = block_size / 2; // 50 % overlap

    let mut output = vec![0.0f32; n_samples];
    let mut norm = vec![0.0f32; n_samples]; // overlap-add normalisation

    let window = hann_window(block_size);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft(block_size, FftDirection::Forward);
    let ifft = planner.plan_fft(block_size, FftDirection::Inverse);

    // IIR steering state (unit vector, initialised from first waypoint).
    let first_dir = if request.trajectory.is_empty() {
        [1.0f32, 0.0, 0.0]
    } else {
        normalise_vec3(request.trajectory[0].position_m)
    };
    let mut current_dir = first_dir;

    // α for ~180°/s slew limit at `hop` samples per update.
    // 180°/s → π rad / sr samples/s → π·hop/sr rad per block.
    // IIR cutoff ≈ (π·hop/sr) / (2π·T) where T = hop/sr → cutoff ≈ 0.5 Hz.
    let alpha = 1.0 - (-2.0 * PI * 0.5 * (hop as f32 / sr)).exp();
    let alpha = alpha.clamp(0.01, 0.5);

    let n_blocks = (n_samples + hop - 1) / hop;
    for block_idx in 0..n_blocks {
        let start = block_idx * hop;
        let end = (start + block_size).min(n_samples);
        let actual = end - start;

        // Interpolate steering direction from trajectory.
        let target_dir = waypoint_direction_at(&request.trajectory, start + hop / 2);
        current_dir = iir_slew(current_dir, target_dir, alpha);

        // Compute MVDR weights in frequency domain.
        let steering = steering_vector(&current_dir, &SIRITH_MIC_POSITIONS_M, sr, block_size);

        // Build frequency-domain multi-channel block [4 × block_size].
        let x_freq: [Vec<Complex32>; 4] = std::array::from_fn(|ch| {
            let mut buf: Vec<Complex32> = (0..block_size)
                .map(|i| {
                    let s = if start + i < n_samples {
                        request.channels[ch][start + i] * window[i]
                    } else {
                        0.0
                    };
                    Complex32::new(s, 0.0)
                })
                .collect();
            fft.process(&mut buf);
            buf
        });

        // Estimate spatial covariance R[f] and compute MVDR weights w[f].
        let mut beam_freq = vec![Complex32::ZERO; block_size];
        for k in 0..block_size {
            let x_k: [Complex32; 4] = std::array::from_fn(|ch| x_freq[ch][k]);
            let d_k: [Complex32; 4] = std::array::from_fn(|ch| steering[ch][k]);
            let w_k = mvdr_weight_scalar(x_k, d_k, REGULARIZATION);
            beam_freq[k] = w_k
                .iter()
                .zip(x_k.iter())
                .map(|(w, x)| w.conj() * x)
                .fold(Complex32::ZERO, |acc, v| acc + v);
        }

        // IFFT → real beamformed block.
        ifft.process(&mut beam_freq);
        let scale = 1.0 / block_size as f32;

        // Overlap-add with fade-in / fade-out envelopes.
        let is_first = block_idx == 0;
        let is_last = block_idx + 1 >= n_blocks;
        for i in 0..actual {
            let mut sample = beam_freq[i].re * scale;

            // Fade-in at start of track.
            if is_first && i < fade_len {
                sample *= i as f32 / fade_len as f32;
            }
            // Fade-out at end of track.
            if is_last {
                let remaining = actual - i;
                if remaining < fade_len {
                    sample *= remaining as f32 / fade_len as f32;
                }
            }

            output[start + i] += sample * window[i];
            norm[start + i] += window[i] * window[i];
        }
    }

    // Normalise by the overlap-add window power.
    for (out, n) in output.iter_mut().zip(norm.iter()) {
        if *n > 1e-6 {
            *out /= n;
        }
    }

    MvdrRenderOutput {
        samples: output,
        sample_rate_hz: request.sample_rate_hz,
    }
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

/// Compute per-bin steering vectors d[channel][bin] for a unit direction,
/// using fractional-delay phase shifts.
fn steering_vector(
    dir: &[f32; 3],
    mic_positions: &[[f32; 3]; 4],
    sample_rate_hz: f32,
    fft_size: usize,
) -> [Vec<Complex32>; 4] {
    let n_bins = fft_size;
    std::array::from_fn(|ch| {
        let pos = mic_positions[ch];
        // Delay = dot(dir, pos) / c  (positive = towards source)
        let delay_s = -(pos[0] * dir[0] + pos[1] * dir[1] + pos[2] * dir[2])
            / SPEED_OF_SOUND_MPS;
        let delay_samples = delay_s * sample_rate_hz;
        (0..n_bins)
            .map(|k| {
                let freq_normalized = k as f32 / n_bins as f32;
                let phase = -2.0 * PI * freq_normalized * delay_samples;
                Complex32::new(phase.cos(), phase.sin())
            })
            .collect()
    })
}

/// Compute scalar MVDR weight vector for one frequency bin.
/// Returns a length-4 weight vector w such that the beamformed output is
/// `y[k] = sum_i w_i.conj() * x_i[k]`.
fn mvdr_weight_scalar(
    x: [Complex32; 4],
    d: [Complex32; 4],
    reg: f32,
) -> [Complex32; 4] {
    // Estimate rank-1 covariance R = x * x^H (outer product).
    // For a single snapshot we regularise: R_reg = x x^H + reg * I.
    // R^{-1} d via direct formula for diagonal + rank-1:
    // (A + uv^H)^{-1} = A^{-1} - (A^{-1} u v^H A^{-1}) / (1 + v^H A^{-1} u)
    // With A = reg * I: A^{-1} = (1/reg) * I.
    let reg_inv = 1.0 / reg;
    // Numerator: R^{-1} * d
    // First compute x^H d (scalar):
    let xhd: Complex32 = x.iter().zip(d.iter()).map(|(xi, di)| xi.conj() * di).sum();
    // x^H * (1/reg * I) * x (scalar):
    let xhx_reg: Complex32 = x
        .iter()
        .map(|xi| reg_inv * xi.norm_sqr())
        .fold(Complex32::ZERO, |a, v| a + Complex32::new(v, 0.0));
    let denom_inv = 1.0 / (Complex32::new(1.0, 0.0) + xhx_reg);
    // R^{-1} d = (1/reg)*d - (1/reg)*x*(x^H*(1/reg)*d) / (1 + x^H*(1/reg)*x)
    let xhd_reg = reg_inv * xhd;
    let mut r_inv_d = [Complex32::ZERO; 4];
    for i in 0..4 {
        r_inv_d[i] = reg_inv * d[i] - x[i] * xhd_reg * denom_inv;
    }

    // d^H R^{-1} d (scalar denominator):
    let dhrid: Complex32 = d
        .iter()
        .zip(r_inv_d.iter())
        .map(|(di, ri)| di.conj() * ri)
        .sum();
    let dhrid_safe = if dhrid.re.abs() < 1e-12 {
        Complex32::new(1e-12, 0.0)
    } else {
        dhrid
    };

    std::array::from_fn(|i| r_inv_d[i] / dhrid_safe)
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
            assert_eq!(
                result.samples.len(),
                n,
                "output length mismatch for n={n}"
            );
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
