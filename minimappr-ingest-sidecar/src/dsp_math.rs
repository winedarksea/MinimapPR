//! Shared DSP helpers used by the live band-split render, ambisonics, and the
//! offline MVDR renderer. Canonical definitions per BEAMFORMED_RENDER_CONTRACT.md;
//! Python mirrors live in `minimappr/core/beamforming.py` and
//! `minimappr/spatial_audio/geometry.py`.

pub const SPEED_OF_SOUND_MPS: f32 = 343.2;
#[allow(dead_code)]
pub const SPEED_OF_SOUND_MPS_F64: f64 = 343.2;

#[allow(dead_code)]
pub fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn norm3(a: [f32; 3]) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

pub fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[allow(dead_code)] // consumed by the render_mvdr migration (Phase 7)
pub fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1_usize;
    while p < n {
        p <<= 1;
    }
    p
}

#[allow(dead_code)] // consumed by the render_mvdr migration (Phase 7)
pub fn hann_window(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / (n - 1) as f32;
            x.sin() * x.sin()
        })
        .collect()
}

#[allow(dead_code)] // consumed by the render_mvdr migration (Phase 7)
pub fn sample_at_fractional(samples: &[f32], index: f32) -> f32 {
    if index <= 0.0 {
        return samples.first().copied().unwrap_or(0.0);
    }
    let lower = index.floor() as usize;
    let upper = lower + 1;
    if upper >= samples.len() {
        return samples.last().copied().unwrap_or(0.0);
    }
    let frac = index - lower as f32;
    samples[lower] * (1.0 - frac) + samples[upper] * frac
}

/// Spatial-aliasing cutoff c / (2·max_baseline). Degenerate geometry falls back
/// to the default Sirith 50 mm baseline.
pub fn alias_cutoff_from_positions(positions: &[[f32; 3]], sound_speed_mps: f32) -> f32 {
    let mut max_baseline_m = 0.0_f32;
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            max_baseline_m = max_baseline_m.max(norm3(sub3(positions[i], positions[j])));
        }
    }
    if max_baseline_m < 1e-6 {
        sound_speed_mps / (2.0 * 0.05)
    } else {
        sound_speed_mps / (2.0 * max_baseline_m)
    }
}

/// f64 variant used by the ambisonics path (spatial_audio.rs).
pub fn alias_cutoff_from_positions_f64(positions: &[[f64; 3]], sound_speed_mps: f64) -> f64 {
    let mut max_baseline_m = 0.0_f64;
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let d = [
                positions[i][0] - positions[j][0],
                positions[i][1] - positions[j][1],
                positions[i][2] - positions[j][2],
            ];
            max_baseline_m = max_baseline_m.max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
        }
    }
    if max_baseline_m < 1e-6 {
        sound_speed_mps / (2.0 * 0.05)
    } else {
        sound_speed_mps / (2.0 * max_baseline_m)
    }
}

/// Near-field point-source steering delays: τ_m = |p_m − p_steer| / c with the
/// minimum subtracted (closest sensor has delay 0). Mirrors the Python
/// `_steering_delays_s` (core/beamforming.py).
pub fn steering_delays_s(
    mic_positions_m: &[[f32; 3]],
    steer_position_m: [f32; 3],
    sound_speed_mps: f32,
) -> Vec<f32> {
    let c = sound_speed_mps.max(1.0);
    let delays: Vec<f32> = mic_positions_m
        .iter()
        .map(|p| norm3(sub3(*p, steer_position_m)) / c)
        .collect();
    let min = delays.iter().copied().fold(f32::INFINITY, f32::min);
    delays.iter().map(|d| d - min).collect()
}

/// Steered-band weight w(f) ∈ [0, 1] with raised-cosine crossovers; the high
/// ramp is centered at the alias cutoff (contract §3). Mirrors the Python
/// `raised_cosine_band_weights`.
pub fn raised_cosine_band_weight(
    freq_hz: f32,
    low_center_hz: f32,
    low_width_hz: f32,
    high_center_hz: f32,
    high_width_hz: f32,
) -> f32 {
    let mut w = 1.0_f32;

    let lo_start = low_center_hz - low_width_hz / 2.0;
    let lo_end = low_center_hz + low_width_hz / 2.0;
    if lo_end > lo_start {
        let x = ((freq_hz - lo_start) / (lo_end - lo_start)).clamp(0.0, 1.0);
        w *= 0.5 * (1.0 - (std::f32::consts::PI * x).cos());
    } else if freq_hz < low_center_hz {
        w = 0.0;
    }

    let hi_start = high_center_hz - high_width_hz / 2.0;
    let hi_end = high_center_hz + high_width_hz / 2.0;
    if hi_end > hi_start {
        let x = ((freq_hz - hi_start) / (hi_end - hi_start)).clamp(0.0, 1.0);
        w *= 0.5 * (1.0 + (std::f32::consts::PI * x).cos());
    } else if freq_hz > high_center_hz {
        w = 0.0;
    }

    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tetra_alias_cutoff_matches_python() {
        let sirith = [
            [0.0, 0.050, 0.0],
            [0.0433, 0.025, 0.0],
            [0.0, 0.0, 0.0],
            [0.02165, 0.025, 0.04082],
        ];
        // Max baseline of the real Sirith geometry is MK1–MK4 ≈ 52.5 mm
        // (slightly over the 50 mm nominal edge), giving ≈ 3266 Hz.
        let cutoff = alias_cutoff_from_positions(&sirith, SPEED_OF_SOUND_MPS);
        assert!((cutoff - 3266.0).abs() < 5.0, "cutoff {cutoff}");
    }

    #[test]
    fn steering_delays_min_subtracted() {
        let mics = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]];
        let delays = steering_delays_s(&mics, [10.0, 0.0, 0.0], SPEED_OF_SOUND_MPS);
        assert!(delays[1].abs() < 1e-9); // closest mic
        assert!((delays[0] - 0.1 / 343.2).abs() < 1e-6);
    }

    #[test]
    fn band_weight_shape() {
        let w_mid = raised_cosine_band_weight(1500.0, 100.0, 100.0, 3432.0, 515.0);
        let w_cutoff = raised_cosine_band_weight(3432.0, 100.0, 100.0, 3432.0, 515.0);
        let w_dc = raised_cosine_band_weight(0.0, 100.0, 100.0, 3432.0, 515.0);
        let w_hi = raised_cosine_band_weight(8000.0, 100.0, 100.0, 3432.0, 515.0);
        assert!(w_mid > 0.99);
        assert!((w_cutoff - 0.5).abs() < 0.01);
        assert!(w_dc < 1e-6);
        assert!(w_hi < 1e-6);
    }

    #[test]
    fn hann_window_endpoints_zero() {
        let w = hann_window(8);
        assert!(w[0].abs() < 1e-7);
        assert!(w[7].abs() < 1e-7);
        assert!(w[3] > 0.9);
    }
}
