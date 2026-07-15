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

/// Smallest non-degenerate pairwise mic spacing (metres). Mirrors the Python
/// `min_pair_spacing_m`. Where processing is per mic-pair, the closest pair sets
/// the highest alias-free frequency (planar's 25 mm pairs → ~6.9 kHz vs the
/// 50 mm diagonal's 3.43 kHz). Returns 0 for <2 mics / all-coincident.
#[allow(dead_code)] // retained for the geometry contract and future pairwise render paths
pub fn min_pair_spacing_m(positions: &[[f32; 3]]) -> f32 {
    let mut min_spacing_m = f32::INFINITY;
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let d = norm3(sub3(positions[i], positions[j]));
            if d > 1e-6 {
                min_spacing_m = min_spacing_m.min(d);
            }
        }
    }
    if min_spacing_m.is_infinite() {
        0.0
    } else {
        min_spacing_m
    }
}

/// Per-pair spatial-alias cutoff c / (2·min_pair_spacing). Opt-in counterpart to
/// [`alias_cutoff_from_positions`] (which uses the widest baseline); keeps the
/// max-baseline default for tetra unless a caller explicitly wants the narrower
/// planar pairs. Degenerate geometry falls back to the default 50 mm baseline.
#[allow(dead_code)] // retained for the geometry contract and future pairwise render paths
pub fn alias_cutoff_min_pair_from_positions(positions: &[[f32; 3]], sound_speed_mps: f32) -> f32 {
    let spacing_m = min_pair_spacing_m(positions);
    if spacing_m < 1e-6 {
        sound_speed_mps / (2.0 * 0.05)
    } else {
        sound_speed_mps / (2.0 * spacing_m)
    }
}

/// Out-of-plane extent (metres): smallest principal-axis spread of the
/// centroid-referenced positions. ~0 for a coplanar array (planar node),
/// non-trivial for a 3D array (tetra). Mirrors the Python
/// `array_out_of_plane_extent_m`.
#[allow(dead_code)] // retained for the geometry contract and future array validation paths
pub fn array_out_of_plane_extent_m(positions: &[[f32; 3]]) -> f32 {
    let n = positions.len();
    if n < 3 {
        return 0.0;
    }
    let mut centroid = [0.0_f64; 3];
    for p in positions {
        centroid[0] += p[0] as f64;
        centroid[1] += p[1] as f64;
        centroid[2] += p[2] as f64;
    }
    for c in &mut centroid {
        *c /= n as f64;
    }
    // Covariance matrix of the centered points; its smallest eigenvalue's sqrt
    // is proportional to the thinnest-axis spread. Full SVD is overkill for 3x3,
    // so use the covariance eigenvalues via a symmetric 3x3 solve.
    let mut cov = [[0.0_f64; 3]; 3];
    for p in positions {
        let d = [
            p[0] as f64 - centroid[0],
            p[1] as f64 - centroid[1],
            p[2] as f64 - centroid[2],
        ];
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += d[r] * d[c];
            }
        }
    }
    let inv_n = 1.0 / n as f64;
    for r in 0..3 {
        for c in 0..3 {
            cov[r][c] *= inv_n;
        }
    }
    let eigs = symmetric_3x3_eigenvalues_desc(cov);
    let min_eig = eigs[2].max(0.0);
    (min_eig.sqrt() * (n as f64).sqrt()) as f32
}

/// True if all mics lie within `tolerance_m` of a common plane (planar node).
#[allow(dead_code)] // retained for the geometry contract and future array validation paths
pub fn is_coplanar(positions: &[[f32; 3]], tolerance_m: f32) -> bool {
    array_out_of_plane_extent_m(positions) <= tolerance_m
}

/// True if the array spans at least two spatial dimensions (i.e. the mics are
/// not all collinear) with the second principal-axis spread exceeding
/// `tolerance_m`. Spatial localization (2-D DOA) needs at least this; a single
/// sensor or a collinear line of mics cannot resolve an unambiguous bearing.
pub fn array_spans_at_least_2d(positions: &[[f32; 3]], tolerance_m: f32) -> bool {
    let n = positions.len();
    if n < 3 {
        return false;
    }
    let mut centroid = [0.0_f64; 3];
    for p in positions {
        for k in 0..3 {
            centroid[k] += p[k] as f64;
        }
    }
    for c in &mut centroid {
        *c /= n as f64;
    }
    let mut cov = [[0.0_f64; 3]; 3];
    for p in positions {
        let d = [
            p[0] as f64 - centroid[0],
            p[1] as f64 - centroid[1],
            p[2] as f64 - centroid[2],
        ];
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += d[r] * d[c];
            }
        }
    }
    let inv_n = 1.0 / n as f64;
    for r in 0..3 {
        for c in 0..3 {
            cov[r][c] *= inv_n;
        }
    }
    // Second-largest eigenvalue's sqrt is the spread along the second principal
    // axis; > tol means the array is not collinear.
    let second = symmetric_3x3_eigenvalues_desc(cov)[1].max(0.0);
    (second.sqrt() * (n as f64).sqrt()) > tolerance_m as f64
}

/// Eigenvalues of a symmetric 3x3 matrix (descending) via the closed-form trig
/// method (Smith 1961). Robust enough for coplanarity / collinearity tests.
fn symmetric_3x3_eigenvalues_desc(a: [[f64; 3]; 3]) -> [f64; 3] {
    let p1 = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    let q = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
    if p1 <= 1e-30 {
        // Diagonal matrix: eigenvalues are the diagonal entries.
        let mut d = [a[0][0], a[1][1], a[2][2]];
        d.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        return d;
    }
    let p2 = (a[0][0] - q).powi(2) + (a[1][1] - q).powi(2) + (a[2][2] - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    let mut b = a;
    for i in 0..3 {
        b[i][i] -= q;
    }
    let det_b = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]);
    let r = (det_b / (2.0 * p.powi(3))).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    let eig1 = q + 2.0 * p * phi.cos();
    let eig3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    let eig2 = 3.0 * q - eig1 - eig3; // trace invariant
    [eig1, eig2, eig3]
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

    fn planar_positions() -> [[f32; 3]; 5] {
        let r = 0.025_f32 * std::f32::consts::FRAC_1_SQRT_2;
        [
            [r, r, 0.0],
            [-r, r, 0.0],
            [-r, -r, 0.0],
            [r, -r, 0.0],
            [0.0, 0.0, 0.0],
        ]
    }

    #[test]
    fn planar_min_pair_cutoff_raises_usable_band() {
        let p = planar_positions();
        // Max-baseline (50 mm diagonal) matches tetra-style ~3.43 kHz.
        let max_baseline = alias_cutoff_from_positions(&p, 343.2);
        assert!((max_baseline - 3432.0).abs() < 20.0, "got {max_baseline}");
        // Min-pair (25 mm corner-to-center) roughly doubles it.
        let min_pair = alias_cutoff_min_pair_from_positions(&p, 343.2);
        assert!((min_pair - 6864.0).abs() < 40.0, "got {min_pair}");
        assert!((min_pair_spacing_m(&p) - 0.025).abs() < 1e-4);
    }

    #[test]
    fn planar_is_coplanar_tetra_is_not() {
        assert!(is_coplanar(&planar_positions(), 1e-3));
        let sirith = [
            [0.0_f32, 0.050, 0.0],
            [0.0433, 0.025, 0.0],
            [0.0, 0.0, 0.0],
            [0.02165, 0.025, 0.04082],
        ];
        assert!(!is_coplanar(&sirith, 1e-3));
        assert!(array_out_of_plane_extent_m(&planar_positions()) < 1e-4);
        assert!(array_out_of_plane_extent_m(&sirith) > 1e-2);
    }

    #[test]
    fn degenerate_geometry_uses_safe_alias_cutoff_fallbacks() {
        let coincident = [[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0]];

        assert_eq!(min_pair_spacing_m(&coincident), 0.0);
        assert_eq!(array_out_of_plane_extent_m(&coincident), 0.0);
        assert!(is_coplanar(&coincident, 0.0));
        assert_eq!(
            alias_cutoff_min_pair_from_positions(&coincident, SPEED_OF_SOUND_MPS),
            SPEED_OF_SOUND_MPS / (2.0 * 0.05)
        );
    }

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
