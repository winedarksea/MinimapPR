//! Map anchoring: fit the rigid chirp layout to every phone's GPS fix at
//! once (weighted Kabsch with Huber IRLS). Three unknowns: east, north,
//! rotation.
//!
//! MDS returns the layout only up to reflection, and Kabsch allows proper
//! rotations only, so the fit runs on the layout and on its mirror image
//! and keeps the lower Huber loss. The loss gap gives a chirality
//! confidence (low when the layout is small next to GPS error under
//! canopy). Any direction cue — a DOA of a chirp, a walking track — locks
//! the chirality outright.

/// Rotation + translation, optionally applied after mirroring y → −y.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidTransform2d {
    pub rotation_rad: f64,
    pub translation_m: [f64; 2],
    pub mirrored: bool,
}

impl RigidTransform2d {
    pub fn apply(&self, point: [f64; 2]) -> [f64; 2] {
        let local = if self.mirrored {
            [point[0], -point[1]]
        } else {
            point
        };
        let (sin, cos) = self.rotation_rad.sin_cos();
        [
            cos * local[0] - sin * local[1] + self.translation_m[0],
            sin * local[0] + cos * local[1] + self.translation_m[1],
        ]
    }
}

/// Weighted least-squares rigid fit source → target (closed-form 2D Kabsch).
pub fn weighted_kabsch_2d(
    source: &[[f64; 2]],
    target: &[[f64; 2]],
    weights: &[f64],
    mirrored: bool,
) -> RigidTransform2d {
    let total: f64 = weights.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    let reflect = |p: [f64; 2]| if mirrored { [p[0], -p[1]] } else { p };
    let mut source_centroid = [0.0; 2];
    let mut target_centroid = [0.0; 2];
    for ((s, t), w) in source.iter().zip(target).zip(weights) {
        let s = reflect(*s);
        for axis in 0..2 {
            source_centroid[axis] += w * s[axis] / total;
            target_centroid[axis] += w * t[axis] / total;
        }
    }
    let (mut cross_cos, mut cross_sin) = (0.0, 0.0);
    for ((s, t), w) in source.iter().zip(target).zip(weights) {
        let s = reflect(*s);
        let (sx, sy) = (s[0] - source_centroid[0], s[1] - source_centroid[1]);
        let (tx, ty) = (t[0] - target_centroid[0], t[1] - target_centroid[1]);
        cross_cos += w * (sx * tx + sy * ty);
        cross_sin += w * (sx * ty - sy * tx);
    }
    let rotation_rad = cross_sin.atan2(cross_cos);
    let (sin, cos) = rotation_rad.sin_cos();
    RigidTransform2d {
        rotation_rad,
        translation_m: [
            target_centroid[0] - (cos * source_centroid[0] - sin * source_centroid[1]),
            target_centroid[1] - (sin * source_centroid[0] + cos * source_centroid[1]),
        ],
        mirrored,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpsFix {
    pub east_m: f64,
    pub north_m: f64,
    /// Effective 1σ horizontal error (see `effective_gps_sigma_m`).
    pub sigma_m: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnchorFitConfig {
    pub huber_threshold_sigma: f64,
    /// Fixes beyond this many σ after the fit are flagged (the phone under a hemlock).
    pub outlier_threshold_sigma: f64,
    pub reweighting_iterations: usize,
    /// Reported accuracy is optimistic under trees: σ = accuracy · (1 + k · (1 − quality)).
    pub gnss_quality_inflation: f64,
}

impl Default for AnchorFitConfig {
    fn default() -> Self {
        Self {
            huber_threshold_sigma: 1.5,
            outlier_threshold_sigma: 3.0,
            reweighting_iterations: 20,
            gnss_quality_inflation: 2.0,
        }
    }
}

/// Inflate a reported horizontal accuracy by a 0–1 GNSS quality score
/// (satellites used, C/N0, L5 lock; 1 = clearing with L5).
pub fn effective_gps_sigma_m(
    reported_accuracy_m: f64,
    gnss_quality: f64,
    config: &AnchorFitConfig,
) -> f64 {
    reported_accuracy_m
        * (1.0 + config.gnss_quality_inflation * (1.0 - gnss_quality.clamp(0.0, 1.0)))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChiralityCue {
    Unknown,
    /// A direction measurement settled it.
    Locked {
        mirrored: bool,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnchorFit {
    pub transform: RigidTransform2d,
    /// P(chosen chirality) in [0.5, 1]; 1 when locked by a cue.
    pub chirality_confidence: f64,
    pub huber_loss: f64,
    pub alternative_huber_loss: f64,
    pub normalized_residuals: Vec<Option<f64>>,
    pub outliers: Vec<bool>,
    pub rotation_sigma_rad: f64,
    pub translation_sigma_m: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AnchorFitError {
    LengthMismatch,
    TooFewFixes,
}

/// Minimum fixes for rotation (two give a weak one; see "Couples").
pub const MINIMUM_FIXES_FOR_ANCHOR: usize = 2;

fn huber_loss(normalized: f64, threshold: f64) -> f64 {
    let magnitude = normalized.abs();
    if magnitude <= threshold {
        0.5 * magnitude * magnitude
    } else {
        threshold * magnitude - 0.5 * threshold * threshold
    }
}

struct ChiralityFit {
    transform: RigidTransform2d,
    loss: f64,
    weights: Vec<f64>,
}

fn robust_fit(
    source: &[[f64; 2]],
    target: &[[f64; 2]],
    sigmas: &[f64],
    mirrored: bool,
    config: &AnchorFitConfig,
) -> ChiralityFit {
    let base: Vec<f64> = sigmas.iter().map(|s| 1.0 / s.powi(2)).collect();
    let mut weights = base.clone();
    let mut transform = weighted_kabsch_2d(source, target, &weights, mirrored);
    for _ in 0..config.reweighting_iterations {
        for (index, (s, t)) in source.iter().zip(target).enumerate() {
            let p = transform.apply(*s);
            let normalized = ((p[0] - t[0]).powi(2) + (p[1] - t[1]).powi(2)).sqrt() / sigmas[index];
            let huber = if normalized <= config.huber_threshold_sigma {
                1.0
            } else {
                config.huber_threshold_sigma / normalized
            };
            weights[index] = base[index] * huber;
        }
        transform = weighted_kabsch_2d(source, target, &weights, mirrored);
    }
    // Huber still lets a 6σ fix pull the fit; drop fixes past the outlier
    // threshold outright (redescending final step) when enough remain.
    let outlier_free: Vec<f64> = source
        .iter()
        .zip(target)
        .enumerate()
        .map(|(index, (s, t))| {
            let p = transform.apply(*s);
            let normalized = ((p[0] - t[0]).powi(2) + (p[1] - t[1]).powi(2)).sqrt() / sigmas[index];
            if normalized > config.outlier_threshold_sigma {
                0.0
            } else {
                weights[index]
            }
        })
        .collect();
    if outlier_free.iter().filter(|w| **w > 0.0).count() >= MINIMUM_FIXES_FOR_ANCHOR {
        weights = outlier_free;
        transform = weighted_kabsch_2d(source, target, &weights, mirrored);
    }
    let loss = source
        .iter()
        .zip(target)
        .zip(sigmas)
        .map(|((s, t), sigma)| {
            let p = transform.apply(*s);
            huber_loss(
                ((p[0] - t[0]).powi(2) + (p[1] - t[1]).powi(2)).sqrt() / sigma,
                config.huber_threshold_sigma,
            )
        })
        .sum();
    ChiralityFit {
        transform,
        loss,
        weights,
    }
}

/// Fit the layout (local x, y per node) to the GPS fixes it has.
pub fn fit_layout_to_gps(
    layout_m: &[[f64; 2]],
    fixes: &[Option<GpsFix>],
    cue: ChiralityCue,
    config: &AnchorFitConfig,
) -> Result<AnchorFit, AnchorFitError> {
    if layout_m.len() != fixes.len() {
        return Err(AnchorFitError::LengthMismatch);
    }
    let used: Vec<usize> = (0..fixes.len())
        .filter(|i| fixes[*i].is_some_and(|f| f.sigma_m > 0.0))
        .collect();
    if used.len() < MINIMUM_FIXES_FOR_ANCHOR {
        return Err(AnchorFitError::TooFewFixes);
    }
    let source: Vec<[f64; 2]> = used.iter().map(|i| layout_m[*i]).collect();
    let target: Vec<[f64; 2]> = used
        .iter()
        .map(|i| fixes[*i].map(|f| [f.east_m, f.north_m]).unwrap_or_default())
        .collect();
    let sigmas: Vec<f64> = used
        .iter()
        .map(|i| fixes[*i].map_or(1.0, |f| f.sigma_m))
        .collect();
    let proper = robust_fit(&source, &target, &sigmas, false, config);
    let mirror = robust_fit(&source, &target, &sigmas, true, config);
    let (chosen, alternative, chirality_confidence) = match cue {
        ChiralityCue::Locked { mirrored } => {
            if mirrored {
                (mirror, proper, 1.0)
            } else {
                (proper, mirror, 1.0)
            }
        }
        ChiralityCue::Unknown => {
            let (best, other) = if proper.loss <= mirror.loss {
                (proper, mirror)
            } else {
                (mirror, proper)
            };
            // Huber loss is a negative log-likelihood: P = 1 / (1 + e^{−ΔL}).
            let confidence = 1.0 / (1.0 + (-(other.loss - best.loss)).exp());
            (best, other, confidence)
        }
    };

    let mut normalized_residuals = vec![None; fixes.len()];
    let mut outliers = vec![false; fixes.len()];
    let centroid = {
        let total: f64 = chosen.weights.iter().sum::<f64>().max(f64::MIN_POSITIVE);
        let mut c = [0.0; 2];
        for (s, w) in source.iter().zip(&chosen.weights) {
            c[0] += w * s[0] / total;
            c[1] += w * s[1] / total;
        }
        c
    };
    let mut rotational_information = 0.0;
    for (k, &node) in used.iter().enumerate() {
        let p = chosen.transform.apply(source[k]);
        let normalized =
            ((p[0] - target[k][0]).powi(2) + (p[1] - target[k][1]).powi(2)).sqrt() / sigmas[k];
        normalized_residuals[node] = Some(normalized);
        outliers[node] = normalized > config.outlier_threshold_sigma;
        let lever_squared =
            (source[k][0] - centroid[0]).powi(2) + (source[k][1] - centroid[1]).powi(2);
        rotational_information += chosen.weights[k] * lever_squared;
    }
    let translation_information: f64 = chosen.weights.iter().sum();
    Ok(AnchorFit {
        transform: chosen.transform,
        chirality_confidence,
        huber_loss: chosen.loss,
        alternative_huber_loss: alternative.loss,
        normalized_residuals,
        outliers,
        rotation_sigma_rad: 1.0 / rotational_information.max(f64::MIN_POSITIVE).sqrt(),
        translation_sigma_m: 1.0 / translation_information.max(f64::MIN_POSITIVE).sqrt(),
    })
}

/// Which chirality agrees with a world-frame bearing (radians, CCW from
/// east) measured from layout node `from` towards node `to` (e.g. an array
/// node's DOA of a phone's chirp, or two phones' walking track).
pub fn chirality_from_bearing(
    layout_m: &[[f64; 2]],
    from: usize,
    to: usize,
    world_bearing_rad: f64,
    proper: &RigidTransform2d,
    mirrored: &RigidTransform2d,
) -> ChiralityCue {
    let angular_error = |transform: &RigidTransform2d| {
        let a = transform.apply(layout_m[from]);
        let b = transform.apply(layout_m[to]);
        let bearing = (b[1] - a[1]).atan2(b[0] - a[0]);
        let difference = (bearing - world_bearing_rad).rem_euclid(std::f64::consts::TAU);
        difference.min(std::f64::consts::TAU - difference)
    };
    ChiralityCue::Locked {
        mirrored: angular_error(mirrored) < angular_error(proper),
    }
}

#[cfg(test)]
#[path = "gps_anchor_fit_tests.rs"]
mod tests;
