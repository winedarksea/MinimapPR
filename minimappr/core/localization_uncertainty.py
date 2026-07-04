from __future__ import annotations

import numpy as np


EPSILON = 1e-9


def clamp_covariance_eigenvalues(
    covariance_m2: list[list[float]] | np.ndarray | None,
    *,
    maximum_std_m: float,
) -> list[list[float]] | None:
    """Clamp a covariance matrix so no eigenvalue exceeds ``maximum_std_m**2``.

    Applied at the localization seam to keep track position uncertainty physical
    (no kilometre-scale ellipses) regardless of which solver produced the matrix,
    including the explicit cone covariance that bypasses ``_stabilize_covariance``.
    Returns a plain nested list (the wire/storage form) or ``None`` when the input
    is missing or not a finite square matrix.
    """
    if covariance_m2 is None or maximum_std_m <= 0.0:
        return covariance_m2 if isinstance(covariance_m2, list) or covariance_m2 is None else None
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or covariance.size == 0:
        return None
    if not np.all(np.isfinite(covariance)):
        return None

    ceiling_m2 = float(maximum_std_m) ** 2
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)):
        return None
    if np.all(eigenvalues <= ceiling_m2):
        return covariance.tolist()
    clamped = np.minimum(eigenvalues, ceiling_m2)
    capped = eigenvectors @ np.diag(clamped) @ eigenvectors.T
    capped = 0.5 * (capped + capped.T)
    return capped.tolist()


def clamp_covariance_eigenvalues_range_proportional(
    covariance_m2: list[list[float]] | np.ndarray | None,
    *,
    range_m: float,
    std_factor: float,
    floor_std_m: float,
    ceiling_std_m: float,
) -> tuple[list[list[float]] | None, bool]:
    """Clamp covariance eigenvalues to a range-proportional ceiling.

    A single fixed ``maximum_std_m`` ceiling is wrong across a 1 m–1000 m envelope:
    it is absurdly loose up close and clips honest uncertainty far away. Instead the
    effective per-axis std ceiling scales with the distance from the contributing
    sensors::

        effective_ceiling_std = min(max(std_factor * range_m, floor_std_m), ceiling_std_m)

    ``ceiling_std_m`` is the absolute hard cap (kept for physicality / storage sanity);
    ``floor_std_m`` keeps a sensible minimum near the array. When ``std_factor <= 0``
    the range term is disabled and the legacy fixed ``ceiling_std_m`` clamp is used
    (backward compatible).

    Returns ``(clamped_nested_list_or_None, was_range_capped)`` where the flag is True
    only when the range-proportional ceiling (below the absolute ceiling) actually
    reduced an eigenvalue — used to drive the ``localization_covariance_range_capped``
    metric. Mirrors :func:`clamp_covariance_eigenvalues` for the None/degenerate paths.
    """
    if covariance_m2 is None or ceiling_std_m <= 0.0:
        passthrough = covariance_m2 if isinstance(covariance_m2, list) or covariance_m2 is None else None
        return passthrough, False
    if std_factor > 0.0:
        effective_ceiling_std_m = min(
            max(float(std_factor) * max(float(range_m), 0.0), float(floor_std_m)),
            float(ceiling_std_m),
        )
    else:
        effective_ceiling_std_m = float(ceiling_std_m)
    range_capped = effective_ceiling_std_m < float(ceiling_std_m) - EPSILON

    covariance = np.asarray(covariance_m2, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or covariance.size == 0:
        return None, False
    if not np.all(np.isfinite(covariance)):
        return None, False

    ceiling_m2 = effective_ceiling_std_m ** 2
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)):
        return None, False
    if np.all(eigenvalues <= ceiling_m2):
        return covariance.tolist(), False
    clamped = np.minimum(eigenvalues, ceiling_m2)
    capped = eigenvectors @ np.diag(clamped) @ eigenvectors.T
    capped = 0.5 * (capped + capped.T)
    # Only report a range-specific cap when the range term (not the absolute ceiling)
    # is what bit — i.e. an eigenvalue exceeded the range-proportional ceiling while
    # the ceiling itself is below the absolute cap.
    return capped.tolist(), range_capped


def _stabilize_covariance(
    covariance_m2: np.ndarray,
    *,
    minimum_std_m: float,
) -> np.ndarray | None:
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or covariance.size == 0:
        return None
    if not np.all(np.isfinite(covariance)):
        return None

    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)):
        return None

    variance_floor_m2 = max(float(minimum_std_m), 1.0e-3) ** 2
    clamped = np.maximum(eigenvalues, variance_floor_m2)
    stabilized = eigenvectors @ np.diag(clamped) @ eigenvectors.T
    return 0.5 * (stabilized + stabilized.T)


def covariance_from_jacobian(
    jacobian: np.ndarray,
    residual_rms_seconds: float,
    *,
    minimum_time_std_s: float,
    minimum_std_m: float,
) -> np.ndarray | None:
    jacobian_matrix = np.asarray(jacobian, dtype=np.float64)
    if jacobian_matrix.ndim != 2 or jacobian_matrix.size == 0:
        return None
    if not np.all(np.isfinite(jacobian_matrix)):
        return None

    information = jacobian_matrix.T @ jacobian_matrix
    if not np.all(np.isfinite(information)):
        return None

    covariance = np.linalg.pinv(information)
    if not np.all(np.isfinite(covariance)):
        return None

    time_std_s = max(float(residual_rms_seconds), float(minimum_time_std_s), 1.0e-6)
    covariance = covariance * (time_std_s * time_std_s)
    return _stabilize_covariance(covariance, minimum_std_m=minimum_std_m)


def embed_planar_covariance_m2(
    planar_covariance_m2: np.ndarray | None,
    *,
    vertical_std_m: float,
    minimum_std_m: float,
) -> np.ndarray | None:
    if planar_covariance_m2 is None:
        return None
    planar = np.asarray(planar_covariance_m2, dtype=np.float64)
    if planar.shape != (2, 2) or not np.all(np.isfinite(planar)):
        return None

    full = np.zeros((3, 3), dtype=np.float64)
    full[:2, :2] = planar
    full[2, 2] = max(float(vertical_std_m), 1.0e-3) ** 2
    return _stabilize_covariance(full, minimum_std_m=minimum_std_m)


def directional_covariance_m2(
    direction: np.ndarray,
    *,
    lateral_std_m: float,
    radial_std_m: float,
    minimum_std_m: float,
) -> np.ndarray | None:
    direction_vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    if direction_vector.size != 3:
        return None

    lateral_variance_m2 = max(float(lateral_std_m), 1.0e-3) ** 2
    radial_variance_m2 = max(float(radial_std_m), float(lateral_std_m), 1.0e-3) ** 2
    norm = float(np.linalg.norm(direction_vector))
    if norm < EPSILON:
        return _stabilize_covariance(
            np.eye(3, dtype=np.float64) * lateral_variance_m2,
            minimum_std_m=minimum_std_m,
        )

    radial_axis = direction_vector / norm
    helper_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(radial_axis, helper_axis))) > 0.9:
        helper_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

    lateral_axis_a = np.cross(radial_axis, helper_axis)
    lateral_axis_a /= np.linalg.norm(lateral_axis_a) + EPSILON
    lateral_axis_b = np.cross(radial_axis, lateral_axis_a)
    lateral_axis_b /= np.linalg.norm(lateral_axis_b) + EPSILON

    covariance = (
        radial_variance_m2 * np.outer(radial_axis, radial_axis)
        + lateral_variance_m2 * np.outer(lateral_axis_a, lateral_axis_a)
        + lateral_variance_m2 * np.outer(lateral_axis_b, lateral_axis_b)
    )
    return _stabilize_covariance(covariance, minimum_std_m=minimum_std_m)


def range_observability_from_covariance(
    covariance_m2: np.ndarray | None,
    direction: np.ndarray | None = None,
) -> float | None:
    if covariance_m2 is None:
        return None
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or covariance.size == 0:
        return None
    if not np.all(np.isfinite(covariance)):
        return None

    direction_vector = None if direction is None else np.asarray(direction, dtype=np.float64).reshape(-1)
    if direction_vector is not None and direction_vector.size == covariance.shape[0]:
        norm = float(np.linalg.norm(direction_vector))
        if norm >= EPSILON:
            radial_axis = direction_vector / norm
            radial_variance_m2 = float(radial_axis @ covariance @ radial_axis)
            lateral_variance_m2 = max(
                (float(np.trace(covariance)) - radial_variance_m2) / max(covariance.shape[0] - 1, 1),
                EPSILON,
            )
            ratio = lateral_variance_m2 / max(radial_variance_m2, lateral_variance_m2, EPSILON)
            return float(np.clip(np.sqrt(ratio), 0.0, 1.0))

    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.all(np.isfinite(eigenvalues)):
        return None
    min_variance_m2 = max(float(np.min(eigenvalues)), EPSILON)
    max_variance_m2 = max(float(np.max(eigenvalues)), min_variance_m2)
    return float(np.clip(np.sqrt(min_variance_m2 / max_variance_m2), 0.0, 1.0))


def apply_frequency_covariance_scaling(
    covariance_m2: np.ndarray,
    bearing_unit_vec: np.ndarray,
    dominant_frequency_hz: float,
    alias_cutoff_hz: float,
    *,
    frequency_floor: float = 0.25,
) -> np.ndarray:
    """Inflate lateral covariance when signal frequency is well below alias cutoff.

    Angular resolution degrades when the dominant signal frequency is much lower
    than the array's spatial aliasing cutoff (c / 2·max_baseline). This function
    widens the covariance along directions perpendicular to the bearing by the
    factor (alias_cutoff_hz / dominant_frequency_hz)², making the uncertainty
    representation more honest for low-frequency signals.

    At or above alias_cutoff_hz the covariance is returned unchanged.
    The inflation is capped at 1/frequency_floor² to avoid degenerate blowup.
    """
    if alias_cutoff_hz <= 0.0 or dominant_frequency_hz <= 0.0:
        return covariance_m2
    lateral_scale = float(np.clip(dominant_frequency_hz / alias_cutoff_hz, frequency_floor, 1.0))
    if lateral_scale >= 1.0 - 1.0e-6:
        return covariance_m2
    e = np.asarray(bearing_unit_vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(e))
    if norm < EPSILON:
        return covariance_m2
    e = e / norm
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    if not np.all(np.isfinite(covariance)):
        return covariance_m2
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)):
        return covariance_m2
    inflation = 1.0 / (lateral_scale ** 2)  # > 1: wider
    scaled_eigenvalues = np.empty_like(eigenvalues)
    for i, v in enumerate(eigenvectors.T):
        # Fraction of this eigenvector that lies in the lateral plane
        lateral_fraction = 1.0 - float(np.dot(e, v)) ** 2
        scale = 1.0 + (inflation - 1.0) * lateral_fraction
        scaled_eigenvalues[i] = eigenvalues[i] * max(scale, 1.0)
    result = eigenvectors @ np.diag(scaled_eigenvalues) @ eigenvectors.T
    return 0.5 * (result + result.T)


def covariance_to_nested_list(covariance_m2: np.ndarray | None) -> list[list[float]] | None:
    if covariance_m2 is None:
        return None
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not np.all(np.isfinite(covariance)):
        return None
    return covariance.astype(float).tolist()