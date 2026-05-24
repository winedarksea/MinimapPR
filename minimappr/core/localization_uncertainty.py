from __future__ import annotations

import numpy as np


EPSILON = 1e-9


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


def covariance_to_nested_list(covariance_m2: np.ndarray | None) -> list[list[float]] | None:
    if covariance_m2 is None:
        return None
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not np.all(np.isfinite(covariance)):
        return None
    return covariance.astype(float).tolist()