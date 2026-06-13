"""Unbounded one-dimensional range refinement for TDOA bearing estimates."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from minimappr.core.range_projection import RANGE_ASYMPTOTIC, RANGE_REFINED


EPSILON = 1.0e-12
MAX_RADIAL_SEARCH_EXPANSIONS = 40
RADIAL_SEARCH_EXPANSION_FACTOR = 2.0


@dataclass(frozen=True, slots=True)
class RadialRefinement:
    position_m: np.ndarray
    radial_std_m: float
    projection_mode: str
    fit_residual_rms_seconds: float


def radial_standard_deviation(
    *,
    position_m: np.ndarray,
    centroid_m: np.ndarray,
    jacobian: np.ndarray,
    time_std_s: float,
) -> float:
    radial_offset = position_m - centroid_m
    radial_norm = float(np.linalg.norm(radial_offset))
    if radial_norm < EPSILON:
        return float("inf")
    radial_axis = radial_offset / radial_norm
    radial_information = float(np.linalg.norm(jacobian @ radial_axis))
    if radial_information < EPSILON:
        return float("inf")
    return time_std_s / radial_information


def adaptive_radial_refinement(
    *,
    centroid_m: np.ndarray,
    direction: np.ndarray,
    initial_radius_m: float,
    minimum_radius_m: float,
    residual_function: Callable[[np.ndarray], np.ndarray],
    effective_time_std_s: float,
    asymptotic_radial_std_fraction: float | None = None,
) -> RadialRefinement:
    normalized_direction = direction / max(float(np.linalg.norm(direction)), EPSILON)

    def squared_cost_for_log_radius(log_radius: float) -> float:
        radius_m = math.exp(float(log_radius))
        residual = residual_function(centroid_m + normalized_direction * radius_m)
        if not np.all(np.isfinite(residual)):
            return float("inf")
        return float(np.dot(residual, residual))

    lower_radius_m = max(minimum_radius_m, EPSILON)
    center_radius_m = max(initial_radius_m, lower_radius_m * RADIAL_SEARCH_EXPANSION_FACTOR)
    lower_log_radius = math.log(lower_radius_m)
    center_log_radius = math.log(center_radius_m)
    lower_cost = squared_cost_for_log_radius(lower_log_radius)
    center_cost = squared_cost_for_log_radius(center_log_radius)
    upper_radius_m = center_radius_m * RADIAL_SEARCH_EXPANSION_FACTOR
    upper_log_radius = math.log(upper_radius_m)
    upper_cost = squared_cost_for_log_radius(upper_log_radius)

    expansion_count = 0
    while (
        np.isfinite(upper_cost)
        and upper_cost < center_cost
        and expansion_count < MAX_RADIAL_SEARCH_EXPANSIONS
    ):
        lower_log_radius, lower_cost = center_log_radius, center_cost
        center_log_radius, center_cost = upper_log_radius, upper_cost
        upper_radius_m *= RADIAL_SEARCH_EXPANSION_FACTOR
        if not np.isfinite(upper_radius_m):
            break
        upper_log_radius = math.log(upper_radius_m)
        upper_cost = squared_cost_for_log_radius(upper_log_radius)
        expansion_count += 1

    has_finite_interior_minimum = (
        np.isfinite(lower_cost)
        and np.isfinite(center_cost)
        and np.isfinite(upper_cost)
        and center_cost <= lower_cost
        and center_cost <= upper_cost
    )
    if has_finite_interior_minimum:
        optimized = minimize_scalar(
            squared_cost_for_log_radius,
            bounds=(lower_log_radius, upper_log_radius),
            method="bounded",
            options={"xatol": 1.0e-10, "maxiter": 200},
        )
        if optimized.success and np.isfinite(optimized.fun):
            solved_radius_m = math.exp(float(optimized.x))
            expanded_boundary_radius_m = center_radius_m / math.sqrt(
                RADIAL_SEARCH_EXPANSION_FACTOR
            )
            projection_mode = (
                RANGE_ASYMPTOTIC
                if expansion_count > 0 and solved_radius_m >= expanded_boundary_radius_m
                else RANGE_REFINED
            )
        else:
            solved_radius_m = center_radius_m
            projection_mode = RANGE_ASYMPTOTIC
    else:
        finite_candidates = [
            (cost, log_radius)
            for cost, log_radius in (
                (lower_cost, lower_log_radius),
                (center_cost, center_log_radius),
                (upper_cost, upper_log_radius),
            )
            if np.isfinite(cost)
        ]
        if not finite_candidates:
            raise ValueError("Radial range refinement produced no finite candidate")
        _, farthest_finite_log_radius = max(finite_candidates, key=lambda item: item[1])
        solved_radius_m = math.exp(farthest_finite_log_radius)
        projection_mode = RANGE_ASYMPTOTIC

    fitted_position_m = centroid_m + normalized_direction * solved_radius_m
    fitted_residual = residual_function(fitted_position_m)
    fit_residual_rms_seconds = float(np.sqrt(np.mean(np.square(fitted_residual))))

    def radial_standard_deviation_at(radius_m: float) -> float:
        derivative_delta_m = max(radius_m * 1.0e-4, minimum_radius_m * 0.1, 1.0e-3)
        left_radius_m = max(radius_m - derivative_delta_m, lower_radius_m)
        right_radius_m = radius_m + derivative_delta_m
        left_residual = residual_function(centroid_m + normalized_direction * left_radius_m)
        right_residual = residual_function(centroid_m + normalized_direction * right_radius_m)
        derivative = (right_residual - left_residual) / max(
            right_radius_m - left_radius_m,
            EPSILON,
        )
        radial_information = float(np.linalg.norm(derivative))
        return (
            effective_time_std_s / radial_information
            if radial_information > EPSILON
            else radius_m * RADIAL_SEARCH_EXPANSION_FACTOR
        )

    fitted_radial_std_m = radial_standard_deviation_at(solved_radius_m)
    if (
        asymptotic_radial_std_fraction is not None
        and fitted_radial_std_m >= solved_radius_m * asymptotic_radial_std_fraction
    ):
        projection_mode = RANGE_ASYMPTOTIC
    published_radius_m = (
        max(solved_radius_m, initial_radius_m)
        if projection_mode == RANGE_ASYMPTOTIC
        else solved_radius_m
    )
    published_position_m = centroid_m + normalized_direction * published_radius_m
    radial_std_m = radial_standard_deviation_at(published_radius_m)
    if projection_mode == RANGE_ASYMPTOTIC:
        radial_std_m = max(radial_std_m, published_radius_m)
    return RadialRefinement(
        position_m=published_position_m,
        radial_std_m=float(radial_std_m),
        projection_mode=projection_mode,
        fit_residual_rms_seconds=fit_residual_rms_seconds,
    )
