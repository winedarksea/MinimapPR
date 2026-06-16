"""Phase 2 localizers using distinct bearing methods and shared Cartesian range."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from minimappr.core.cartesian_tdoa import (
    localization_result_from_cartesian_solve,
    solve_cartesian_tdoa,
)
from minimappr.core.localization import LocalizationError, gcc_phat, speed_of_sound_mps
from minimappr.core.spatial_constraints import build_node_spatial_constraints
from minimappr.core.subspace_bearing import (
    BearingPrior,
    esprit_bearing_prior,
    music_bearing_prior,
)
from minimappr.core.tdoa_measurements import measure_pair_tdoas, reference_tdoas
from minimappr.models import LocalizationResult

BearingPriorEstimator = Callable[
    [dict[str, np.ndarray], dict[str, np.ndarray], list[str], int, float],
    BearingPrior,
]
NodeBearingEstimator = Callable[
    [dict[str, np.ndarray], dict[str, np.ndarray], list[str]],
    tuple[np.ndarray, float],
]


def _uses_multiple_sensor_nodes(
    sensor_ids: list[str],
    sensor_node_ids: dict[str, str] | None,
) -> bool:
    return sensor_node_ids is not None and len(
        {sensor_node_ids.get(sensor_id) for sensor_id in sensor_ids}
    ) > 1


def _localize_cartesian(
    *,
    sensor_positions: dict[str, np.ndarray],
    sensor_windows: dict[str, np.ndarray],
    sample_rate_hz: int,
    temperature_c: float,
    humidity_fraction: float,
    max_tau_s: float,
    interpolation_factor: int,
    sensor_weights: dict[str, float] | None,
    bearing_prior: BearingPrior | None = None,
    bearing_prior_estimator: BearingPriorEstimator | None = None,
    sensor_node_ids: dict[str, str] | None = None,
    sensor_gain_offsets_db: dict[str, float] | None = None,
    node_bearing_estimator: NodeBearingEstimator | None = None,
    node_bearing_strength: float = 1.0,
    amplitude_ratio_strength: float = 0.0,
    far_field_initial_range_m: float = 50.0,
    tight_array_aperture_m: float = 0.35,
) -> LocalizationResult:
    sensor_ids = sorted(sensor_id for sensor_id in sensor_positions if sensor_id in sensor_windows)
    if len(sensor_ids) < 4:
        raise LocalizationError("Need at least 4 active sensors for Cartesian TDOA localization")
    sound_speed_mps = speed_of_sound_mps(
        temperature_c=temperature_c,
        humidity_fraction=humidity_fraction,
    )
    has_multiple_nodes = _uses_multiple_sensor_nodes(sensor_ids, sensor_node_ids)
    active_node_bearing_estimator = node_bearing_estimator
    if bearing_prior_estimator is not None:
        def estimate_bearing(
            node_positions: dict[str, np.ndarray],
            node_windows: dict[str, np.ndarray],
            node_sensor_ids: list[str],
        ) -> tuple[np.ndarray, float]:
            prior = bearing_prior_estimator(
                node_positions,
                node_windows,
                node_sensor_ids,
                sample_rate_hz,
                sound_speed_mps,
            )
            return prior.direction, prior.quality

        if has_multiple_nodes:
            active_node_bearing_estimator = estimate_bearing
        elif bearing_prior is None:
            try:
                bearing_prior = bearing_prior_estimator(
                    sensor_positions,
                    sensor_windows,
                    sensor_ids,
                    sample_rate_hz,
                    sound_speed_mps,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise LocalizationError(str(exc)) from exc
    measurements = measure_pair_tdoas(
        sensor_positions=sensor_positions,
        sensor_windows=sensor_windows,
        sensor_ids=sensor_ids,
        sample_rate_hz=sample_rate_hz,
        sound_speed_mps=sound_speed_mps,
        max_tau_s=max_tau_s,
        interpolation_factor=interpolation_factor,
        sensor_weights=sensor_weights,
        gcc_phat_function=gcc_phat,
        sensor_node_ids=sensor_node_ids,
    )
    array_aperture_m = max(
        (
            float(np.linalg.norm(sensor_positions[a] - sensor_positions[b]))
            for index, a in enumerate(sensor_ids)
            for b in sensor_ids[index + 1 :]
        ),
        default=0.0,
    )
    bearing_constraints, amplitude_constraints = build_node_spatial_constraints(
        sensor_positions=sensor_positions,
        sensor_windows=sensor_windows,
        sensor_node_ids=sensor_node_ids,
        sample_rate_hz=sample_rate_hz,
        sound_speed_mps=sound_speed_mps,
        max_tau_s=max_tau_s,
        interpolation_factor=interpolation_factor,
        gcc_phat_function=gcc_phat,
        bearing_strength=node_bearing_strength,
        amplitude_ratio_strength=amplitude_ratio_strength,
        pair_measurements=measurements,
        sensor_gain_offsets_db=sensor_gain_offsets_db,
        node_bearing_estimator=active_node_bearing_estimator,
    )
    if (
        len(measurements) < 3
        and len(bearing_constraints) < 2
        and len(amplitude_constraints) < 3
    ):
        raise LocalizationError("Insufficient TDOA, bearing, or amplitude constraints")
    reference_sensor, reference_tdoa_s = reference_tdoas(
        measurements=measurements,
        sensor_windows=sensor_windows,
        sensor_ids=sensor_ids,
    )
    try:
        solve = solve_cartesian_tdoa(
            sensor_positions={sensor_id: sensor_positions[sensor_id] for sensor_id in sensor_ids},
            measurements=measurements,
            sound_speed_mps=sound_speed_mps,
            sample_rate_hz=sample_rate_hz,
            interpolation_factor=interpolation_factor,
            reference_sensor=reference_sensor,
            reference_tdoa_s=reference_tdoa_s,
            bearing_prior=bearing_prior,
            bearing_constraints=bearing_constraints,
            amplitude_constraints=amplitude_constraints,
            far_field_initial_range_m=far_field_initial_range_m,
            radial_refinement_enabled=array_aperture_m <= tight_array_aperture_m,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise LocalizationError(str(exc)) from exc
    return localization_result_from_cartesian_solve(
        solve=solve,
        reference_sensor=reference_sensor,
        reference_tdoa_s=reference_tdoa_s,
    )


@dataclass(slots=True)
class SRPPhatLocalizer:
    # Legacy grid/max-range fields are ignored; default range seeds unbounded radial search.
    max_tau_s: float = 0.02
    grid_resolution_m: float = 0.5
    search_padding_m: float = 2.0
    interp: int = 4
    max_grid_points: int = 60_000
    tight_array_aperture_m: float = 0.35
    far_field_default_range_m: float = 50.0
    far_field_max_range_m: float = 250.0
    far_field_azimuth_step_deg: float = 6.0
    far_field_elevation_step_deg: float = 8.0
    node_bearing_strength: float = 1.0
    amplitude_ratio_strength: float = 0.15

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        sensor_weights: dict[str, float] | None = None,
        sensor_node_ids: dict[str, str] | None = None,
        sensor_gain_offsets_db: dict[str, float] | None = None,
    ) -> LocalizationResult:
        return _localize_cartesian(
            sensor_positions=sensor_positions,
            sensor_windows=sensor_windows,
            sample_rate_hz=sample_rate_hz,
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
            max_tau_s=self.max_tau_s,
            interpolation_factor=self.interp,
            sensor_weights=sensor_weights,
            sensor_node_ids=sensor_node_ids,
            sensor_gain_offsets_db=sensor_gain_offsets_db,
            node_bearing_strength=self.node_bearing_strength,
            amplitude_ratio_strength=self.amplitude_ratio_strength,
            far_field_initial_range_m=self.far_field_default_range_m,
            tight_array_aperture_m=self.tight_array_aperture_m,
        )


@dataclass(slots=True)
class MusicLocalizer:
    max_tau_s: float = 0.02
    azimuth_step_deg: float = 6.0
    elevation_step_deg: float = 8.0
    freq_min_hz: float = 300.0
    freq_max_hz: float = 3500.0
    source_count: int | None = None
    interp: int = 4
    far_field_default_range_m: float = 50.0
    far_field_max_range_m: float = 250.0
    node_bearing_strength: float = 1.0
    amplitude_ratio_strength: float = 0.15

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        sensor_weights: dict[str, float] | None = None,
        sensor_node_ids: dict[str, str] | None = None,
        sensor_gain_offsets_db: dict[str, float] | None = None,
    ) -> LocalizationResult:
        def estimate_bearing_prior(
            estimator_sensor_positions: dict[str, np.ndarray],
            estimator_sensor_windows: dict[str, np.ndarray],
            estimator_sensor_ids: list[str],
            estimator_sample_rate_hz: int,
            estimator_sound_speed_mps: float,
        ) -> BearingPrior:
            return music_bearing_prior(
                sensor_positions=estimator_sensor_positions,
                sensor_windows=estimator_sensor_windows,
                sensor_ids=estimator_sensor_ids,
                sample_rate_hz=estimator_sample_rate_hz,
                sound_speed_mps=estimator_sound_speed_mps,
                azimuth_step_deg=self.azimuth_step_deg,
                elevation_step_deg=self.elevation_step_deg,
                minimum_frequency_hz=self.freq_min_hz,
                maximum_frequency_hz=self.freq_max_hz,
                source_count=self.source_count,
            )

        return _localize_cartesian(
            sensor_positions=sensor_positions,
            sensor_windows=sensor_windows,
            sample_rate_hz=sample_rate_hz,
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
            max_tau_s=self.max_tau_s,
            interpolation_factor=self.interp,
            sensor_weights=sensor_weights,
            bearing_prior_estimator=estimate_bearing_prior,
            sensor_node_ids=sensor_node_ids,
            sensor_gain_offsets_db=sensor_gain_offsets_db,
            node_bearing_strength=self.node_bearing_strength,
            amplitude_ratio_strength=self.amplitude_ratio_strength,
            far_field_initial_range_m=self.far_field_default_range_m,
        )


@dataclass(slots=True)
class EspritLocalizer:
    max_tau_s: float = 0.02
    freq_min_hz: float = 300.0
    freq_max_hz: float = 3500.0
    interp: int = 4
    far_field_default_range_m: float = 50.0
    far_field_max_range_m: float = 250.0
    node_bearing_strength: float = 1.0
    amplitude_ratio_strength: float = 0.15

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        sensor_weights: dict[str, float] | None = None,
        sensor_node_ids: dict[str, str] | None = None,
        sensor_gain_offsets_db: dict[str, float] | None = None,
    ) -> LocalizationResult:
        def estimate_bearing_prior(
            estimator_sensor_positions: dict[str, np.ndarray],
            estimator_sensor_windows: dict[str, np.ndarray],
            estimator_sensor_ids: list[str],
            estimator_sample_rate_hz: int,
            estimator_sound_speed_mps: float,
        ) -> BearingPrior:
            return esprit_bearing_prior(
                sensor_positions=estimator_sensor_positions,
                sensor_windows=estimator_sensor_windows,
                sensor_ids=estimator_sensor_ids,
                sample_rate_hz=estimator_sample_rate_hz,
                sound_speed_mps=estimator_sound_speed_mps,
                minimum_frequency_hz=self.freq_min_hz,
                maximum_frequency_hz=self.freq_max_hz,
            )

        return _localize_cartesian(
            sensor_positions=sensor_positions,
            sensor_windows=sensor_windows,
            sample_rate_hz=sample_rate_hz,
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
            max_tau_s=self.max_tau_s,
            interpolation_factor=self.interp,
            sensor_weights=sensor_weights,
            bearing_prior_estimator=estimate_bearing_prior,
            sensor_node_ids=sensor_node_ids,
            sensor_gain_offsets_db=sensor_gain_offsets_db,
            node_bearing_strength=self.node_bearing_strength,
            amplitude_ratio_strength=self.amplitude_ratio_strength,
            far_field_initial_range_m=self.far_field_default_range_m,
        )
