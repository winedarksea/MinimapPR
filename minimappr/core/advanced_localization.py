"""Phase 2 localizer interfaces backed by one broadband Cartesian TDOA solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minimappr.core.cartesian_tdoa import (
    localization_result_from_cartesian_solve,
    measure_pair_tdoas,
    reference_tdoas,
    solve_cartesian_tdoa,
)
from minimappr.core.localization import LocalizationError, gcc_phat, speed_of_sound_mps
from minimappr.models import LocalizationResult


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
) -> LocalizationResult:
    sensor_ids = sorted(sensor_id for sensor_id in sensor_positions if sensor_id in sensor_windows)
    if len(sensor_ids) < 4:
        raise LocalizationError("Need at least 4 active sensors for Cartesian TDOA localization")
    sound_speed_mps = speed_of_sound_mps(
        temperature_c=temperature_c,
        humidity_fraction=humidity_fraction,
    )
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
    )
    if len(measurements) < 3:
        raise LocalizationError("Insufficient finite TDOA pairs for Cartesian localization")
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
    # Legacy grid/range fields remain constructor-compatible but no longer affect solving.
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

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        sensor_weights: dict[str, float] | None = None,
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
        )


@dataclass(slots=True)
class MusicLocalizer:
    # Narrowband settings remain constructor-compatible; broadband TDOA owns position.
    max_tau_s: float = 0.02
    azimuth_step_deg: float = 6.0
    elevation_step_deg: float = 8.0
    freq_min_hz: float = 300.0
    freq_max_hz: float = 3500.0
    source_count: int | None = None
    interp: int = 4
    far_field_default_range_m: float = 50.0
    far_field_max_range_m: float = 250.0

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        sensor_weights: dict[str, float] | None = None,
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
        )


@dataclass(slots=True)
class EspritLocalizer:
    # Narrowband settings remain constructor-compatible; broadband TDOA owns position.
    max_tau_s: float = 0.02
    freq_min_hz: float = 300.0
    freq_max_hz: float = 3500.0
    interp: int = 4
    far_field_default_range_m: float = 50.0
    far_field_max_range_m: float = 250.0

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        sensor_weights: dict[str, float] | None = None,
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
        )
