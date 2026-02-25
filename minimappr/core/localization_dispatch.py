"""Runtime localization algorithm dispatch for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from minimappr.config import LocalizationConfig, Settings
from minimappr.core.advanced_localization import EspritLocalizer, MusicLocalizer, SRPPhatLocalizer
from minimappr.core.localization import LocalizationEngine, LocalizationError
from minimappr.interfaces import Localizer
from minimappr.models import LocalizationResult


def _array_aperture_m(sensor_positions: dict[str, np.ndarray]) -> float:
    sensor_ids = sorted(sensor_positions.keys())
    if len(sensor_ids) < 2:
        return 0.0
    positions = [sensor_positions[sensor_id] for sensor_id in sensor_ids]
    return float(
        max(
            np.linalg.norm(positions[i] - positions[j])
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
        )
    )


@dataclass(slots=True)
class LocalizationDispatcher:
    strategy: str = "fixed"
    default_algorithm: str = "gcc_phat"
    refine_confidence_threshold: float = 0.45
    tight_array_aperture_m: float = 0.35
    algorithms: dict[str, Localizer] = field(default_factory=dict)
    _fallback: Localizer = field(init=False, repr=False)
    _last_algorithm: str = field(init=False, default="gcc_phat", repr=False)

    def __post_init__(self) -> None:
        if "gcc_phat" not in self.algorithms:
            self.algorithms["gcc_phat"] = LocalizationEngine()
        self._fallback = self.algorithms["gcc_phat"]
        if self.default_algorithm not in self.algorithms:
            self.default_algorithm = "gcc_phat"
        self._last_algorithm = "gcc_phat"

    def last_algorithm_name(self) -> str:
        return self._last_algorithm

    def select_algorithm_name(self, sensor_positions: dict[str, np.ndarray]) -> str:
        strategy = self.strategy.strip().lower()
        if strategy == "fixed":
            return self.default_algorithm
        if strategy == "geometry_aware":
            return self._geometry_aware_choice(sensor_positions)
        if strategy == "cascade":
            return self.default_algorithm
        return self.default_algorithm

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> LocalizationResult:
        strategy = self.strategy.strip().lower()
        if strategy == "cascade":
            return self._localize_with_cascade(
                sensor_positions=sensor_positions,
                sensor_windows=sensor_windows,
                sample_rate_hz=sample_rate_hz,
                temperature_c=temperature_c,
                humidity_fraction=humidity_fraction,
            )

        name = self.select_algorithm_name(sensor_positions)
        return self._run_algorithm(
            name=name,
            sensor_positions=sensor_positions,
            sensor_windows=sensor_windows,
            sample_rate_hz=sample_rate_hz,
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
        )

    def localize_2d(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        fixed_z_m: float | None = None,
    ) -> LocalizationResult:
        fallback = self._fallback
        if not hasattr(fallback, "localize_2d"):
            return self.localize(
                sensor_positions=sensor_positions,
                sensor_windows=sensor_windows,
                sample_rate_hz=sample_rate_hz,
                temperature_c=temperature_c,
                humidity_fraction=humidity_fraction,
            )
        self._last_algorithm = "gcc_phat"
        return fallback.localize_2d(
            sensor_positions=sensor_positions,
            sensor_windows=sensor_windows,
            sample_rate_hz=sample_rate_hz,
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
            fixed_z_m=fixed_z_m,
        )

    def _geometry_aware_choice(self, sensor_positions: dict[str, np.ndarray]) -> str:
        sensor_count = len(sensor_positions)
        aperture = _array_aperture_m(sensor_positions)

        if sensor_count >= 4 and aperture <= self.tight_array_aperture_m and "esprit" in self.algorithms:
            return "esprit"
        if sensor_count >= 4 and aperture <= (self.tight_array_aperture_m * 5.0) and "music" in self.algorithms:
            return "music"
        if sensor_count >= 4 and "srp_phat" in self.algorithms:
            return "srp_phat"
        return "gcc_phat"

    def _localize_with_cascade(
        self,
        *,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> LocalizationResult:
        result = self._run_algorithm(
            name=self.default_algorithm,
            sensor_positions=sensor_positions,
            sensor_windows=sensor_windows,
            sample_rate_hz=sample_rate_hz,
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
        )
        if result.confidence >= self.refine_confidence_threshold:
            return result

        best = result
        best_name = self.default_algorithm
        for name in ("music", "esprit", "srp_phat"):
            if name == self.default_algorithm or name not in self.algorithms:
                continue
            try:
                candidate = self._run_algorithm(
                    name=name,
                    sensor_positions=sensor_positions,
                    sensor_windows=sensor_windows,
                    sample_rate_hz=sample_rate_hz,
                    temperature_c=temperature_c,
                    humidity_fraction=humidity_fraction,
                )
                if candidate.confidence > best.confidence:
                    best = candidate
                    best_name = name
            except LocalizationError:
                continue
        self._last_algorithm = best_name
        return best

    def _run_algorithm(
        self,
        *,
        name: str,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> LocalizationResult:
        localizer = self.algorithms.get(name, self._fallback)
        try:
            result = localizer.localize(
                sensor_positions=sensor_positions,
                sensor_windows=sensor_windows,
                sample_rate_hz=sample_rate_hz,
                temperature_c=temperature_c,
                humidity_fraction=humidity_fraction,
            )
            self._last_algorithm = name if name in self.algorithms else "gcc_phat"
            return result
        except LocalizationError:
            if name == "gcc_phat":
                raise
            result = self._fallback.localize(
                sensor_positions=sensor_positions,
                sensor_windows=sensor_windows,
                sample_rate_hz=sample_rate_hz,
                temperature_c=temperature_c,
                humidity_fraction=humidity_fraction,
            )
            self._last_algorithm = "gcc_phat"
            return result


def build_localizer_from_settings(settings: Settings | LocalizationConfig) -> Localizer:
    cfg = settings.localization_config() if isinstance(settings, Settings) else settings
    gcc = LocalizationEngine(
        max_tau_s=cfg.localization_max_tau_s,
        interp_factor=cfg.gcc_phat_interp_factor,
    )
    algorithms: dict[str, Localizer] = {
        "gcc_phat": gcc,
        "srp_phat": SRPPhatLocalizer(
            max_tau_s=cfg.localization_max_tau_s,
            grid_resolution_m=cfg.localization_srp_grid_resolution_m,
            search_padding_m=cfg.localization_search_padding_m,
            interp=cfg.gcc_phat_interp_factor,
        ),
        "music": MusicLocalizer(
            max_tau_s=cfg.localization_max_tau_s,
            azimuth_step_deg=cfg.localization_music_azimuth_step_deg,
            elevation_step_deg=cfg.localization_music_elevation_step_deg,
            freq_min_hz=cfg.localization_subspace_freq_min_hz,
            freq_max_hz=cfg.localization_subspace_freq_max_hz,
            interp=cfg.gcc_phat_interp_factor,
        ),
        "esprit": EspritLocalizer(
            max_tau_s=cfg.localization_max_tau_s,
            freq_min_hz=cfg.localization_subspace_freq_min_hz,
            freq_max_hz=cfg.localization_subspace_freq_max_hz,
            interp=cfg.gcc_phat_interp_factor,
        ),
    }
    return LocalizationDispatcher(
        strategy=cfg.localization_strategy,
        default_algorithm=cfg.localization_algorithm,
        refine_confidence_threshold=cfg.localization_refine_confidence_threshold,
        tight_array_aperture_m=cfg.localization_tight_array_aperture_m,
        algorithms=algorithms,
    )
