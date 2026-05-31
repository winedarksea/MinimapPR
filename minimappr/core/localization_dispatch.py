"""Runtime localization algorithm dispatch for Phase 2."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field

import numpy as np

from minimappr.config import LocalizationConfig, Settings
from minimappr.core.advanced_localization import EspritLocalizer, MusicLocalizer, SRPPhatLocalizer
from minimappr.core.ambi_atob import alias_cutoff_from_positions
from minimappr.core.localization import (
    LocalizationEngine,
    LocalizationError,
    dominant_frequency_hz,
    speed_of_sound_mps,
)
from minimappr.interfaces import Localizer
from minimappr.models import LocalizationResult


logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class _WavelengthObservability:
    alias_cutoff_hz: float
    dominant_frequency_hz: float
    wavelength_factor: float


@dataclass(slots=True)
class LocalizationDispatcher:
    strategy: str = "fixed"
    default_algorithm: str = "gcc_phat"
    refine_confidence_threshold: float = 0.45
    tight_array_aperture_m: float = 0.35
    wavelength_gating_enabled: bool = True
    wavelength_penalty_floor: float = 0.25
    algorithms: dict[str, Localizer] = field(default_factory=dict)
    _fallback: Localizer = field(init=False, repr=False)
    _last_algorithm: str = field(init=False, default="gcc_phat", repr=False)
    _last_attempted: str = field(init=False, default="gcc_phat", repr=False)
    _fallback_count: int = field(init=False, default=0, repr=False)
    _config_bypassed_count: int = field(init=False, default=0, repr=False)
    _config_bypass_warned: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if "gcc_phat" not in self.algorithms:
            self.algorithms["gcc_phat"] = LocalizationEngine()
        if self.wavelength_penalty_floor < 0.0 or self.wavelength_penalty_floor > 1.0:
            raise ValueError("wavelength_penalty_floor must be in [0,1]")
        self._fallback = self.algorithms["gcc_phat"]
        if self.default_algorithm not in self.algorithms:
            self.default_algorithm = "gcc_phat"
        self._last_algorithm = "gcc_phat"
        self._last_attempted = "gcc_phat"
        self._fallback_count = 0
        self._config_bypassed_count = 0
        self._config_bypass_warned = False

    def last_algorithm_name(self) -> str:
        return self._last_algorithm

    def last_attempted_algorithm_name(self) -> str:
        return self._last_attempted

    def fallback_count(self) -> int:
        return self._fallback_count

    def config_bypassed_count(self) -> int:
        return self._config_bypassed_count

    def _normalized_strategy(self) -> str:
        return self.strategy.strip().lower()

    def select_algorithm_name(self, sensor_positions: dict[str, np.ndarray]) -> str:
        strategy = self._normalized_strategy()
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
        sensor_weights: dict[str, float] | None = None,
    ) -> LocalizationResult:
        result = self._localize_for_strategy(
            sensor_positions=sensor_positions,
            sensor_windows=sensor_windows,
            sample_rate_hz=sample_rate_hz,
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
            sensor_weights=sensor_weights,
        )
        return self._apply_wavelength_penalty(
            result=result,
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
        sensor_weights: dict[str, float] | None = None,
    ) -> LocalizationResult:
        fallback = self._fallback
        if not hasattr(fallback, "localize_2d"):
            return self.localize(
                sensor_positions=sensor_positions,
                sensor_windows=sensor_windows,
                sample_rate_hz=sample_rate_hz,
                temperature_c=temperature_c,
                humidity_fraction=humidity_fraction,
                sensor_weights=sensor_weights,
            )
        self._last_attempted = "gcc_phat"
        if self.default_algorithm != "gcc_phat":
            self._config_bypassed_count += 1
            if not self._config_bypass_warned:
                self._config_bypass_warned = True
                logging.getLogger(__name__).warning(
                    "localize_2d: configured algorithm %r is bypassed — 2D path always uses gcc_phat. "
                    "Fix the position/geometry before treating this as the correct algorithm.",
                    self.default_algorithm,
                )
        if sensor_weights is not None and self._localizer_supports_sensor_weights(
            fallback,
            method_name="localize_2d",
        ):
            result = fallback.localize_2d(
                sensor_positions=sensor_positions,
                sensor_windows=sensor_windows,
                sample_rate_hz=sample_rate_hz,
                temperature_c=temperature_c,
                humidity_fraction=humidity_fraction,
                fixed_z_m=fixed_z_m,
                sensor_weights=sensor_weights,
            )
        else:
            result = fallback.localize_2d(
                sensor_positions=sensor_positions,
                sensor_windows=sensor_windows,
                sample_rate_hz=sample_rate_hz,
                temperature_c=temperature_c,
                humidity_fraction=humidity_fraction,
                fixed_z_m=fixed_z_m,
            )
        self._record_algorithm_provenance(
            result,
            attempted_algorithm="gcc_phat",
            resolved_algorithm="gcc_phat",
        )
        return self._apply_wavelength_penalty(
            result=result,
            sensor_positions=sensor_positions,
            sensor_windows=sensor_windows,
            sample_rate_hz=sample_rate_hz,
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
        )

    def _localize_for_strategy(
        self,
        *,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        sensor_weights: dict[str, float] | None = None,
    ) -> LocalizationResult:
        if self._normalized_strategy() == "cascade":
            return self._localize_with_cascade(
                sensor_positions=sensor_positions,
                sensor_windows=sensor_windows,
                sample_rate_hz=sample_rate_hz,
                temperature_c=temperature_c,
                humidity_fraction=humidity_fraction,
                sensor_weights=sensor_weights,
            )
        return self._localize_selected_algorithm(
            name=self.select_algorithm_name(sensor_positions),
            sensor_positions=sensor_positions,
            sensor_windows=sensor_windows,
            sample_rate_hz=sample_rate_hz,
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
            sensor_weights=sensor_weights,
        )

    def _localize_selected_algorithm(
        self,
        *,
        name: str,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        sensor_weights: dict[str, float] | None = None,
    ) -> LocalizationResult:
        return self._run_algorithm(
            name=name,
            sensor_positions=sensor_positions,
            sensor_windows=sensor_windows,
            sample_rate_hz=sample_rate_hz,
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
            sensor_weights=sensor_weights,
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
        sensor_weights: dict[str, float] | None = None,
    ) -> LocalizationResult:
        result = self._localize_selected_algorithm(
            name=self.default_algorithm,
            sensor_positions=sensor_positions,
            sensor_windows=sensor_windows,
            sample_rate_hz=sample_rate_hz,
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
            sensor_weights=sensor_weights,
        )
        if not self._should_refine_cascade_result(result):
            return result

        best = result
        best_name = self.default_algorithm
        for name in self._cascade_refinement_names():
            if name == self.default_algorithm or name not in self.algorithms:
                continue
            try:
                candidate = self._localize_selected_algorithm(
                    name=name,
                    sensor_positions=sensor_positions,
                    sensor_windows=sensor_windows,
                    sample_rate_hz=sample_rate_hz,
                    temperature_c=temperature_c,
                    humidity_fraction=humidity_fraction,
                    sensor_weights=sensor_weights,
                )
                if candidate.confidence > best.confidence:
                    best = candidate
                    best_name = name
            except LocalizationError:
                continue
        self._record_algorithm_provenance(
            best,
            attempted_algorithm=self.default_algorithm,
            resolved_algorithm=best_name,
        )
        if best_name != self.default_algorithm:
            self._fallback_count += 1
        return best

    def _should_refine_cascade_result(self, result: LocalizationResult) -> bool:
        return result.confidence < self.refine_confidence_threshold

    @staticmethod
    def _cascade_refinement_names() -> tuple[str, ...]:
        return ("music", "esprit", "srp_phat")

    def _run_algorithm(
        self,
        *,
        name: str,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        sensor_weights: dict[str, float] | None = None,
    ) -> LocalizationResult:
        localizer = self.algorithms.get(name, self._fallback)
        attempted = name if name in self.algorithms else "gcc_phat"
        self._last_attempted = attempted
        try:
            result = self._call_localizer(
                localizer=localizer,
                localizer_name=attempted,
                sensor_positions=sensor_positions,
                sensor_windows=sensor_windows,
                sample_rate_hz=sample_rate_hz,
                temperature_c=temperature_c,
                humidity_fraction=humidity_fraction,
                sensor_weights=sensor_weights,
            )
            self._record_algorithm_provenance(
                result,
                attempted_algorithm=attempted,
                resolved_algorithm=attempted,
            )
            return result
        except LocalizationError:
            if name == "gcc_phat":
                raise
            result = self._call_localizer(
                localizer=self._fallback,
                localizer_name="gcc_phat",
                sensor_positions=sensor_positions,
                sensor_windows=sensor_windows,
                sample_rate_hz=sample_rate_hz,
                temperature_c=temperature_c,
                humidity_fraction=humidity_fraction,
                sensor_weights=sensor_weights,
            )
            self._fallback_count += 1
            self._record_algorithm_provenance(
                result,
                attempted_algorithm=attempted,
                resolved_algorithm="gcc_phat",
            )
            return result

    def _record_algorithm_provenance(
        self,
        result: LocalizationResult,
        *,
        attempted_algorithm: str,
        resolved_algorithm: str,
    ) -> None:
        self._last_attempted = attempted_algorithm
        self._last_algorithm = resolved_algorithm
        result.attempted_algorithm = attempted_algorithm
        result.resolved_algorithm = resolved_algorithm

    def _apply_wavelength_penalty(
        self,
        *,
        result: LocalizationResult,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> LocalizationResult:
        if not self.wavelength_gating_enabled:
            return result

        observability = self._compute_wavelength_observability(
            reference_sensor=result.reference_sensor,
            sensor_positions=sensor_positions,
            sensor_windows=sensor_windows,
            sample_rate_hz=sample_rate_hz,
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
        )
        if observability is None:
            return result

        result.alias_cutoff_hz = observability.alias_cutoff_hz
        result.dominant_frequency_hz = observability.dominant_frequency_hz
        result.wavelength_factor = observability.wavelength_factor
        result.confidence = float(
            np.clip(result.confidence * observability.wavelength_factor, 0.0, 1.0)
        )
        return result

    def _compute_wavelength_observability(
        self,
        *,
        reference_sensor: str,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> _WavelengthObservability | None:
        reference_window = self._reference_window_for_observability(
            reference_sensor=reference_sensor,
            sensor_windows=sensor_windows,
        )
        if reference_window is None or len(sensor_positions) < 2:
            return None

        sensor_ids = sorted(sensor_positions.keys())
        positions = np.vstack([np.asarray(sensor_positions[sensor_id], dtype=np.float64) for sensor_id in sensor_ids])
        alias_cutoff_hz = alias_cutoff_from_positions(
            positions,
            c_sound=speed_of_sound_mps(temperature_c, humidity_fraction),
        )
        dominant_hz = dominant_frequency_hz(reference_window, sample_rate_hz)
        return _WavelengthObservability(
            alias_cutoff_hz=float(max(alias_cutoff_hz, 0.0)),
            dominant_frequency_hz=float(max(dominant_hz, 0.0)),
            wavelength_factor=self._wavelength_factor(alias_cutoff_hz, dominant_hz),
        )

    @staticmethod
    def _reference_window_for_observability(
        *,
        reference_sensor: str,
        sensor_windows: dict[str, np.ndarray],
    ) -> np.ndarray | None:
        reference_window = sensor_windows.get(reference_sensor)
        if reference_window is None and sensor_windows:
            return next(iter(sensor_windows.values()))
        return reference_window

    def _wavelength_factor(self, alias_cutoff_hz: float, dominant_hz: float) -> float:
        if alias_cutoff_hz <= 0.0 or dominant_hz <= 0.0:
            return 1.0
        return float(
            np.clip(
                alias_cutoff_hz / max(dominant_hz, alias_cutoff_hz),
                self.wavelength_penalty_floor,
                1.0,
            )
        )

    @staticmethod
    def _localizer_supports_sensor_weights(localizer, *, method_name: str = "localize") -> bool:
        method = getattr(localizer, method_name, None)
        if method is None:
            return False
        try:
            return "sensor_weights" in inspect.signature(method).parameters
        except (TypeError, ValueError):
            return isinstance(localizer, LocalizationEngine)

    @staticmethod
    def _call_localizer(
        *,
        localizer,
        localizer_name: str,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        sensor_weights: dict[str, float] | None = None,
    ) -> LocalizationResult:
        """Call localizer.localize(), passing sensor_weights only when supported."""
        if sensor_weights is not None and LocalizationDispatcher._localizer_supports_sensor_weights(localizer):
            return localizer.localize(
                sensor_positions=sensor_positions,
                sensor_windows=sensor_windows,
                sample_rate_hz=sample_rate_hz,
                temperature_c=temperature_c,
                humidity_fraction=humidity_fraction,
                sensor_weights=sensor_weights,
            )
        if sensor_weights is not None:
            logger.warning(
                "Localization algorithm %s does not support sensor_weights; falling back to gcc_phat",
                localizer_name,
            )
            raise LocalizationError(
                f"Localization algorithm '{localizer_name}' does not support sensor_weights"
            )
        return localizer.localize(
            sensor_positions=sensor_positions,
            sensor_windows=sensor_windows,
            sample_rate_hz=sample_rate_hz,
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
        )


def build_localizer_from_settings(settings: Settings | LocalizationConfig) -> Localizer:
    cfg = settings.localization_config() if isinstance(settings, Settings) else settings
    gcc = LocalizationEngine(
        max_tau_s=cfg.localization_max_tau_seconds,
        interp_factor=cfg.gcc_phat_interp_factor,
    )
    algorithms: dict[str, Localizer] = {
        "gcc_phat": gcc,
        "srp_phat": SRPPhatLocalizer(
            max_tau_s=cfg.localization_max_tau_seconds,
            grid_resolution_m=cfg.localization_srp_grid_resolution_m,
            search_padding_m=cfg.localization_search_padding_m,
            interp=cfg.gcc_phat_interp_factor,
            tight_array_aperture_m=cfg.localization_tight_array_aperture_m,
            far_field_default_range_m=cfg.localization_far_field_default_range_m,
            far_field_max_range_m=cfg.localization_far_field_max_range_m,
            far_field_azimuth_step_deg=cfg.localization_music_azimuth_step_deg,
            far_field_elevation_step_deg=cfg.localization_music_elevation_step_deg,
        ),
        "music": MusicLocalizer(
            max_tau_s=cfg.localization_max_tau_seconds,
            azimuth_step_deg=cfg.localization_music_azimuth_step_deg,
            elevation_step_deg=cfg.localization_music_elevation_step_deg,
            freq_min_hz=cfg.localization_subspace_freq_min_hz,
            freq_max_hz=cfg.localization_subspace_freq_max_hz,
            interp=cfg.gcc_phat_interp_factor,
            far_field_default_range_m=cfg.localization_far_field_default_range_m,
            far_field_max_range_m=cfg.localization_far_field_max_range_m,
        ),
        "esprit": EspritLocalizer(
            max_tau_s=cfg.localization_max_tau_seconds,
            freq_min_hz=cfg.localization_subspace_freq_min_hz,
            freq_max_hz=cfg.localization_subspace_freq_max_hz,
            interp=cfg.gcc_phat_interp_factor,
            far_field_default_range_m=cfg.localization_far_field_default_range_m,
            far_field_max_range_m=cfg.localization_far_field_max_range_m,
        ),
    }
    return LocalizationDispatcher(
        strategy=cfg.localization_strategy,
        default_algorithm=cfg.localization_algorithm,
        refine_confidence_threshold=cfg.localization_refine_confidence_threshold,
        tight_array_aperture_m=cfg.localization_tight_array_aperture_m,
        wavelength_gating_enabled=cfg.wavelength_gating_enabled,
        wavelength_penalty_floor=cfg.wavelength_penalty_floor,
        algorithms=algorithms,
    )
