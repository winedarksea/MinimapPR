"""BirdNET v2.4 -> V3.0 confidence calibration by score-distribution matching.

V3 confidences sit on a different scale from v2.4 (different training, a
sigmoid inside the graph, pooling over 2-6 s windows instead of 3 s
segments), so a v2.4 threshold such as 0.5 does not mean the same thing under
V3. Thresholds are carried across by *quantile matching* on the same audio:
the V3 threshold is the score that the same fraction of clips clears under V3
as clears the v2.4 threshold under v2.4::

    t_v3 = Q_v3(1 - P(score_v24 >= t_v24))

Matching the pass fraction keeps the operating point (alert volume, retention
volume, track-feed volume) unchanged across the cutover, which is what a
threshold encodes operationally. It does not claim V3 is equally *right* at
that point; that is what the replay gate measures.

``ConfidenceQuantileMap`` stores the matched curve as monotone knots so
thresholds that live outside Settings (routing ``min_confidence``, rule
``min_confidence``, retention ``confidence_min``) can be carried across the
same way at load time. The committed curve is
``assets/birdnet/v24_to_v3_confidence_map.json``, written by
``scripts/calibrate_birdnet_v3_thresholds.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_V24_TO_V3_CONFIDENCE_MAP_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "birdnet" / "v24_to_v3_confidence_map.json"
)
CONFIDENCE_FLOOR = 0.0
CONFIDENCE_CEILING = 1.0
# Knot grid on the v2.4 axis: dense where thresholds live (0.05-0.6).
DEFAULT_SOURCE_KNOT_GRID: tuple[float, ...] = (
    0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0,
)


def matched_threshold(
    source_scores: np.ndarray,
    target_scores: np.ndarray,
    source_threshold: float,
) -> float:
    """Target-model threshold passing the same clip fraction as ``source_threshold``.

    Raises ``ValueError`` on empty input: a calibration with no clips is a
    measurement failure, not a reason to fall back to the source value.
    """
    source = np.asarray(source_scores, dtype=np.float64)
    target = np.asarray(target_scores, dtype=np.float64)
    if source.size == 0 or target.size == 0:
        raise ValueError("matched_threshold needs non-empty score arrays")
    pass_fraction = float(np.mean(source >= source_threshold))
    if pass_fraction <= 0.0:
        # Nothing passes under v2.4: the equivalent V3 gate sits just above
        # every observed V3 score (no extrapolation past the data).
        return float(min(CONFIDENCE_CEILING, np.nextafter(target.max(), np.inf)))
    if pass_fraction >= 1.0:
        return float(max(CONFIDENCE_FLOOR, target.min()))
    # "higher" keeps the realised pass fraction <= the source's.
    return float(np.quantile(target, 1.0 - pass_fraction, method="higher"))


def bootstrap_matched_threshold_interval(
    source_scores: np.ndarray,
    target_scores: np.ndarray,
    source_threshold: float,
    *,
    resample_count: int,
    interval_fraction: float,
    seed: int,
) -> tuple[float, float]:
    """Paired bootstrap interval of ``matched_threshold`` (clips resampled jointly)."""
    source = np.asarray(source_scores, dtype=np.float64)
    target = np.asarray(target_scores, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError("bootstrap needs paired scores (one per clip per model)")
    generator = np.random.default_rng(seed)
    estimates = np.empty(resample_count)
    for index in range(resample_count):
        picked = generator.integers(0, source.size, source.size)
        estimates[index] = matched_threshold(source[picked], target[picked], source_threshold)
    tail = (1.0 - interval_fraction) / 2.0
    return float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1.0 - tail))


@dataclass(frozen=True, slots=True)
class ConfidenceQuantileMap:
    """Monotone piecewise-linear map from one model's confidence scale to another's."""

    source_model_version: str
    target_model_version: str
    source_knots: tuple[float, ...]
    target_knots: tuple[float, ...]
    calibration_clip_count: int = 0
    provenance: str = ""

    def __post_init__(self) -> None:
        if len(self.source_knots) != len(self.target_knots) or len(self.source_knots) < 2:
            raise ValueError("confidence map needs >= 2 paired knots")
        if any(b < a for a, b in zip(self.source_knots, self.source_knots[1:])):
            raise ValueError("confidence map source knots must be non-decreasing")
        if any(b < a for a, b in zip(self.target_knots, self.target_knots[1:])):
            raise ValueError("confidence map target knots must be non-decreasing")

    def map_confidence(self, source_confidence: float) -> float:
        value = float(np.interp(float(source_confidence), self.source_knots, self.target_knots))
        return float(min(CONFIDENCE_CEILING, max(CONFIDENCE_FLOOR, value)))

    def to_dict(self) -> dict:
        return {
            "source_model_version": self.source_model_version,
            "target_model_version": self.target_model_version,
            "source_knots": list(self.source_knots),
            "target_knots": list(self.target_knots),
            "calibration_clip_count": self.calibration_clip_count,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ConfidenceQuantileMap":
        return cls(
            source_model_version=str(payload["source_model_version"]),
            target_model_version=str(payload["target_model_version"]),
            source_knots=tuple(float(value) for value in payload["source_knots"]),
            target_knots=tuple(float(value) for value in payload["target_knots"]),
            calibration_clip_count=int(payload.get("calibration_clip_count", 0)),
            provenance=str(payload.get("provenance", "")),
        )


def build_confidence_quantile_map(
    source_scores: np.ndarray,
    target_scores: np.ndarray,
    *,
    source_model_version: str,
    target_model_version: str,
    source_knot_grid: tuple[float, ...] = DEFAULT_SOURCE_KNOT_GRID,
    provenance: str = "",
) -> ConfidenceQuantileMap:
    """Matched threshold at every grid knot, forced monotone, anchored at 0 and 1."""
    targets = [matched_threshold(source_scores, target_scores, knot) for knot in source_knot_grid]
    monotone_targets = list(np.maximum.accumulate(np.asarray(targets, dtype=np.float64)))
    if source_knot_grid[0] <= CONFIDENCE_FLOOR:
        monotone_targets[0] = CONFIDENCE_FLOOR
    if source_knot_grid[-1] >= CONFIDENCE_CEILING:
        monotone_targets[-1] = CONFIDENCE_CEILING
    monotone_targets = list(np.maximum.accumulate(np.asarray(monotone_targets)))
    return ConfidenceQuantileMap(
        source_model_version=source_model_version,
        target_model_version=target_model_version,
        source_knots=tuple(float(knot) for knot in source_knot_grid),
        target_knots=tuple(round(float(value), 6) for value in monotone_targets),
        calibration_clip_count=int(np.asarray(source_scores).size),
        provenance=provenance,
    )


def load_confidence_quantile_map(
    map_path: str | Path = DEFAULT_V24_TO_V3_CONFIDENCE_MAP_PATH,
) -> ConfidenceQuantileMap:
    return ConfidenceQuantileMap.from_dict(json.loads(Path(map_path).read_text(encoding="utf-8")))


def write_confidence_quantile_map(confidence_map: ConfidenceQuantileMap, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(confidence_map.to_dict(), indent=2) + "\n", encoding="utf-8")
