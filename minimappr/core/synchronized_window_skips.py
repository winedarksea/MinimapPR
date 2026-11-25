"""Counts of sensors a synchronized buffer read left out, per sensor and reason (plan item P8).

``MultiSensorBuffer`` builds synchronized windows only from sensors buffered
at the requested rate; mismatched-rate sensors stay excluded (guard now,
resample later). Before P8 the skip was silent, so an operator could not see
why a node never contributed. The reason spelling matches the Rust solve
contract's ``excluded_sensors`` reason, so both ingest paths read the same.
"""

from __future__ import annotations

from dataclasses import dataclass

from minimappr.api.rust_localization_solve_contract import SolveExclusionReason

SAMPLE_RATE_MISMATCH_SKIP_REASON = SolveExclusionReason.SAMPLE_RATE_MISMATCH.value


@dataclass(frozen=True, slots=True)
class SampleRateMismatch:
    buffered_sample_rate_hz: int
    requested_sample_rate_hz: int


class SynchronizedWindowSkipLedger:
    """Per-sensor, per-reason skip counters plus the last mismatch seen.

    Bounded by the number of distinct sensors ever buffered, like the
    buffer's own per-sensor state.
    """

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = {}
        self._last_sample_rate_mismatch: dict[str, SampleRateMismatch] = {}

    def record_sample_rate_mismatch(self, sensor_id: str, mismatch: SampleRateMismatch) -> None:
        reasons = self._counts.setdefault(sensor_id, {})
        reasons[SAMPLE_RATE_MISMATCH_SKIP_REASON] = reasons.get(SAMPLE_RATE_MISMATCH_SKIP_REASON, 0) + 1
        self._last_sample_rate_mismatch[sensor_id] = mismatch

    def counts_for_sensor(self, sensor_id: str) -> dict[str, int]:
        return dict(self._counts.get(sensor_id, {}))

    def counts_by_sensor(self) -> dict[str, dict[str, int]]:
        return {sensor_id: dict(reasons) for sensor_id, reasons in self._counts.items()}

    def counts_by_reason(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for reasons in self._counts.values():
            for reason, count in reasons.items():
                totals[reason] = totals.get(reason, 0) + count
        return totals

    def last_sample_rate_mismatch(self, sensor_id: str) -> SampleRateMismatch | None:
        return self._last_sample_rate_mismatch.get(sensor_id)


__all__ = ["SAMPLE_RATE_MISMATCH_SKIP_REASON", "SampleRateMismatch", "SynchronizedWindowSkipLedger"]
