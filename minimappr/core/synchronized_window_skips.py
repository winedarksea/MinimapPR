"""Why a synchronized buffer read leaves a sensor out (plan item P8).

``MultiSensorBuffer`` builds synchronized windows only from sensors buffered
at the requested rate; mismatched-rate sensors stay excluded (guard now,
resample later). The reason spelling matches the Rust solve contract's
``excluded_sensors`` reason, so both ingest paths read the same. Exclusions
are counted once per solve by ``python_ingest_solve_exclusions``.
"""

from __future__ import annotations

from dataclasses import dataclass

from minimappr.api.rust_localization_solve_contract import SolveExclusionReason

SAMPLE_RATE_MISMATCH_SKIP_REASON = SolveExclusionReason.SAMPLE_RATE_MISMATCH.value


@dataclass(frozen=True, slots=True)
class SampleRateMismatch:
    buffered_sample_rate_hz: int
    requested_sample_rate_hz: int


__all__ = ["SAMPLE_RATE_MISMATCH_SKIP_REASON", "SampleRateMismatch"]
