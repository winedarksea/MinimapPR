"""A classified render whose solve breaks the wire contract keeps its classification.

Version skew (an enum value the parser does not know without a schema bump) or
a non-finite f32 the sidecar serialized as ``null`` makes the solve unusable,
but not the render: the classification is still valid audio evidence. The
render is degraded to omni at its node with a stated fallback reason, so the
handoff counts it (``localization_drops_by_reason``) instead of losing the
whole detection.
"""

from __future__ import annotations

import logging
from typing import Any

from minimappr.api.rust_localization_solve_contract import (
    RustLocalizationSolve,
    SolveContractError,
    parse_rust_localization_solve,
)

logger = logging.getLogger(__name__)

SOLVE_CONTRACT_VIOLATION_FALLBACK_REASON = "solve_contract_violation"
# Distinct messages logged so far; capped so a sidecar emitting ever-new
# messages cannot grow it without bound (later ones are still counted).
_MAX_DISTINCT_LOGGED_SOLVE_CONTRACT_ERRORS = 256
_logged_solve_contract_errors: set[str] = set()


def parse_rust_localization_solve_or_degrade(
    localization_payload: Any,
    *,
    manifest_id: str,
) -> tuple[RustLocalizationSolve | None, str | None]:
    """``(solve, None)``, or ``(None, SOLVE_CONTRACT_VIOLATION_FALLBACK_REASON)``."""
    try:
        return parse_rust_localization_solve(localization_payload), None
    except SolveContractError as exc:
        _log_once_per_distinct_error(str(exc), manifest_id=manifest_id)
        return None, SOLVE_CONTRACT_VIOLATION_FALLBACK_REASON


def _log_once_per_distinct_error(message: str, *, manifest_id: str) -> None:
    if message in _logged_solve_contract_errors:
        return
    if len(_logged_solve_contract_errors) >= _MAX_DISTINCT_LOGGED_SOLVE_CONTRACT_ERRORS:
        return
    _logged_solve_contract_errors.add(message)
    logger.warning(
        "Rust solve breaks the localization solve contract (%s, first seen on manifest %s); "
        "render degraded to omni, classification kept",
        message,
        manifest_id,
    )
