"""Settings for the one-observation-per-solve tracker feed (plan item P5).

``position_observation_queue_max_items`` and
``position_observation_unclassified_track_policy`` were declared by P0 in
``localization_overhaul_config.py``; this module owns the P5-only keys, their
validation, and the typed view the observation pipeline reads. It imports
nothing from ``localization_overhaul_config`` so that module can register these
keys on the flat config surface without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

# The ledger remembers which track each solve updated until that solve's
# classification arrives (possibly seconds later, behind a classification
# backlog). It must outlive the FIFO: an entry evicted while still queued would
# lose its idempotency record, so validation requires ledger >= queue.
DEFAULT_POSITION_OBSERVATION_SOLVE_LEDGER_MAX_ENTRIES = 8192
# Solve ids remembered after their full ledger entry is evicted (id -> track id
# only, ~150 B each), so a covering render or a classification that arrives
# after eviction is still recognised and never applies the position twice.
# Sizing: it must cover the longest render-interval x event-rate (or classifier
# backlog) span; 32768 ids last ~34 min at 16 event solves per second.
DEFAULT_POSITION_OBSERVATION_SOLVE_ID_MEMORY_MAX_ENTRIES = 32768
# A fix held for its classification (it gates only a labelled stationary_hold
# track) is applied geometry-only if no classification resolves it in this
# long. It must exceed the classifier render interval plus classification
# latency, or a covered event's label would arrive after its fix was released.
DEFAULT_POSITION_OBSERVATION_HELD_FIX_MAX_WAIT_SECONDS = 30.0

POSITION_OBSERVATION_P5_SETTING_KEYS: tuple[str, ...] = (
    "position_observation_solve_ledger_max_entries",
    "position_observation_solve_id_memory_max_entries",
    "position_observation_held_fix_max_wait_seconds",
)


@dataclass(frozen=True, slots=True)
class PositionObservationConfig:
    queue_max_items: int
    solve_ledger_max_entries: int
    unclassified_track_policy: str
    solve_id_memory_max_entries: int = DEFAULT_POSITION_OBSERVATION_SOLVE_ID_MEMORY_MAX_ENTRIES
    held_fix_max_wait_seconds: float = DEFAULT_POSITION_OBSERVATION_HELD_FIX_MAX_WAIT_SECONDS


def position_observation_config_from_settings(settings) -> PositionObservationConfig:
    return PositionObservationConfig(
        queue_max_items=int(settings.position_observation_queue_max_items),
        solve_ledger_max_entries=int(settings.position_observation_solve_ledger_max_entries),
        unclassified_track_policy=str(settings.position_observation_unclassified_track_policy),
        solve_id_memory_max_entries=int(settings.position_observation_solve_id_memory_max_entries),
        held_fix_max_wait_seconds=float(settings.position_observation_held_fix_max_wait_seconds),
    )


def _is_strict_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def position_observation_setting_errors(settings) -> list[str]:
    errors: list[str] = []
    ledger_max_entries = settings.position_observation_solve_ledger_max_entries
    if not _is_strict_int(ledger_max_entries) or ledger_max_entries < 1:
        errors.append("position_observation_solve_ledger_max_entries must be an integer >= 1")
    elif ledger_max_entries < settings.position_observation_queue_max_items:
        errors.append(
            "position_observation_solve_ledger_max_entries must be >= position_observation_queue_max_items "
            "(a queued solve must keep its idempotency record)"
        )
    memory_max_entries = settings.position_observation_solve_id_memory_max_entries
    if not _is_strict_int(memory_max_entries) or (
        _is_strict_int(ledger_max_entries) and memory_max_entries < ledger_max_entries
    ):
        errors.append(
            "position_observation_solve_id_memory_max_entries must be an integer >= "
            "position_observation_solve_ledger_max_entries (it outlives evicted ledger entries)"
        )
    held_wait_seconds = settings.position_observation_held_fix_max_wait_seconds
    if isinstance(held_wait_seconds, bool) or not isinstance(held_wait_seconds, (int, float)) or held_wait_seconds <= 0:
        errors.append("position_observation_held_fix_max_wait_seconds must be a number > 0")
    return errors


__all__ = [
    "DEFAULT_POSITION_OBSERVATION_HELD_FIX_MAX_WAIT_SECONDS",
    "DEFAULT_POSITION_OBSERVATION_SOLVE_ID_MEMORY_MAX_ENTRIES",
    "DEFAULT_POSITION_OBSERVATION_SOLVE_LEDGER_MAX_ENTRIES",
    "POSITION_OBSERVATION_P5_SETTING_KEYS",
    "PositionObservationConfig",
    "position_observation_config_from_settings",
    "position_observation_setting_errors",
]
