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

POSITION_OBSERVATION_P5_SETTING_KEYS: tuple[str, ...] = ("position_observation_solve_ledger_max_entries",)


@dataclass(frozen=True, slots=True)
class PositionObservationConfig:
    queue_max_items: int
    solve_ledger_max_entries: int
    unclassified_track_policy: str


def position_observation_config_from_settings(settings) -> PositionObservationConfig:
    return PositionObservationConfig(
        queue_max_items=int(settings.position_observation_queue_max_items),
        solve_ledger_max_entries=int(settings.position_observation_solve_ledger_max_entries),
        unclassified_track_policy=str(settings.position_observation_unclassified_track_policy),
    )


def position_observation_setting_errors(settings) -> list[str]:
    errors: list[str] = []
    ledger_max_entries = settings.position_observation_solve_ledger_max_entries
    if not isinstance(ledger_max_entries, int) or isinstance(ledger_max_entries, bool) or ledger_max_entries < 1:
        errors.append("position_observation_solve_ledger_max_entries must be an integer >= 1")
    elif ledger_max_entries < settings.position_observation_queue_max_items:
        errors.append(
            "position_observation_solve_ledger_max_entries must be >= position_observation_queue_max_items "
            "(a queued solve must keep its idempotency record)"
        )
    return errors


__all__ = [
    "DEFAULT_POSITION_OBSERVATION_SOLVE_LEDGER_MAX_ENTRIES",
    "POSITION_OBSERVATION_P5_SETTING_KEYS",
    "PositionObservationConfig",
    "position_observation_config_from_settings",
    "position_observation_setting_errors",
]
