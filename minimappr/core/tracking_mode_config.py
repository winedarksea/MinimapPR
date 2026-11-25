"""Per-category tracking modes: which dynamics a track runs, or none at all.

The Settings key ``tracking_mode_by_category`` maps a label category to a mode,
with a ``default`` entry for every category not listed:

- ``continuous_motion``: the 6D constant-velocity filter (drones, vehicles,
  people). A track keeps moving between fixes.
- ``stationary_hold``: the perched random-walk filter with the relocation gate
  and concurrent-perch guard (``core/track_filters_perched.py``). A singing
  bird sits still between bouts, so a constant-velocity filter would read fix
  jitter as velocity and coast the bird off its perch.
- ``none``: the category is not tracked; its fixes never start or feed a track.

A track without a label (category ``unknown``) always uses
``continuous_motion``: it is hidden until classified anyway, and the label that
arrives later re-routes it. So ``unknown`` may not be configured.

Rust mirror: ``minimappr-core/src/wildlife_tracking/tracking_mode.rs``; parity
case in ``tests/wildlife_tracking_parity_cases.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping

from minimappr.core.taxonomy import label_category_for_name


class TrackingMode(str, Enum):
    CONTINUOUS_MOTION = "continuous_motion"
    STATIONARY_HOLD = "stationary_hold"
    NONE = "none"


TRACKING_MODE_NAMES = frozenset(mode.value for mode in TrackingMode)
# The mapping entry that applies to every category not listed explicitly.
TRACKING_MODE_DEFAULT_CATEGORY_KEY = "default"
# The category of a track that has no label yet.
UNLABELED_TRACK_CATEGORY = "unknown"
UNLABELED_TRACK_MODE = TrackingMode.CONTINUOUS_MOTION


def default_tracking_mode_by_category() -> dict[str, str]:
    """Shipped defaults: birds hold position, everything else keeps moving."""
    return {
        "wildlife": TrackingMode.STATIONARY_HOLD.value,
        TRACKING_MODE_DEFAULT_CATEGORY_KEY: TrackingMode.CONTINUOUS_MOTION.value,
    }


def tracking_mode_by_category_with(category: str, mode: TrackingMode) -> dict[str, str]:
    """The shipped defaults with one category's mode replaced (sim and test configurations)."""
    return {**default_tracking_mode_by_category(), category: mode.value}


def _normalised_category(category: str | None) -> str:
    return (category or "").strip().lower()


def tracking_mode_by_category_errors(mode_by_category: object) -> list[str]:
    """Validation errors for a ``tracking_mode_by_category`` value."""
    if not isinstance(mode_by_category, dict):
        return ["tracking_mode_by_category must be a mapping of label category -> tracking mode"]
    errors: list[str] = []
    if TRACKING_MODE_DEFAULT_CATEGORY_KEY not in mode_by_category:
        errors.append(f"tracking_mode_by_category must include a {TRACKING_MODE_DEFAULT_CATEGORY_KEY!r} entry")
    for category, mode in mode_by_category.items():
        if not isinstance(category, str) or not _normalised_category(category):
            errors.append(f"tracking_mode_by_category has an empty or non-string category {category!r}")
        elif _normalised_category(category) == UNLABELED_TRACK_CATEGORY:
            errors.append(
                f"tracking_mode_by_category may not configure {UNLABELED_TRACK_CATEGORY!r}: "
                f"unlabeled tracks always use {UNLABELED_TRACK_MODE.value}"
            )
        if mode not in TRACKING_MODE_NAMES:
            errors.append(
                f"tracking_mode_by_category[{category!r}] must be one of {sorted(TRACKING_MODE_NAMES)}; got {mode!r}"
            )
    return errors


@dataclass(frozen=True, slots=True)
class TrackingModeConfig:
    """Resolved ``tracking_mode_by_category``: category or label -> mode."""

    modes_by_category: Mapping[str, TrackingMode]
    default_mode: TrackingMode
    # Label -> category for callers that only hold a label (the taxonomy default).
    label_category_resolver: Callable[[str], str] = field(default=label_category_for_name, compare=False)

    @classmethod
    def from_mapping(
        cls,
        mode_by_category: Mapping[str, str] | None,
        *,
        label_category_resolver: Callable[[str], str] | None = None,
    ) -> TrackingModeConfig:
        """Build from a Settings value; raises ``ValueError`` on an invalid one."""
        raw = default_tracking_mode_by_category() if mode_by_category is None else dict(mode_by_category)
        errors = tracking_mode_by_category_errors(raw)
        if errors:
            raise ValueError("; ".join(errors))
        modes = {
            _normalised_category(category): TrackingMode(mode)
            for category, mode in raw.items()
            if category != TRACKING_MODE_DEFAULT_CATEGORY_KEY
        }
        return cls(
            modes_by_category=MappingProxyType(modes),
            default_mode=TrackingMode(raw[TRACKING_MODE_DEFAULT_CATEGORY_KEY]),
            label_category_resolver=label_category_resolver or label_category_for_name,
        )

    @classmethod
    def shipped_defaults(cls) -> TrackingModeConfig:
        return cls.from_mapping(default_tracking_mode_by_category())

    def mode_for_category(self, category: str | None) -> TrackingMode:
        normalised = _normalised_category(category)
        if not normalised or normalised == UNLABELED_TRACK_CATEGORY:
            return UNLABELED_TRACK_MODE
        return self.modes_by_category.get(normalised, self.default_mode)

    def mode_for_label(self, label: str | None) -> TrackingMode:
        if not (label or "").strip():
            return UNLABELED_TRACK_MODE
        return self.mode_for_category(self.label_category_resolver(label))

    def tracks_category(self, category: str | None) -> bool:
        return self.mode_for_category(category) is not TrackingMode.NONE

    def holds_station(self, category: str | None) -> bool:
        return self.mode_for_category(category) is TrackingMode.STATIONARY_HOLD

    def any_category_uses(self, mode: TrackingMode) -> bool:
        """Whether some labelled category resolves to *mode* (incl. the default)."""
        return self.default_mode is mode or any(value is mode for value in self.modes_by_category.values())
