"""Operational constants for point-network clip classification (plan item P7).

Only the on/off switch (``point_network_clip_classification_enabled``) and the
distinct-caller separation are operator Settings; the latter reuses
``localized_handoff_duplicate_merge_max_separation_m``, the distance beyond
which two solves already count as different callers for duplicate merging.
The DSP constants below are a domain sub-config with defaults, not Settings
keys: they shape a measurement, not a site policy.
"""

from __future__ import annotations

from dataclasses import dataclass

# In-band SNR band. Wide enough for bird, voice and drone energy, and high
# enough to keep wind rumble (which dominates point-node noise floors) out of
# the clearest-node choice. The upper edge is capped at Nyquist per call.
DEFAULT_SNR_BAND_LOW_HZ = 300.0
DEFAULT_SNR_BAND_HIGH_HZ = 10_000.0
# Frame length for framed in-band energy: short enough to resolve a single
# note inside a localization window, long enough for a stable energy estimate.
DEFAULT_ENERGY_FRAME_SECONDS = 0.01
# Signal level = this quantile of the event span's frame energies (a peak that
# one click cannot set); noise floor = the low quantile over the whole clip.
DEFAULT_SIGNAL_ENERGY_QUANTILE = 0.9
DEFAULT_NOISE_FLOOR_ENERGY_QUANTILE = 0.2
# Another caller whose level in the clip is within this margin of the call's
# own level could own the label as much as the call does.
DEFAULT_OVERLAPPING_CALLER_COMPARABLE_LEVEL_DB = 6.0
# ...but only if it is audible at all: a span that is just the noise floor is
# not a competing caller, even when the call itself is faint.
DEFAULT_OVERLAPPING_CALLER_MIN_AUDIBLE_SNR_DB = 6.0
# Localized calls are remembered long enough to overlap any later clip
# (a classification window plus the largest cross-node lag), bounded in count.
DEFAULT_LOCALIZED_CALL_RETENTION_SECONDS = 120.0
DEFAULT_LOCALIZED_CALL_MAX_RECORDS = 512


@dataclass(frozen=True, slots=True)
class PointNetworkClipConfig:
    enabled: bool
    distinct_caller_min_separation_m: float
    snr_band_low_hz: float = DEFAULT_SNR_BAND_LOW_HZ
    snr_band_high_hz: float = DEFAULT_SNR_BAND_HIGH_HZ
    energy_frame_seconds: float = DEFAULT_ENERGY_FRAME_SECONDS
    signal_energy_quantile: float = DEFAULT_SIGNAL_ENERGY_QUANTILE
    noise_floor_energy_quantile: float = DEFAULT_NOISE_FLOOR_ENERGY_QUANTILE
    overlapping_caller_comparable_level_db: float = DEFAULT_OVERLAPPING_CALLER_COMPARABLE_LEVEL_DB
    overlapping_caller_min_audible_snr_db: float = DEFAULT_OVERLAPPING_CALLER_MIN_AUDIBLE_SNR_DB
    localized_call_retention_seconds: float = DEFAULT_LOCALIZED_CALL_RETENTION_SECONDS
    localized_call_max_records: int = DEFAULT_LOCALIZED_CALL_MAX_RECORDS


def point_network_clip_config_from_settings(settings) -> PointNetworkClipConfig:
    return PointNetworkClipConfig(
        enabled=bool(settings.point_network_clip_classification_enabled),
        distinct_caller_min_separation_m=float(settings.localized_handoff_duplicate_merge_max_separation_m),
    )


__all__ = ["PointNetworkClipConfig", "point_network_clip_config_from_settings"]
