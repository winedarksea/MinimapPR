"""PATCH /api/v1/config validation for the BirdNET+ V3.0 keys.

Every V3 key is restart-required, so a bad value would otherwise be accepted,
persisted, and only surface at the next startup - either as a Settings
ValueError (server will not boot) or as the BirdNET member silently dropping
out of routing with one WARNING. Checks run on the *merged* values (patch over
current settings) so cross-field rules such as ``min <= max`` hold.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from minimappr.config import BIRDNET_MODEL_VERSIONS, Settings

BIRDNET_V3_MODEL_VERSION = "3.0"


def birdnet_v3_patch_errors(coerced: dict[str, object], settings: Settings) -> list[str]:
    """Human-readable errors; normalises ``coerced["birdnet_model_version"]`` in place."""

    def merged(key: str) -> Any:
        return coerced.get(key, getattr(settings, key))

    errors: list[str] = []
    if "birdnet_model_version" in coerced:
        version = str(coerced["birdnet_model_version"]).strip()
        if version not in BIRDNET_MODEL_VERSIONS:
            errors.append(f"birdnet_model_version: must be one of {list(BIRDNET_MODEL_VERSIONS)}")
        else:
            coerced["birdnet_model_version"] = version
            if version == BIRDNET_V3_MODEL_VERSION:
                errors.extend(_missing_model_file_errors(settings))
    min_window = float(merged("birdnet_v3_min_window_seconds"))
    max_window = float(merged("birdnet_v3_max_window_seconds"))
    overlap = float(merged("birdnet_v3_window_overlap_seconds"))
    if not (math.isfinite(min_window) and math.isfinite(max_window) and 0.0 < min_window <= max_window):
        errors.append("birdnet_v3_min_window_seconds: need 0 < min <= birdnet_v3_max_window_seconds")
    if not (math.isfinite(overlap) and 0.0 <= overlap < max_window):
        errors.append("birdnet_v3_window_overlap_seconds: must be in [0, birdnet_v3_max_window_seconds)")
    if not 0.0 <= float(merged("birdnet_v3_max_nonfinite_sample_fraction")) <= 1.0:
        errors.append("birdnet_v3_max_nonfinite_sample_fraction: must be in [0, 1]")
    if not 0.0 <= float(merged("birdnet_v3_pad_noise_floor_percentile")) <= 100.0:
        errors.append("birdnet_v3_pad_noise_floor_percentile: must be in [0, 100]")
    if int(merged("birdnet_v3_num_threads")) < 1:
        errors.append("birdnet_v3_num_threads: must be >= 1")
    if "birdnet_v3_species_list_path" in coerced:
        species_list = str(coerced["birdnet_v3_species_list_path"]).strip()
        coerced["birdnet_v3_species_list_path"] = species_list
        if species_list and not Path(species_list).is_file():
            errors.append(f"birdnet_v3_species_list_path: file not found: {species_list}")
    return errors


def _missing_model_file_errors(settings: Settings) -> list[str]:
    # Model/labels paths are env-only (not PATCHable), so current settings are authoritative.
    return [
        f"birdnet_model_version: 3.0 needs {label} at {path} (run scripts/fetch_birdnet_v3.py)"
        for label, path in (
            ("model", Path(settings.birdnet_v3_model_path)),
            ("labels", Path(settings.birdnet_v3_labels_path)),
        )
        if not path.is_file()
    ]
