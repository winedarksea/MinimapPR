"""Simulator scenarios whose sources are the committed field-audio fixtures.

Used by W3-birdnet-cutover to compare BirdNET v2.4 and V3.0 on the *same*
audio after realistic propagation (1/r, ISO 9613 absorption, ground and trunk
echoes, ambient noise), rendered by the Rust ``minimappr-sim`` binary:

* ``build_distance_ladder_scenario``: one tetra node; every bird fixture is
  played once per distance in a ladder (2 m ... 250 m), one call after the
  other, then a noise-only tail. Gives positives across the SNR range a
  deployment sees, plus negatives, for score-distribution matching.
* ``build_species_gate_scenario``: three tetra nodes (the 20 m triangle of
  ``synthetic_entity_scenarios``) with birds inside/near the array, for the
  full-pipeline replay gate (entity metrics + exact label accuracy).

Scenarios are plain dicts serialised with a minimal TOML emitter (the repo has
no TOML writer dependency; the subset needed is tables, arrays of tables,
scalars, arrays and inline tables).
"""

from __future__ import annotations

import math
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIELD_AUDIO_DIRECTORY = REPOSITORY_ROOT / "tests" / "fixtures" / "field_audio"
FIELD_AUDIO_MANIFEST_PATH = FIELD_AUDIO_DIRECTORY / "manifest.toml"
DEFAULT_SIM_BINARY_PATH = REPOSITORY_ROOT / "target" / "release" / "minimappr-sim"

SIM_SAMPLE_RATE_HZ = 48_000
SIM_SITE = {
    "origin_lat": 45.0,
    "origin_lon": -93.0,
    "origin_alt_m": 250.0,
    "start_time_ns": 1_763_700_000_000_000_000,
}
SIM_ENVIRONMENT = {"temperature_c": 18.0, "humidity_fraction": 0.6}
GROUND_REFLECTION_COEFFICIENT = 0.6
RANDOM_TRUNK_COUNT_RANGE = [2, 4]
# Quiet rural/forest background (sim default is 35 dB SPL).
AMBIENT_NOISE_LEVEL_DB_SPL = 35.0
# Typical songbird song level at 1 m; fixtures are normalised to unit active RMS.
SONGBIRD_LEVEL_DB_SPL_AT_1M = 90.0
SOURCE_HEIGHT_M = 3.0
NODE_HEIGHT_M = 1.5
DISTANCE_LADDER_M: tuple[float, ...] = (2.0, 10.0, 30.0, 60.0, 120.0, 250.0)
SILENCE_BETWEEN_CALLS_S = 1.5
NOISE_TAIL_SECONDS = 30.0
# Golden-angle bearing step so successive ladder sources do not share a bearing.
GOLDEN_ANGLE_DEG = 137.50776405
SPECIES_GATE_NODE_POSITIONS_M: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, NODE_HEIGHT_M),
    (20.0, 0.0, NODE_HEIGHT_M),
    (10.0, 18.0, NODE_HEIGHT_M),
)
# Bird perches inside / just outside the triangle (ENU metres).
SPECIES_GATE_PERCHES_M: tuple[tuple[float, float, float], ...] = (
    (6.0, 6.0, SOURCE_HEIGHT_M),
    (14.0, 7.0, SOURCE_HEIGHT_M),
    (10.0, 13.0, SOURCE_HEIGHT_M),
    (-6.0, 4.0, SOURCE_HEIGHT_M),
    (26.0, 8.0, SOURCE_HEIGHT_M),
    (10.0, 26.0, SOURCE_HEIGHT_M),
    (4.0, -8.0, SOURCE_HEIGHT_M),
    (18.0, 16.0, SOURCE_HEIGHT_M),
)


@dataclass(frozen=True, slots=True)
class FixtureBirdClip:
    file_path: Path
    species: str
    scientific_name: str
    duration_s: float

    @property
    def pipeline_label(self) -> str:
        """BirdNET pipeline label convention: lower-cased English common name."""
        return self.species.strip().lower()

    @property
    def short_code(self) -> str:
        return self.file_path.name.split("_", 1)[0]


def load_fixture_bird_clips(manifest_path: Path = FIELD_AUDIO_MANIFEST_PATH) -> list[FixtureBirdClip]:
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        FixtureBirdClip(
            file_path=manifest_path.parent / clip["file"],
            species=clip["species"],
            scientific_name=clip["scientific_name"],
            duration_s=float(clip["duration_s"]),
        )
        for clip in manifest["clip"]
        if clip.get("role") == "bird"
    ]


def _base_scenario(name: str, seed: int, duration_s: float) -> dict[str, Any]:
    return {
        "name": name,
        "seed": seed,
        "duration_s": round(duration_s, 3),
        "sample_rate_hz": SIM_SAMPLE_RATE_HZ,
        "site": dict(SIM_SITE),
        "environment": dict(SIM_ENVIRONMENT),
        "ground": {"enabled": True, "reflection_coefficient": GROUND_REFLECTION_COEFFICIENT},
        "reflectors": {"random_trunks": {"count_range": list(RANDOM_TRUNK_COUNT_RANGE)}},
        "ambient_noise": {"level_db_spl": AMBIENT_NOISE_LEVEL_DB_SPL},
        "devices": [],
        "sources": [],
    }


def _bird_source(
    clip: FixtureBirdClip,
    individual_id: str,
    position_m: tuple[float, float, float],
    call_start_times_s: list[float],
) -> dict[str, Any]:
    return {
        "individual_id": individual_id,
        "species": clip.species,
        # Scored labels: the lower-cased pipeline label is the "true" entry and
        # the species name a candidate, so `match = "exact"` accepts either.
        "label": clip.pipeline_label,
        "position_m": [round(value, 3) for value in position_m],
        "calls": [
            {
                "start_s": round(start_s, 3),
                "level_db_spl_at_1m": SONGBIRD_LEVEL_DB_SPL_AT_1M,
                "generator": {"kind": "wav_file", "path": str(clip.file_path)},
            }
            for start_s in call_start_times_s
        ],
    }


def build_distance_ladder_scenario(
    clips: list[FixtureBirdClip],
    *,
    seed: int,
    distances_m: tuple[float, ...] = DISTANCE_LADDER_M,
) -> dict[str, Any]:
    """One tetra node at the origin; each clip once per distance, then noise."""
    sources: list[dict[str, Any]] = []
    cursor_s = SILENCE_BETWEEN_CALLS_S
    source_index = 0
    for clip in clips:
        for distance_m in distances_m:
            bearing_rad = math.radians(source_index * GOLDEN_ANGLE_DEG)
            position = (
                distance_m * math.cos(bearing_rad),
                distance_m * math.sin(bearing_rad),
                SOURCE_HEIGHT_M,
            )
            individual_id = f"{clip.short_code}-{int(distance_m)}m"
            sources.append(_bird_source(clip, individual_id, position, [cursor_s]))
            # Leave room for the propagation delay of the next, farther call.
            cursor_s += clip.duration_s + SILENCE_BETWEEN_CALLS_S
            source_index += 1
    scenario = _base_scenario(f"birdnet_distance_ladder_s{seed}", seed, cursor_s + NOISE_TAIL_SECONDS)
    scenario["devices"] = [
        {"device_id": "tetra-a", "kind": "sirith_tetra", "position_m": [0.0, 0.0, NODE_HEIGHT_M]}
    ]
    scenario["sources"] = sources
    scenario["noise_tail_start_s"] = cursor_s  # read back by callers; stripped before TOML
    return scenario


def build_species_gate_scenario(
    clips: list[FixtureBirdClip],
    *,
    seed: int,
    bouts_per_bird: int = 2,
) -> dict[str, Any]:
    """Three tetra nodes; each fixture bird sings ``bouts_per_bird`` times from one perch."""
    sources: list[dict[str, Any]] = []
    cursor_s = SILENCE_BETWEEN_CALLS_S
    starts_by_clip: list[list[float]] = [[] for _ in clips]
    for _ in range(bouts_per_bird):
        for index, clip in enumerate(clips):
            starts_by_clip[index].append(cursor_s)
            cursor_s += clip.duration_s + SILENCE_BETWEEN_CALLS_S
    for index, clip in enumerate(clips):
        perch = SPECIES_GATE_PERCHES_M[index % len(SPECIES_GATE_PERCHES_M)]
        sources.append(_bird_source(clip, f"{clip.short_code}-1", perch, starts_by_clip[index]))
    scenario = _base_scenario(f"birdnet_species_gate_s{seed}", seed, cursor_s + SILENCE_BETWEEN_CALLS_S)
    scenario["devices"] = [
        {"device_id": f"tetra-{chr(ord('a') + index)}", "kind": "sirith_tetra", "position_m": list(position)}
        for index, position in enumerate(SPECIES_GATE_NODE_POSITIONS_M)
    ]
    scenario["sources"] = sources
    return scenario


# ── minimal TOML emitter ────────────────────────────────────────────────────

def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {_toml_scalar(item)}" for key, item in value.items()) + " }"
    raise TypeError(f"unsupported TOML value {value!r}")


def _is_array_of_tables(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, dict) for item in value)


def _emit_table(lines: list[str], table: dict[str, Any], prefix: str) -> None:
    nested_tables, arrays_of_tables = [], []
    for key, value in table.items():
        if _is_array_of_tables(value):
            arrays_of_tables.append((key, value))
        elif isinstance(value, dict) and not (prefix.endswith("generator") or key == "generator"):
            nested_tables.append((key, value))
        else:
            lines.append(f"{key} = {_toml_scalar(value)}")
    for key, value in nested_tables:
        lines.append("")
        lines.append(f"[{prefix}{key}]")
        _emit_table(lines, value, f"{prefix}{key}.")
    for key, items in arrays_of_tables:
        for item in items:
            lines.append("")
            lines.append(f"[[{prefix}{key}]]")
            _emit_table(lines, item, f"{prefix}{key}.")


def scenario_to_toml(scenario: dict[str, Any]) -> str:
    """Serialise a scenario dict; keys the simulator does not know are dropped."""
    emitted = {key: value for key, value in scenario.items() if key != "noise_tail_start_s"}
    lines: list[str] = []
    _emit_table(lines, emitted, "")
    return "\n".join(lines) + "\n"


def render_scenario_bundle(
    scenario: dict[str, Any],
    output_zip_path: Path,
    *,
    sim_binary_path: Path = DEFAULT_SIM_BINARY_PATH,
    render_timeout_seconds: float = 600.0,
) -> Path:
    """Write the scenario TOML next to the zip and render it with ``minimappr-sim``."""
    if not sim_binary_path.exists():
        raise FileNotFoundError(
            f"{sim_binary_path} missing; build it with `cargo build --release -p minimappr-sim`"
        )
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path = output_zip_path.with_suffix(".toml")
    scenario_path.write_text(scenario_to_toml(scenario), encoding="utf-8")
    subprocess.run(
        [str(sim_binary_path), "render-bundle", "--scenario", str(scenario_path), "--out", str(output_zip_path), "--stored"],
        check=True,
        capture_output=True,
        timeout=render_timeout_seconds,
    )
    return output_zip_path
