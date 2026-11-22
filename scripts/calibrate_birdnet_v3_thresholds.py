#!/usr/bin/env python3
"""Derive V3.0 equivalents of the v2.4 BirdNET thresholds (W3-birdnet-cutover).

Both models score the same clips; every v2.4 threshold is carried to V3 by
quantile matching (``minimappr/classifiers/birdnet_confidence_calibration.py``).
Clip sources:

* ``fixture``: the committed field-audio bird clips, whole, and the 30 s
  ambience bed in 3 s pieces (unlabelled: it holds faint real birds);
* ``sim``: distance-ladder bundles rendered by ``minimappr-sim`` from the same
  fixtures (2-250 m, ground + trunk echoes, 35 dB SPL ambient), one window per
  call on the node's channel 0, plus noise-only windows from the tail;
* ``local``: any extra folders passed with ``--local-audio NAME=DIR[:MAX]``
  (unlabelled real audio; ``--local-species NAME=Scientific name`` labels a
  folder whose clips are all one species).

Three matchings are reported per threshold: pooled clip top-1 (primary,
keeps the pass fraction), labelled true-class score (keeps recall on known
birds) and negative-only top-1 (keeps the false-positive rate). The chosen
V3 value is the pooled match; the others show how far the operating point
would move under a different criterion.

    python scripts/calibrate_birdnet_v3_thresholds.py --seeds 0 1 \\
        --local-audio site_ambient=../drone_dataset/wav/ambient:150 \\
        --output-json data/birdnet_v3_threshold_calibration.json --write-map
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from compare_birdnet_versions import run_v24_model, run_v3_model  # noqa: E402
from quantize_birdnet_v3 import collect_audio_files, load_mono_audio  # noqa: E402

from minimappr.calibration.bundle import load_bundle  # noqa: E402
from minimappr.classifiers.birdnet_confidence_calibration import (  # noqa: E402
    DEFAULT_V24_TO_V3_CONFIDENCE_MAP_PATH,
    bootstrap_matched_threshold_interval,
    build_confidence_quantile_map,
    matched_threshold,
    write_confidence_quantile_map,
)
from minimappr.classifiers.birdnet_v3_audio import BirdNETV3WindowPolicy  # noqa: E402
from minimappr.classifiers.birdnet_v3_labels import load_birdnet_v3_labels  # noqa: E402
from minimappr.sim.fixture_audio_scenarios import (  # noqa: E402
    FIELD_AUDIO_DIRECTORY,
    build_distance_ladder_scenario,
    load_fixture_bird_clips,
    render_scenario_bundle,
)

logger = logging.getLogger("calibrate_birdnet_v3_thresholds")

MODEL_DIRECTORY = REPOSITORY_ROOT / "data/models/birdnet_v3"
V3_MODEL_PATH = MODEL_DIRECTORY / "BirdNET+_V3.0_Global_12K_FP16_pruned.onnx"
V3_LABELS_PATH = MODEL_DIRECTORY / "BirdNET+_V3.0_Global_12K_Labels.csv"
AMBIENCE_FIXTURE = FIELD_AUDIO_DIRECTORY / "ambience_forest_breeze_xc980972.wav"
NEGATIVE_WINDOW_SECONDS = 3.0
CALL_WINDOW_PADDING_SECONDS = 0.5
SPEED_OF_SOUND_FALLBACK_MPS = 343.0
# v2.4 thresholds carried across, with where each lives.
V24_THRESHOLDS: dict[str, float] = {
    "birdnet_trigger_min_confidence (display floor, retention confident bucket)": 0.50,
    "tracking_species_display_min_belief": 0.50,
    "routing classifiers.birdnet.min_confidence": 0.40,
    "birdnet_localize_to_track_min_confidence": 0.15,
    "tracking_species_log_odds_prior_probability": 0.10,
}
BOOTSTRAP_RESAMPLES = 500
BOOTSTRAP_INTERVAL_FRACTION = 0.9
BOOTSTRAP_SEED = 20261122
SOURCE_KIND_FIXTURE, SOURCE_KIND_SIM, SOURCE_KIND_LOCAL = "fixture", "sim", "local"


@dataclass
class CalibrationClip:
    name: str
    audio: np.ndarray
    sample_rate_hz: int
    source_kind: str
    true_scientific_name: str | None  # None = unlabelled; "" = known negative
    distance_m: float | None = None


def _chop(audio: np.ndarray, sample_rate_hz: int, window_seconds: float) -> list[np.ndarray]:
    step = int(window_seconds * sample_rate_hz)
    return [audio[start : start + step] for start in range(0, audio.size - step + 1, step)]


def fixture_clips() -> list[CalibrationClip]:
    clips = []
    for bird in load_fixture_bird_clips():
        audio, rate = load_mono_audio(bird.file_path)
        clips.append(CalibrationClip(bird.file_path.name, audio, rate, SOURCE_KIND_FIXTURE, bird.scientific_name))
    audio, rate = load_mono_audio(AMBIENCE_FIXTURE)
    for index, piece in enumerate(_chop(audio, rate, NEGATIVE_WINDOW_SECONDS)):
        # Unlabelled, not negative: the bed keeps sparse distant birds that V3
        # (level-invariant) picks up at 0.5-0.7 while v2.4 stays under 0.25.
        clips.append(CalibrationClip(f"ambience-{index}", piece, rate, SOURCE_KIND_FIXTURE, None))
    return clips


def sim_ladder_clips(seed: int, work_directory: Path) -> list[CalibrationClip]:
    birds = load_fixture_bird_clips()
    scientific_by_label = {bird.pipeline_label: bird.scientific_name for bird in birds}
    scenario = build_distance_ladder_scenario(birds, seed=seed)
    bundle = load_bundle(render_scenario_bundle(scenario, work_directory / f"ladder-{seed}.zip"))
    node = bundle.manifest["nodes"][0]
    audio, rate = bundle.node_audio(node["node_id"])
    channel = audio[0].astype(np.float32)
    start_ns = int(node["audio_start_time_ns"])
    node_position = np.asarray(node["position_m"], dtype=np.float64)
    speed_of_sound = float(bundle.manifest.get("environment", {}).get("speed_of_sound_mps", SPEED_OF_SOUND_FALLBACK_MPS))
    clips = []
    for event in bundle.events:
        if event.get("is_echo_source"):
            continue
        distance = float(np.linalg.norm(np.asarray(event["position_m"]) - node_position))
        arrival_offset_s = (int(event["start_ns"]) - start_ns) / 1e9 + distance / speed_of_sound
        duration_s = (int(event["end_ns"]) - int(event["start_ns"])) / 1e9
        first = max(0, int((arrival_offset_s - CALL_WINDOW_PADDING_SECONDS) * rate))
        last = min(channel.size, int((arrival_offset_s + duration_s + CALL_WINDOW_PADDING_SECONDS) * rate))
        clips.append(
            CalibrationClip(
                f"sim{seed}-{event['individual_id']}", channel[first:last], rate, SOURCE_KIND_SIM,
                scientific_by_label[event["label"]], round(distance),
            )
        )
    tail = channel[int(scenario["noise_tail_start_s"] * rate) + rate :]
    for index, piece in enumerate(_chop(tail, rate, NEGATIVE_WINDOW_SECONDS)):
        clips.append(CalibrationClip(f"sim{seed}-noise-{index}", piece, rate, SOURCE_KIND_SIM, ""))
    return clips


def local_clips(spec: str, species_by_name: dict[str, str], seed: int) -> list[CalibrationClip]:
    name, _, location = spec.partition("=")
    directory_text, _, limit_text = location.rpartition(":")
    if not limit_text.isdigit():
        directory_text, limit_text = location, ""
    directory = Path(directory_text)
    files = collect_audio_files([directory])
    if limit_text:
        generator = np.random.default_rng(seed)
        files = [files[i] for i in sorted(generator.choice(len(files), min(int(limit_text), len(files)), replace=False))]
    clips = []
    for path in files:
        try:
            audio, rate = load_mono_audio(path)
        except Exception as error:  # noqa: BLE001 - skip undecodable files
            logger.warning("skip %s: %s", path, error)
            continue
        clips.append(CalibrationClip(f"{name}/{path.name}", audio, rate, SOURCE_KIND_LOCAL, species_by_name.get(name)))
    return clips


def _class_index(names: list[str], scientific_name: str) -> int | None:
    folded = scientific_name.casefold()
    for index, name in enumerate(names):
        if name.split("_", 1)[0].casefold() == folded:
            return index
    return None


def _match_row(v24: np.ndarray, v3: np.ndarray, threshold: float, with_interval: bool) -> dict:
    row = {
        "clips": int(v24.size),
        "v24_pass_fraction": float(np.mean(v24 >= threshold)) if v24.size else None,
        "v3_threshold": matched_threshold(v24, v3, threshold) if v24.size else None,
    }
    if v24.size:
        row["v3_pass_fraction_at_v3_threshold"] = float(np.mean(v3 >= row["v3_threshold"]))
    if with_interval and v24.size:
        row["v3_threshold_interval_90"] = bootstrap_matched_threshold_interval(
            v24, v3, threshold, resample_count=BOOTSTRAP_RESAMPLES,
            interval_fraction=BOOTSTRAP_INTERVAL_FRACTION, seed=BOOTSTRAP_SEED,
        )
    return row


def analyse(clips: list[CalibrationClip], v24_probabilities, v24_species, v3_probabilities, v3_labels) -> dict:
    v3_names = [f"{label.scientific_name}_{label.common_name_en}" for label in v3_labels]
    v24_top = np.asarray([p.max() for p in v24_probabilities])
    v3_top = np.asarray([p.max() for p in v3_probabilities])
    labelled = [i for i, clip in enumerate(clips) if clip.true_scientific_name]
    negatives = [i for i, clip in enumerate(clips) if clip.true_scientific_name == ""]
    v24_true, v3_true, per_clip = [], [], []
    for i in labelled:
        clip = clips[i]
        v24_index = _class_index(v24_species, clip.true_scientific_name)
        v3_index = _class_index(v3_names, clip.true_scientific_name)
        if v24_index is None or v3_index is None:
            continue
        v24_true.append(float(v24_probabilities[i][v24_index]))
        v3_true.append(float(v3_probabilities[i][v3_index]))
        per_clip.append({
            "clip": clip.name, "source": clip.source_kind, "distance_m": clip.distance_m,
            "v24_true_score": v24_true[-1], "v3_true_score": v3_true[-1],
            "v24_top1_correct": int(np.argmax(v24_probabilities[i])) == v24_index,
            "v3_top1_correct": int(np.argmax(v3_probabilities[i])) == v3_index,
            "v24_top1_score": float(v24_top[i]), "v3_top1_score": float(v3_top[i]),
        })
    v24_true_array, v3_true_array = np.asarray(v24_true), np.asarray(v3_true)
    sources = sorted({clip.source_kind for clip in clips})
    thresholds = {}
    for name, value in V24_THRESHOLDS.items():
        by_source = {
            source: _match_row(
                v24_top[[i for i, c in enumerate(clips) if c.source_kind == source]],
                v3_top[[i for i, c in enumerate(clips) if c.source_kind == source]], value, False,
            )
            for source in sources
        }
        pooled_v3_threshold = matched_threshold(v24_top, v3_top, value)
        thresholds[name] = {
            "v24": value,
            # Operating point of the chosen (pooled) V3 value on known birds / pure noise.
            "labelled_true_class_recall": {
                "v24_at_v24_threshold": float(np.mean(v24_true_array >= value)) if v24_true else None,
                "v3_at_pooled_v3_threshold": float(np.mean(v3_true_array >= pooled_v3_threshold)) if v3_true else None,
            },
            "negative_pass_fraction": {
                "v24_at_v24_threshold": float(np.mean(v24_top[negatives] >= value)) if negatives else None,
                "v3_at_pooled_v3_threshold": float(np.mean(v3_top[negatives] >= pooled_v3_threshold)) if negatives else None,
            },
            "pooled_top1": _match_row(v24_top, v3_top, value, True),
            "labelled_true_class": _match_row(v24_true_array, v3_true_array, value, True),
            "negative_top1": _match_row(v24_top[negatives], v3_top[negatives], value, False),
            "by_source_top1": by_source,
        }
    distances = sorted({row["distance_m"] for row in per_clip if row["distance_m"] is not None})
    accuracy_by_distance = {
        str(distance): {
            "clips": len(rows),
            "v24_top1_accuracy": float(np.mean([r["v24_top1_correct"] for r in rows])),
            "v3_top1_accuracy": float(np.mean([r["v3_top1_correct"] for r in rows])),
            "v24_median_true_score": float(np.median([r["v24_true_score"] for r in rows])),
            "v3_median_true_score": float(np.median([r["v3_true_score"] for r in rows])),
        }
        for distance in distances
        for rows in [[r for r in per_clip if r["distance_m"] == distance]]
    }
    return {
        "clip_counts": {source: sum(c.source_kind == source for c in clips) for source in sources},
        "labelled_clips": len(per_clip),
        "negative_clips": len(negatives),
        "thresholds": thresholds,
        "labelled_accuracy_by_sim_distance_m": accuracy_by_distance,
        "labelled_clip_rows": per_clip,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--local-audio", action="append", default=[], help="NAME=DIR[:MAX_CLIPS]")
    parser.add_argument("--local-species", action="append", default=[], help="NAME=Scientific name")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--write-map", action="store_true", help=f"write {DEFAULT_V24_TO_V3_CONFIDENCE_MAP_PATH.name}")
    parser.add_argument("--map-output", type=Path, default=DEFAULT_V24_TO_V3_CONFIDENCE_MAP_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not V3_MODEL_PATH.is_file():
        logger.error("V3 model missing; run scripts/fetch_birdnet_v3.py")
        return 1

    species_by_name = dict(spec.split("=", 1) for spec in args.local_species)
    clips = fixture_clips()
    with tempfile.TemporaryDirectory(prefix="birdnet-ladder-") as work:
        for seed in args.seeds:
            clips += sim_ladder_clips(seed, Path(work))
    for spec in args.local_audio:
        clips += local_clips(spec, species_by_name, BOOTSTRAP_SEED)
    logger.info("%d clips (%.1f audio minutes)", len(clips), sum(c.audio.size / c.sample_rate_hz for c in clips) / 60)

    model_clips = [(clip.name, clip.audio, clip.sample_rate_hz) for clip in clips]
    v3_run = run_v3_model(V3_MODEL_PATH, V3_LABELS_PATH, model_clips, BirdNETV3WindowPolicy(), args.threads)
    v24_run, v24_species = run_v24_model(model_clips, args.threads)
    report = analyse(clips, v24_run.clip_probabilities, v24_species, v3_run.clip_probabilities,
                     load_birdnet_v3_labels(V3_LABELS_PATH))
    v24_top = np.asarray([p.max() for p in v24_run.clip_probabilities])
    v3_top = np.asarray([p.max() for p in v3_run.clip_probabilities])
    confidence_map = build_confidence_quantile_map(
        v24_top, v3_top, source_model_version="2.4", target_model_version="3.0",
        provenance=(
            "scripts/calibrate_birdnet_v3_thresholds.py pooled clip top-1 quantile match; "
            f"sources {report['clip_counts']}"
        ),
    )
    report["confidence_map"] = confidence_map.to_dict()
    summary = {name: row["pooled_top1"]["v3_threshold"] for name, row in report["thresholds"].items()}
    print(json.dumps({k: v for k, v in report.items() if k != "labelled_clip_rows"}, indent=2, default=float))
    print("pooled-top1 V3 thresholds:", json.dumps(summary, indent=2))
    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    if args.write_map:
        write_confidence_quantile_map(confidence_map, args.map_output)
        logger.info("wrote %s", args.map_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
