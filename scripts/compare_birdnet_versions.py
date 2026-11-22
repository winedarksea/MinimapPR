#!/usr/bin/env python3
"""Compare BirdNET+ V3.0 FP16 vs INT8 vs BirdNET v2.4 on a folder of clips.

Per clip, each model yields one clip-level probability vector (max over its
windows, exactly as the classifiers aggregate). Reported:

* V3 reference (FP16) vs each candidate (INT8 variants): top-1 agreement on
  confident clips (reference top score >= ``--confident-score``) and on all
  clips; per-clip Spearman rank correlation over the union of both models'
  top-``--rank-top-k`` classes; Pearson of the reference top-1 class score
  across clips; latency (median ms per clip, ms per audio second).
* v2.4 vs V3 FP16: top-1 agreement after mapping v2.4 labels into V3 with
  ``minimappr/assets/birdnet/v24_to_v3_label_map.csv`` (clips where both are
  confident), and latency.

The INT8 accept gate (stated before measuring, see
docs/field_mode_progress/birdnet_v3.md) is evaluated and printed.

Usage (local)::

    python scripts/compare_birdnet_versions.py --audio tests/house_finch.wav ~/clips \\
        --output-json /tmp/birdnet_compare.json

Usage (Bluefin, from the repo checkout, after fetch + quantize)::

    python scripts/compare_birdnet_versions.py --audio data/snippets \\
        --max-clips 500 --output-json data/birdnet_v3_compare.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from minimappr.classifiers.birdnet_v3 import BirdNETV3Classifier  # noqa: E402
from minimappr.classifiers.birdnet_v3_audio import BirdNETV3WindowPolicy, prepare_birdnet_v3_windows  # noqa: E402
from minimappr.classifiers.birdnet_v3_labels import load_v24_to_v3_label_map  # noqa: E402
from quantize_birdnet_v3 import collect_audio_files, load_mono_audio  # noqa: E402

logger = logging.getLogger("compare_birdnet_versions")

MODEL_DIRECTORY = REPOSITORY_ROOT / "data/models/birdnet_v3"
DEFAULT_LABELS = MODEL_DIRECTORY / "BirdNET+_V3.0_Global_12K_Labels.csv"
DEFAULT_V3_MODELS = {
    "fp16_pruned": MODEL_DIRECTORY / "BirdNET+_V3.0_Global_12K_FP16_pruned.onnx",
    "int8_dynamic": MODEL_DIRECTORY / "BirdNET+_V3.0_Global_12K_pruned_INT8_dynamic.onnx",
    "int8_static": MODEL_DIRECTORY / "BirdNET+_V3.0_Global_12K_pruned_INT8_static.onnx",
}
REFERENCE_MODEL = "fp16_pruned"

# INT8 accept gate, fixed before any measurement.
GATE_MIN_CONFIDENT_TOP1_AGREEMENT = 0.95
GATE_MIN_MEDIAN_SPEARMAN = 0.90
GATE_MIN_LATENCY_SPEEDUP = 1.2
GATE_MIN_CONFIDENT_CLIPS = 30

V24_SAMPLE_RATE_HZ = 48_000
V24_SEGMENT_SAMPLES = 144_000
V24_OVERLAP_SECONDS = 1.5
WARMUP_RUNS = 2


@dataclass
class ModelRun:
    clip_probabilities: list[np.ndarray] = field(default_factory=list)
    clip_milliseconds: list[float] = field(default_factory=list)


def _clip_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes(), usedforsecurity=False).hexdigest()


def load_clips(paths: list[Path], max_clips: int) -> list[tuple[str, np.ndarray, int]]:
    seen: set[str] = set()
    clips: list[tuple[str, np.ndarray, int]] = []
    for path in collect_audio_files(paths):
        digest = _clip_hash(path)
        if digest in seen:
            continue
        seen.add(digest)
        try:
            audio, sample_rate_hz = load_mono_audio(path)
        except Exception as error:  # noqa: BLE001 - skip undecodable clips
            logger.warning("skip %s: %s", path, error)
            continue
        clips.append((path.name, audio, sample_rate_hz))
        if len(clips) >= max_clips:
            break
    return clips


def run_v3_model(model_path: Path, labels_path: Path, clips, policy: BirdNETV3WindowPolicy, threads: int) -> ModelRun:
    classifier = BirdNETV3Classifier(
        model_path=model_path, labels_path=labels_path, window_policy=policy, num_threads=threads
    )
    run = ModelRun()
    for index, (_, audio, sample_rate_hz) in enumerate(clips):
        repeats = 1 + (WARMUP_RUNS if index == 0 else 0)
        for _ in range(repeats):
            started = time.perf_counter()
            windows, verdict = prepare_birdnet_v3_windows(audio, sample_rate_hz, policy)
            if verdict.accepted and windows:
                probabilities = classifier.predict_window_probabilities([w for _, _, w in windows]).max(axis=0)
            else:
                probabilities = np.zeros(len(classifier.labels))
            elapsed_ms = (time.perf_counter() - started) * 1e3
        run.clip_probabilities.append(probabilities)
        run.clip_milliseconds.append(elapsed_ms)
    classifier.close()
    return run


def run_v24_model(clips, threads: int) -> tuple[ModelRun, list[str]]:
    from birdnet import model_loader  # noqa: PLC0415 - optional dependency
    from scipy.signal import resample_poly  # noqa: PLC0415

    from minimappr.classifiers.birdnet import _resolve_interpreter_cls  # noqa: PLC0415

    model = model_loader.load("acoustic", "2.4", "tf", library="tflite")
    interpreter = _resolve_interpreter_cls()(model_path=str(model._model_path), num_threads=threads)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    hop = V24_SEGMENT_SAMPLES - int(V24_OVERLAP_SECONDS * V24_SAMPLE_RATE_HZ)
    run = ModelRun()
    for index, (_, audio, sample_rate_hz) in enumerate(clips):
        for _ in range(1 + (WARMUP_RUNS if index == 0 else 0)):
            started = time.perf_counter()
            resampled = audio if sample_rate_hz == V24_SAMPLE_RATE_HZ else resample_poly(audio, V24_SAMPLE_RATE_HZ, sample_rate_hz)
            best = None
            for start in range(0, max(1, resampled.size - V24_SEGMENT_SAMPLES + hop), hop):
                buffer = np.zeros((1, V24_SEGMENT_SAMPLES), np.float32)
                segment = resampled[start : start + V24_SEGMENT_SAMPLES]
                buffer[0, : segment.size] = segment
                interpreter.set_tensor(input_index, buffer)
                interpreter.invoke()
                scores = 1.0 / (1.0 + np.exp(-interpreter.get_tensor(output_index)[0].astype(np.float64)))
                best = scores if best is None else np.maximum(best, scores)
            elapsed_ms = (time.perf_counter() - started) * 1e3
        run.clip_probabilities.append(best)
        run.clip_milliseconds.append(elapsed_ms)
    return run, list(model.species_list)


def rank_correlation_over_top_classes(reference: np.ndarray, candidate: np.ndarray, top_k: int) -> float:
    classes = np.union1d(np.argsort(reference)[-top_k:], np.argsort(candidate)[-top_k:])
    rho = spearmanr(reference[classes], candidate[classes]).statistic
    return float(rho) if np.isfinite(rho) else 0.0


def compare_v3_pair(reference: ModelRun, candidate: ModelRun, confident_score: float, top_k: int) -> dict:
    confident = [i for i, p in enumerate(reference.clip_probabilities) if p.max() >= confident_score]
    agree = [
        int(np.argmax(reference.clip_probabilities[i]) == np.argmax(candidate.clip_probabilities[i]))
        for i in range(len(reference.clip_probabilities))
    ]
    spearman = [
        rank_correlation_over_top_classes(r, c, top_k)
        for r, c in zip(reference.clip_probabilities, candidate.clip_probabilities)
    ]
    reference_top = [float(p.max()) for p in reference.clip_probabilities]
    candidate_same_class = [
        float(c[int(np.argmax(r))]) for r, c in zip(reference.clip_probabilities, candidate.clip_probabilities)
    ]
    pearson = pearsonr(reference_top, candidate_same_class).statistic if len(reference_top) > 2 else float("nan")
    max_abs_diff = [float(np.max(np.abs(r - c))) for r, c in zip(reference.clip_probabilities, candidate.clip_probabilities)]
    return {
        "clips": len(agree),
        "confident_clips": len(confident),
        "top1_agreement_all": float(np.mean(agree)) if agree else float("nan"),
        "top1_agreement_confident": float(np.mean([agree[i] for i in confident])) if confident else float("nan"),
        "median_spearman_top_classes": float(statistics.median(spearman)) if spearman else float("nan"),
        "min_spearman_top_classes": float(min(spearman)) if spearman else float("nan"),
        "pearson_reference_top1_score": float(pearson),
        "median_max_abs_probability_diff": float(statistics.median(max_abs_diff)) if max_abs_diff else float("nan"),
        "latency_speedup": latency_summary(reference)["median_ms_per_clip"] / latency_summary(candidate)["median_ms_per_clip"],
    }


def latency_summary(run: ModelRun, clips=None) -> dict:
    summary = {"median_ms_per_clip": float(statistics.median(run.clip_milliseconds))}
    if clips is not None:
        per_second = [ms / (audio.size / rate) for ms, (_, audio, rate) in zip(run.clip_milliseconds, clips)]
        summary["median_ms_per_audio_second"] = float(statistics.median(per_second))
    return summary


def compare_v24_to_v3(v24: ModelRun, v24_species: list[str], v3: ModelRun, confident_score: float) -> dict:
    mapping = load_v24_to_v3_label_map()
    both_confident, agree = 0, 0
    for v24_probabilities, v3_probabilities in zip(v24.clip_probabilities, v3.clip_probabilities):
        if v24_probabilities.max() < confident_score or v3_probabilities.max() < confident_score:
            continue
        both_confident += 1
        mapped = mapping.get(v24_species[int(np.argmax(v24_probabilities))])
        agree += int(mapped is not None and mapped.v3_index == int(np.argmax(v3_probabilities)))
    return {
        "both_confident_clips": both_confident,
        "top1_agreement_mapped": agree / both_confident if both_confident else float("nan"),
        "v24_confident_clips": sum(p.max() >= confident_score for p in v24.clip_probabilities),
        "v3_confident_clips": sum(p.max() >= confident_score for p in v3.clip_probabilities),
    }


def int8_gate_verdict(pair: dict) -> tuple[str, list[str]]:
    reasons = []
    if pair["confident_clips"] < GATE_MIN_CONFIDENT_CLIPS:
        reasons.append(f"only {pair['confident_clips']} confident clips (< {GATE_MIN_CONFIDENT_CLIPS}): provisional")
    if not pair["top1_agreement_confident"] >= GATE_MIN_CONFIDENT_TOP1_AGREEMENT:
        reasons.append(f"confident top-1 agreement {pair['top1_agreement_confident']:.3f} < {GATE_MIN_CONFIDENT_TOP1_AGREEMENT}")
    if not pair["median_spearman_top_classes"] >= GATE_MIN_MEDIAN_SPEARMAN:
        reasons.append(f"median Spearman {pair['median_spearman_top_classes']:.3f} < {GATE_MIN_MEDIAN_SPEARMAN}")
    if not pair["latency_speedup"] >= GATE_MIN_LATENCY_SPEEDUP:
        reasons.append(f"speedup {pair['latency_speedup']:.2f}x < {GATE_MIN_LATENCY_SPEEDUP}x")
    failed = [r for r in reasons if "provisional" not in r]
    if failed:
        return "REJECT", reasons
    return ("ACCEPT (provisional)" if reasons else "ACCEPT"), reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model", action="append", default=[], help="name=path; default fp16 + both INT8 variants")
    parser.add_argument("--skip-v24", action="store_true")
    parser.add_argument("--max-clips", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--confident-score", type=float, default=0.5)
    parser.add_argument("--rank-top-k", type=int, default=50)
    parser.add_argument("--max-window-seconds", type=float, default=6.0)
    parser.add_argument("--window-overlap-seconds", type=float, default=3.0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    models = dict(DEFAULT_V3_MODELS)
    for spec in args.model:
        name, _, path = spec.partition("=")
        models[name] = Path(path)
    models = {name: path for name, path in models.items() if path.is_file()}
    if REFERENCE_MODEL not in models:
        logger.error("reference model %s missing; run scripts/fetch_birdnet_v3.py", REFERENCE_MODEL)
        return 1
    clips = load_clips(args.audio, args.max_clips)
    logger.info("%d unique clips, %.1f audio minutes", len(clips), sum(a.size / r for _, a, r in clips) / 60)
    policy = BirdNETV3WindowPolicy(max_window_seconds=args.max_window_seconds, window_overlap_seconds=args.window_overlap_seconds)

    runs = {name: run_v3_model(path, args.labels, clips, policy, args.threads) for name, path in models.items()}
    report: dict = {
        "clips": len(clips),
        "audio_seconds": sum(a.size / r for _, a, r in clips),
        "confident_score": args.confident_score,
        "latency": {name: latency_summary(run, clips) for name, run in runs.items()},
        "v3_pairs": {},
        "gate": {
            "min_confident_top1_agreement": GATE_MIN_CONFIDENT_TOP1_AGREEMENT,
            "min_median_spearman": GATE_MIN_MEDIAN_SPEARMAN,
            "min_latency_speedup": GATE_MIN_LATENCY_SPEEDUP,
            "min_confident_clips_for_final_verdict": GATE_MIN_CONFIDENT_CLIPS,
        },
    }
    for name, run in runs.items():
        if name == REFERENCE_MODEL:
            continue
        pair = compare_v3_pair(runs[REFERENCE_MODEL], run, args.confident_score, args.rank_top_k)
        pair["verdict"], pair["verdict_reasons"] = int8_gate_verdict(pair)
        report["v3_pairs"][f"{REFERENCE_MODEL}_vs_{name}"] = pair
    if not args.skip_v24:
        try:
            v24_run, v24_species = run_v24_model(clips, args.threads)
            report["latency"]["v2.4_tflite"] = latency_summary(v24_run, clips)
            report["v24_vs_v3"] = compare_v24_to_v3(v24_run, v24_species, runs[REFERENCE_MODEL], args.confident_score)
        except Exception as error:  # noqa: BLE001 - v2.4 is optional (birdnet package)
            logger.warning("v2.4 comparison skipped: %s", error)

    print(json.dumps(report, indent=2, default=float))
    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
