"""Golden cases for the Rust S5 port (relative trigger + attribution decisions).

Python is the reference:

- ``NoiseFloorRelativeTrigger.observe`` on interleaved multi-node frame
  sequences (steady floors, calls, gusts, non-finite frames, LRU eviction);
- the real ``OnsetClusterAttributor.attribute`` cascade on random cluster
  evidence, with the host-owned signal work scripted: masking returns a given
  masked-cell fraction and the classifier rerun a given answer (the Rust port
  leaves both to its ``LocoRerunHost`` callback).

``python -m tests.attribution_trigger_parity_cases`` writes
``tests/fixtures/attribution_trigger_parity_goldens.npz`` (``case_inputs_json``,
``expected_outputs_json``). The test feeds inputs to ``attribution-oracle``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from minimappr.config import Settings
from minimappr.core.ingest_relative_trigger import NoiseFloorRelativeTrigger
from minimappr.core.onset_cluster_attribution import AttributionThresholds, ClusterAttributionContext, loco_decision
from minimappr.core.onset_cluster_attribution_evidence import ClusterAttributionEvidence, ClusterOnsetArrival
from minimappr.core.onset_cluster_attribution_stage import ATTRIBUTION_FEATURE_KEY, OnsetClusterAttributor
from minimappr.core.onset_cluster_masking import MaskedSignal
from minimappr.models import ClassificationResult

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "attribution_trigger_parity_goldens.npz"
NODES = ("tetra-a", "tetra-b", "tetra-c", "phone-d")
LABELS = ("american robin", "scarlet tanager", "red-eyed vireo", "wood thrush", "unknown", "background")
SAMPLE_RATE_HZ = 24_000
LRU_NODE_COUNT = 1030


def relative_trigger_case(seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    settings = Settings(trigger_noise_floor_relative_enabled=True)
    trigger = NoiseFloorRelativeTrigger(settings.relative_trigger_config())
    floors = {node: float(10 ** rng.uniform(-4.5, -3.0)) for node in NODES[:3]}
    frames = []
    for index in range(600):
        node = NODES[index % 3]
        rms = floors[node] * float(np.exp(rng.normal(scale=0.15)))
        if rng.uniform() < 0.04:
            rms *= float(10 ** rng.uniform(0.2, 1.5))  # a call
        if 300 <= index < 400 and node == "tetra-b":
            rms *= float(np.exp(rng.normal(scale=1.0)))  # gusts
        frames.append([node, None if rng.uniform() < 0.005 else rms, 0.02])
    outputs = []
    for node, rms, seconds in frames:
        triggered = trigger.observe(node, float("nan") if rms is None else rms, seconds)
        outputs.append([triggered, trigger.floor_rms(node)])
    return {"kind": "relative_trigger", "frames": frames}, {"steps": outputs}


def relative_trigger_lru_case() -> tuple[dict, dict]:
    trigger = NoiseFloorRelativeTrigger(Settings().relative_trigger_config())
    frames = [[f"node-{index}", 1e-3, 0.02] for index in range(LRU_NODE_COUNT)]
    frames.insert(5, ["node-0", 1e-3, 0.02])  # node-0 heard again: node-1 is now the oldest
    for node, rms, seconds in frames:
        trigger.observe(node, rms, seconds)
    probes = ["node-0", "node-1", "node-6", "node-7", f"node-{LRU_NODE_COUNT - 1}"]
    return (
        {"kind": "relative_trigger_lru", "frames": frames, "probes": probes},
        {"tracked": trigger.tracked_node_count(), "probe_floors": [trigger.floor_rms(node) for node in probes]},
    )


def _random_evidence(rng: np.random.Generator, window_start: int, window_end: int) -> dict:
    nodes = list(NODES[: int(rng.integers(1, 5))])
    band_min = float(rng.uniform(1000, 5000))
    band = (band_min, band_min + float(rng.uniform(300, 3000)))
    arrivals = {}
    for node in nodes:
        arrivals[node] = [
            [int(rng.integers(window_start - SAMPLE_RATE_HZ, window_end + SAMPLE_RATE_HZ // 2)), band[0], band[1]]
            for _ in range(int(rng.integers(1, 4)))
        ]
    fitted = float(rng.uniform(20, 60)) if rng.uniform() < 0.85 else None
    return {
        "node_band_snr_db": {node: float(rng.uniform(-2, 30)) for node in nodes},
        "node_distances_m": {node: float(rng.uniform(0.5, 60)) for node in nodes},
        "node_onset_arrivals": arrivals,
        "fitted_source_level_db": fitted,
        "spreading_residual_db": float(rng.uniform(0, 9)) if fitted is not None and len(nodes) > 1 else None,
    }


def _scores(rng: np.random.Generator) -> dict[str, float]:
    picks = rng.choice(len(LABELS), size=int(rng.integers(0, 4)), replace=False)
    return {LABELS[index]: float(np.round(rng.uniform(0.05, 0.95), 3)) for index in picks}


def _top(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "unknown", 0.0
    label, value = max(scores.items(), key=lambda item: item[1])
    return label, value


def attribution_case(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    window_start = int(rng.integers(0, 4 * SAMPLE_RATE_HZ))
    window_end = window_start + 3 * SAMPLE_RATE_HZ
    clusters = {str(index): _random_evidence(rng, window_start, window_end) for index in range(int(rng.integers(1, 4)))}
    target = int(rng.integers(0, len(clusters)))
    classifying = next(iter(clusters[str(target)]["node_band_snr_db"]))
    original_scores, masked_scores = _scores(rng), _scores(rng)
    original_label, original_confidence = _top(original_scores)
    masked_label, masked_confidence = _top(masked_scores)
    if rng.uniform() < 0.4:
        # Target near the classifying node, interferers far, levels on the spreading
        # fit: exercises the settled gradient branch and its band-overlap guard.
        for index, evidence in clusters.items():
            if classifying not in evidence["node_distances_m"]:
                continue
            near = int(index) == target
            evidence["node_distances_m"][classifying] = float(rng.uniform(1, 3) if near else rng.uniform(25, 60))
            evidence["fitted_source_level_db"] = float(rng.uniform(40, 50))
            evidence["spreading_residual_db"] = float(rng.uniform(0, 4))
            distance = evidence["node_distances_m"][classifying]
            evidence["node_band_snr_db"][classifying] = evidence["fitted_source_level_db"] - 20 * math.log10(distance)
    return {
        "kind": "attribution", "window_id": f"w{seed // 6}", "target_index": target, "clusters": clusters,
        "classifying_node_id": classifying, "window_start_sample": window_start, "window_end_sample": window_end,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "shared_onset_max_lag_samples": int(rng.integers(100, 1200)) if rng.uniform() < 0.6 else None,
        # Scores travel as [label, score] pairs: dict order breaks ties on both sides.
        "original": {"label": original_label, "confidence": original_confidence, "scores": list(map(list, original_scores.items()))},
        "masked": {"label": masked_label, "confidence": masked_confidence, "scores": list(map(list, masked_scores.items()))},
        "masked_cell_fraction": 0.0 if rng.uniform() < 0.15 else float(rng.uniform(0.01, 0.4)),
        "loco_enabled": bool(rng.uniform() < 0.9),
    }


def attribution_band_overlap_case(seed: int) -> dict:
    """Two clusters in one band, both near the classifying node, levels on the
    spreading fit: dominance is measured and predicted, but only the band
    overlap guard (geometry too weak to justify it) can refuse it. Five near
    interferers (refused, then LOCO in one shared window: the rerun cap is hit)
    and one far one (settled)."""
    rng = np.random.default_rng(seed)
    window_start = int(rng.integers(0, 4 * SAMPLE_RATE_HZ))
    window_end = window_start + 3 * SAMPLE_RATE_HZ
    band_min = float(rng.uniform(2000, 4000))
    band = (band_min, band_min + float(rng.uniform(500, 2000)))
    target_distance = float(rng.uniform(1.5, 2.5))
    interferer_distance = target_distance * (6.0 if seed % 6 == 5 else 1.4)
    target_level = float(rng.uniform(55, 65))
    clusters = {}
    for index, (distance, level) in enumerate(((target_distance, target_level), (interferer_distance, target_level - 14.0))):
        distances = {"tetra-a": distance, "tetra-b": distance + 10.0, "tetra-c": distance + 20.0}
        clusters[str(index)] = {
            "node_band_snr_db": {node: level - 20 * math.log10(d) for node, d in distances.items()},
            "node_distances_m": distances,
            "node_onset_arrivals": {
                node: [[int(rng.integers(window_start, window_end)), band[0], band[1]]] for node in distances
            },
            "fitted_source_level_db": level,
            "spreading_residual_db": float(rng.uniform(0, 2)),
        }
    scores = {"american robin": 0.8, "scarlet tanager": 0.3}
    return {
        "kind": "attribution", "window_id": "overlap", "target_index": 0, "clusters": clusters,
        "classifying_node_id": "tetra-a", "window_start_sample": window_start, "window_end_sample": window_end,
        "sample_rate_hz": SAMPLE_RATE_HZ, "shared_onset_max_lag_samples": None,
        "original": {"label": "american robin", "confidence": 0.8, "scores": list(map(list, scores.items()))},
        "masked": {"label": "unknown", "confidence": 0.0, "scores": []},
        "masked_cell_fraction": 0.2, "loco_enabled": True,
    }


class _ScriptedAttributor(OnsetClusterAttributor):
    """The real cascade with the host-owned masking scripted."""

    def __init__(self, config, orchestrator, metrics, masked_fraction_by_call: list[float]) -> None:
        super().__init__(config, orchestrator, lambda: metrics)
        self._masked_fractions = masked_fraction_by_call

    async def _mask_interferers(self, classified, context, interferers) -> MaskedSignal:
        seeds = [onset for interferer in interferers for onset in interferer.onset_seconds_in_window]
        self.last_seed_count = len(seeds)
        self.last_seeds = [[onset, band_min, band_max] for onset, band_min, band_max in seeds]
        return MaskedSignal(samples=np.zeros(4), masked_cell_fraction=self._masked_fractions.pop(0), region_seconds=())


class _ScriptedOrchestrator:
    def __init__(self) -> None:
        self.masked_answers: list[ClassificationResult] = []

    async def rerun_classifier_on_prepared_signal(self, samples, sample_rate_hz):
        return self.masked_answers.pop(0)

    async def relabel_classified_result(self, classified, attributed, *, event_time_ns):
        return SimpleNamespace(classification=attributed, classification_signal=classified.classification_signal)


class _Metrics(SimpleNamespace):
    def __getattr__(self, name):
        return 0


def _evidence(raw: dict) -> ClusterAttributionEvidence:
    return ClusterAttributionEvidence(
        node_band_snr_db=dict(raw["node_band_snr_db"]),
        node_distances_m=dict(raw["node_distances_m"]),
        node_onset_arrivals={
            node: tuple(ClusterOnsetArrival(span_sample=a[0], band_min_hz=a[1], band_max_hz=a[2]) for a in arrivals)
            for node, arrivals in raw["node_onset_arrivals"].items()
        },
        fitted_source_level_db=raw["fitted_source_level_db"],
        spreading_residual_db=raw["spreading_residual_db"],
    )


def _classification(raw: dict) -> ClassificationResult:
    return ClassificationResult(label=raw["label"], confidence=raw["confidence"], scores=dict(map(tuple, raw["scores"])))


async def _run_attribution_cases(cases: list[dict]) -> list[dict]:
    """One attributor across all cases, so the per-window LOCO budget carries over."""
    orchestrator = _ScriptedOrchestrator()
    fractions: list[float] = []
    config = Settings(onset_attribution_enabled=True, localization_onset_clustering_enabled=True).onset_cluster_attribution_config()
    attributor = _ScriptedAttributor(config, orchestrator, _Metrics(), fractions)
    outputs = []
    for case in cases:
        attributor.config = dataclasses.replace(config, loco_enabled=case["loco_enabled"])
        fractions.append(case["masked_cell_fraction"])
        orchestrator.masked_answers.append(_classification(case["masked"]))
        original = _classification(case["original"])
        classified = SimpleNamespace(classification=original, classification_signal=np.zeros(4))
        context = ClusterAttributionContext(
            window_id=case["window_id"], target_index=case["target_index"],
            cluster_evidence={int(index): _evidence(raw) for index, raw in case["clusters"].items()},
            classifying_node_id=case["classifying_node_id"], classification_start_sample=case["window_start_sample"],
            classification_end_sample=case["window_end_sample"], sample_rate_hz=case["sample_rate_hz"],
            shared_onset_max_lag_samples=case["shared_onset_max_lag_samples"],
        )
        attributor.last_seeds = None
        result = await attributor.attribute(classified, context, event_time_ns=0)
        annotation = result.classification.features.get(ATTRIBUTION_FEATURE_KEY, {})
        outputs.append({
            "tier": annotation.get("tier"), "interferers": annotation.get("interferers", 0),
            "gradient_reason": annotation.get("gradient_reason"), "loco_reason": annotation.get("loco_reason"),
            "label": result.classification.label, "confidence": float(result.classification.confidence),
            "dominance_db": {key: value for key, value in annotation.get("dominance_db", {}).items()},
            "collapsed": annotation.get("loco_collapsed_labels"),
            # Why the cascade stopped short of a decision (loco_disabled /
            # interferers_silent_in_window / loco_rerun_cap), and what was masked.
            "reason": annotation.get("reason"),
            "mask_seeds": attributor.last_seeds,
        })
        fractions.clear()
        orchestrator.masked_answers.clear()
    return outputs


def loco_decision_case(seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(80):
        original_scores, masked_scores = _scores(rng), _scores(rng)
        if rng.uniform() < 0.3 and original_scores:
            # The masked rerun keeps the original winner (often the settled branch).
            label, value = _top(original_scores)
            masked_scores[label] = float(np.round(min(0.99, value + rng.uniform(-0.2, 0.3)), 3))
        pairs.append({
            "original": dict(zip(("label", "confidence"), _top(original_scores)), scores=list(map(list, original_scores.items()))),
            "masked": dict(zip(("label", "confidence"), _top(masked_scores)), scores=list(map(list, masked_scores.items()))),
        })
    thresholds = AttributionThresholds(3.0, 10.0, 6.0, 0.3, 0.1, 0.5)
    decisions = []
    for pair in pairs:
        decision = loco_decision(
            original_label=pair["original"]["label"], original_scores=dict(map(tuple, pair["original"]["scores"])),
            masked_label=pair["masked"]["label"], masked_confidence=pair["masked"]["confidence"],
            masked_scores=dict(map(tuple, pair["masked"]["scores"])), thresholds=thresholds,
        )
        decisions.append([decision.settled, decision.reason, decision.label, decision.confidence, list(decision.collapsed_labels)])
    return {"kind": "loco_decision", "pairs": pairs}, {"decisions": decisions}


def all_cases() -> list[tuple[dict, dict]]:
    cases = [relative_trigger_case(seed) for seed in range(4)]
    cases.append(relative_trigger_lru_case())
    cases.extend(loco_decision_case(700 + seed) for seed in range(2))
    attribution_inputs = [attribution_case(seed) for seed in range(160)]
    attribution_inputs += [attribution_band_overlap_case(900 + seed) for seed in range(6)]
    batch = {"kind": "attribution_batch", "cases": attribution_inputs}
    cases.append((batch, {"results": asyncio.run(_run_attribution_cases(attribution_inputs))}))
    return cases


def main() -> None:
    cases = all_cases()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        GOLDEN_PATH,
        case_inputs_json=np.asarray([json.dumps(case) for case, _ in cases]),
        expected_outputs_json=np.asarray([json.dumps(expected) for _, expected in cases]),
    )
    print(f"wrote {len(cases)} cases to {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
