#!/usr/bin/env python3
"""Replay gate: BirdNET v2.4 vs V3.0 through the full pipeline (W3-birdnet-cutover).

Renders ``build_species_gate_scenario`` bundles (three tetra nodes, the eight
fixture birds singing from fixed perches, ``minimappr-sim``), replays each
through a FusionNode whose classifier is the BirdNET member alone (built by
the production factory, so each version's threshold set and the V3 routing
migration apply), and scores:

* exact label accuracy (``match = "exact"``) and event recall (v1 metrics);
* entity metrics (``replay_entity_metrics``) at 15 m and 5 m.

Gate (fixed before measuring; all must hold, summed over seeds):

Gated V3 variant: "3.0+regional" (see ``MODEL_VARIANTS``).

1. V3 exact label accuracy >= v2.4;
2. V3 event recall >= v2.4 - ``MAX_EVENT_RECALL_DROP``;
3. V3 label swaps <= v2.4 and V3 duplicate entities <= v2.4 (15 m);
4. real-site evidence: the Bluefin snippet comparison has been run
   (``--bluefin-report``) and its mapped top-1 agreement on confident clips is
   >= ``MIN_BLUEFIN_CONFIDENT_AGREEMENT``.

Without (4) the verdict is at best ``INSUFFICIENT EVIDENCE`` and the default
stays "2.4": simulated fixture audio alone cannot justify switching.

    python scripts/birdnet_version_replay_gate.py --seeds 0 1 [--json out.json]
        [--bluefin-report data/birdnet_v3_compare.json]
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from measure_replay_entity_baseline import ENTITY_METRIC_KEYS, _collect_settled_detections  # noqa: E402

from minimappr.calibration.bundle import load_bundle  # noqa: E402
from minimappr.classifiers.factory import create_context_classifier  # noqa: E402
from minimappr.classifiers.routing import (  # noqa: E402
    CONTEXT_DETECTION_TRIGGER,
    ContextSpec,
    RoutingConfig,
    apply_settings,
    default_routing,
)
from minimappr.config import Settings  # noqa: E402
from minimappr.sim.fixture_audio_scenarios import (  # noqa: E402
    build_species_gate_scenario,
    load_fixture_bird_clips,
    render_scenario_bundle,
)
from minimappr.sim.replay import build_fusion_for_bundle, evaluate_bundle, replay_bundle  # noqa: E402

# Variant -> (birdnet_model_version, use a regional species list for V3).
# Production v2.4 always runs its geo filter (site lat/lon), so the gated V3
# variant is "3.0+regional": V3 restricted to the same site species list
# (v2.4 geo model output mapped through the label map). Plain "3.0" (all
# 11,725 classes) is reported for information.
MODEL_VARIANTS: dict[str, tuple[str, bool]] = {
    "2.4": ("2.4", False),
    "3.0": ("3.0", False),
    "3.0+regional": ("3.0", True),
}
GATED_V3_VARIANT = "3.0+regional"
# Harness overrides: the trigger level only decides which windows reach the
# classifier in this replay (same role as tests/test_sim_bundle_roundtrip.py);
# 3 s windows match v2.4's segment length and stay inside V3's 6 s maximum.
REPLAY_SETTINGS_OVERRIDES = {"trigger_rms": 0.003, "classification_window_seconds": 3.0}
ASSOCIATION_RADII_M = (15.0, 5.0)
EXPECTATIONS = {
    "schema_version": 1,
    "classification": {"min_label_accuracy": 0.0, "match": "exact"},
    "localization": {"min_event_recall": 0.0, "far_field_distance_m": 120.0},
}
MAX_EVENT_RECALL_DROP = 0.05
MIN_BLUEFIN_CONFIDENT_AGREEMENT = 0.8


def birdnet_only_routing(settings: Settings) -> RoutingConfig:
    """Default routing with the trigger context running only the BirdNET member."""
    routing = default_routing()
    routing.contexts[CONTEXT_DETECTION_TRIGGER] = ContextSpec(name=CONTEXT_DETECTION_TRIGGER, run=("birdnet",))
    return apply_settings(routing, settings)


def write_v3_regional_species_list(latitude: float, longitude: float, geo_min_confidence: float, output_path: Path) -> int:
    """v2.4 geo-model site list, mapped to V3 scientific names (one per line)."""
    from birdnet import model_loader  # noqa: PLC0415 - optional dependency, needed for v2.4 anyway

    from minimappr.classifiers.birdnet import _predict_site_species_list  # noqa: PLC0415
    from minimappr.classifiers.birdnet_v3_labels import load_v24_to_v3_label_map  # noqa: PLC0415

    v24_site_species = _predict_site_species_list(
        model_loader=model_loader, latitude=latitude, longitude=longitude, geo_min_confidence=geo_min_confidence
    )
    if not v24_site_species:
        raise RuntimeError("v2.4 geo model returned no site species list")
    label_map = load_v24_to_v3_label_map()
    names = sorted({label_map[raw].v3_scientific_name for raw in v24_site_species if raw in label_map and label_map[raw].is_mapped})
    output_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return len(names)


def _site_overrides(bundle) -> dict:
    origin = bundle.manifest["site"]["origin"]
    return {"site_origin_lat": origin["lat"], "site_origin_lon": origin["lon"]}


async def replay_one(bundle, variant: str, work_directory: Path) -> dict:
    version, regional = MODEL_VARIANTS[variant]
    overrides = {**REPLAY_SETTINGS_OVERRIDES, **_site_overrides(bundle), "birdnet_model_version": version}
    if regional:
        work_directory.mkdir(parents=True, exist_ok=True)
        species_list_path = work_directory / "v3_regional_species.txt"
        write_v3_regional_species_list(
            overrides["site_origin_lat"], overrides["site_origin_lon"], Settings().birdnet_geo_min_confidence, species_list_path
        )
        overrides["birdnet_v3_species_list_path"] = str(species_list_path)
    settings = Settings(**overrides)
    classifier = create_context_classifier(settings, CONTEXT_DETECTION_TRIGGER, routing=birdnet_only_routing(settings))
    fusion, storage, _ = await build_fusion_for_bundle(
        bundle, work_directory / f"replay-{variant.replace('+', '-')}", overrides=overrides, classifier=classifier
    )
    try:
        await replay_bundle(fusion, bundle)
        detections = await _collect_settled_detections(storage)
    finally:
        await fusion.stop()
        await storage.close()
        classifier.close()
    report = evaluate_bundle(detections, bundle, EXPECTATIONS)
    entity_by_radius = {}
    for radius in ASSOCIATION_RADII_M:
        expectations = copy.deepcopy(EXPECTATIONS)
        expectations["entities"] = {"association_radius_m": radius}
        entity = asdict(evaluate_bundle(detections, bundle, expectations).entity_metrics)
        entity_by_radius[str(radius)] = {key: entity[key] for key in ENTITY_METRIC_KEYS}
    return {
        "variant": variant,
        "version": version,
        "detections": len(detections),
        "labels": sorted(str(d.get("label")) for d in detections),
        "call_events": len(report.event_results),
        "event_recall": report.event_recall,
        "label_accuracy_exact": report.label_accuracy,
        "labelled_events": sum(1 for r in report.event_results if r.matched and r.label_correct is not None),
        "correct_events": sum(1 for r in report.event_results if r.label_correct),
        "entity_metrics_by_radius_m": entity_by_radius,
    }


def _summed(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows)


def gate_verdict(results: dict[str, list[dict]], bluefin_report: dict | None) -> tuple[str, list[str]]:
    results = {"2.4": results["2.4"], "3.0": results[GATED_V3_VARIANT]}
    reasons: list[str] = []
    accuracy = {v: _summed(rows, "correct_events") / max(1.0, _summed(rows, "labelled_events")) for v, rows in results.items()}
    recall = {v: _summed(rows, "event_recall") / len(rows) for v, rows in results.items()}
    swaps = {v: sum(r["entity_metrics_by_radius_m"]["15.0"]["label_swap_count"] for r in rows) for v, rows in results.items()}
    duplicates = {v: sum(r["entity_metrics_by_radius_m"]["15.0"]["duplicate_entities"] for r in rows) for v, rows in results.items()}
    if accuracy["3.0"] < accuracy["2.4"]:
        reasons.append(f"exact label accuracy V3 {accuracy['3.0']:.3f} < v2.4 {accuracy['2.4']:.3f}")
    if recall["3.0"] < recall["2.4"] - MAX_EVENT_RECALL_DROP:
        reasons.append(f"event recall V3 {recall['3.0']:.3f} < v2.4 {recall['2.4']:.3f} - {MAX_EVENT_RECALL_DROP}")
    if swaps["3.0"] > swaps["2.4"]:
        reasons.append(f"label swaps V3 {swaps['3.0']} > v2.4 {swaps['2.4']}")
    if duplicates["3.0"] > duplicates["2.4"]:
        reasons.append(f"duplicate entities V3 {duplicates['3.0']} > v2.4 {duplicates['2.4']}")
    if reasons:
        return "FAIL (keep 2.4)", reasons
    agreement = (bluefin_report or {}).get("v24_vs_v3", {}).get("top1_agreement_mapped")
    if agreement is None:
        return "INSUFFICIENT EVIDENCE (sim passes; no Bluefin snippet comparison) - keep 2.4", [
            "sim gate passed; real-site evidence missing"
        ]
    if float(agreement) < MIN_BLUEFIN_CONFIDENT_AGREEMENT:
        return "FAIL (keep 2.4)", [f"Bluefin mapped top-1 agreement {agreement:.3f} < {MIN_BLUEFIN_CONFIDENT_AGREEMENT}"]
    return "PASS (switch default to 3.0 may be proposed)", []


async def main_async(seeds: list[int]) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {variant: [] for variant in MODEL_VARIANTS}
    clips = load_fixture_bird_clips()
    with tempfile.TemporaryDirectory(prefix="birdnet-gate-") as work:
        for seed in seeds:
            bundle = load_bundle(render_scenario_bundle(build_species_gate_scenario(clips, seed=seed), Path(work) / f"gate-{seed}.zip"))
            for variant in MODEL_VARIANTS:
                row = await replay_one(bundle, variant, Path(work) / f"seed-{seed}")
                row["seed"] = seed
                print(json.dumps({k: v for k, v in row.items() if k != "entity_metrics_by_radius_m"}), flush=True)
                results[variant].append(row)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--bluefin-report", type=Path, default=None, help="compare_birdnet_versions.py --output-json")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    results = asyncio.run(main_async(args.seeds))
    bluefin = json.loads(args.bluefin_report.read_text()) if args.bluefin_report else None
    verdict, reasons = gate_verdict(results, bluefin)
    print("VERDICT:", verdict)
    for reason in reasons:
        print("  -", reason)
    if args.json:
        args.json.write_text(json.dumps({"results": results, "verdict": verdict, "reasons": reasons}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
