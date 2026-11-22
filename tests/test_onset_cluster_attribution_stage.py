"""Field Mode W3-attribution classifier-worker seam: LOCO rerun on in-memory audio,
per-window rerun cap, unsettled -> unlabelled, no disk writes; plus the measured
per-node level constraint that replaces the partial-binding stand-in."""

from __future__ import annotations

import builtins
import dataclasses
import io
import os

import numpy as np
import pytest

from minimappr.classifiers.base import ClassificationResult
from minimappr.config import Settings
from minimappr.core.classification import ClassifiedResult
from minimappr.core.fusion_node import FusionMetrics
from minimappr.core.onset_cluster_attribution import ClusterAttributionContext
from minimappr.core.onset_cluster_attribution_evidence import ClusterAttributionEvidence, ClusterOnsetArrival
from minimappr.core.onset_cluster_attribution_stage import OnsetClusterAttributor
from minimappr.core.track_level_gradient_constraint import measured_level_constraint
from minimappr.core.track_partial_binding import PartialBindingCandidate, choose_track_for_partial_constraints
from minimappr.core.track_partial_binding import PartialBindingConfig
from minimappr.sim.synthetic_entity_scenarios import DominantFrequencySpeciesClassifier

SAMPLE_RATE_HZ = 24_000
BANDS = {"American Robin": (1900.0, 2700.0), "Scarlet Tanager": (3800.0, 4600.0)}


class _FakeOrchestrator:
    """Classifier = spectral-peak oracle; relabel = plain dataclass replace."""

    def __init__(self) -> None:
        self.classifier = DominantFrequencySpeciesClassifier(BANDS)
        self.rerun_count = 0

    async def rerun_classifier_on_prepared_signal(self, signal, sample_rate_hz):
        self.rerun_count += 1
        return self.classifier.classify(signal, sample_rate_hz)

    async def relabel_classified_result(self, classified, classification, *, event_time_ns):
        return dataclasses.replace(classified, classification=classification, label_category="wildlife")


def _signal(robin_level=0.05, tanager_level=0.2) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(SAMPLE_RATE_HZ) / SAMPLE_RATE_HZ
    robin = robin_level * np.sin(2 * np.pi * 2300 * t) * (t < 0.5)
    tanager = tanager_level * np.sin(2 * np.pi * 4200 * t) * ((t >= 0.3) & (t < 0.9))
    return (robin + tanager + 0.001 * rng.standard_normal(t.size)).astype(np.float32)


def _classified(signal) -> ClassifiedResult:
    classification = DominantFrequencySpeciesClassifier(BANDS).classify(signal, SAMPLE_RATE_HZ)
    return ClassifiedResult(
        classification=classification, omni_classification=classification, beamformed_classification=classification,
        classification_path="beamformed:test", classification_signal=signal, beamforming_error=None,
        label_category="wildlife", iff_category="friendly", label_id=None,
    )


def _single_array_context(window_id="w1") -> ClusterAttributionContext:
    robin = ClusterAttributionEvidence({"a": 30.0}, {"a": 14.0}, {"a": (ClusterOnsetArrival(0, 2200.0, 2400.0),)}, 53.0, None)
    tanager = ClusterAttributionEvidence(
        {"a": 40.0}, {"a": 20.0}, {"a": (ClusterOnsetArrival(int(0.3 * SAMPLE_RATE_HZ), 4100.0, 4300.0),)}, 66.0, None
    )
    return ClusterAttributionContext(
        window_id=window_id, target_index=0, cluster_evidence={0: robin, 1: tanager}, classifying_node_id="a",
        classification_start_sample=0, classification_end_sample=SAMPLE_RATE_HZ, sample_rate_hz=SAMPLE_RATE_HZ,
    )


def _attributor(**overrides):
    settings = Settings(onset_attribution_enabled=True, **overrides)
    metrics = FusionMetrics()
    orchestrator = _FakeOrchestrator()
    return OnsetClusterAttributor(settings.onset_cluster_attribution_config(), orchestrator, lambda: metrics), metrics, orchestrator


@pytest.mark.asyncio
async def test_loco_relabels_the_louder_interferers_window_with_the_target_species() -> None:
    attributor, metrics, orchestrator = _attributor()
    signal = _signal()
    classified = _classified(signal)
    assert classified.classification.label == "Scarlet Tanager"  # the shared window's louder bird
    result = await attributor.attribute(classified, _single_array_context(), event_time_ns=0)
    assert result.classification.label == "American Robin"
    annotation = result.classification.features["onset_attribution"]
    assert annotation["tier"] == "loco" and "Scarlet Tanager" in annotation["loco_collapsed_labels"]
    assert metrics.onset_attribution_loco_settled_count == 1 and metrics.onset_attribution_loco_rerun_count == 1
    assert metrics.onset_attribution_loco_rerun_total_ms > 0.0
    # The original (unmasked) audio stays the result's signal: masked audio never reaches a snippet.
    assert result.classification_signal is signal


@pytest.mark.asyncio
async def test_rerun_cap_is_per_window_and_beyond_it_clusters_stay_unlabelled() -> None:
    attributor, metrics, orchestrator = _attributor(onset_attribution_loco_max_reruns_per_window=1)
    first = await attributor.attribute(_classified(_signal()), _single_array_context("w1"), event_time_ns=0)
    second = await attributor.attribute(_classified(_signal()), _single_array_context("w1"), event_time_ns=0)
    other_window = await attributor.attribute(_classified(_signal()), _single_array_context("w2"), event_time_ns=0)
    assert first.classification.label == "American Robin"
    assert second.classification.label == "unknown" and second.classification.confidence == 0.0
    assert second.classification.features["onset_attribution"]["reason"] == "loco_rerun_cap"
    assert other_window.classification.label == "American Robin"
    assert orchestrator.rerun_count == 2
    assert metrics.onset_attribution_loco_rerun_cap_hit_count == 1


@pytest.mark.asyncio
async def test_unsettled_attribution_leaves_the_cluster_unlabelled_instead_of_guessing() -> None:
    attributor, metrics, _ = _attributor()
    # Target band fully overlaps the interferer's mask (same band, same time): masking
    # erases the target too, so the rerun abstains and no species is assigned.
    context = _single_array_context()
    same_band = ClusterAttributionEvidence(
        {"a": 40.0}, {"a": 20.0}, {"a": (ClusterOnsetArrival(0, 2000.0, 2600.0),)}, 66.0, None
    )
    context = dataclasses.replace(context, cluster_evidence={0: context.cluster_evidence[0], 1: same_band})
    signal = _signal(robin_level=0.2, tanager_level=0.0)
    result = await attributor.attribute(_classified(signal), context, event_time_ns=0)
    assert result.classification.label == "unknown"
    annotation = result.classification.features["onset_attribution"]
    assert annotation["tier"] == "unsettled" and annotation["unattributed_scores"]
    assert metrics.onset_attribution_unlabelled_count == 1


@pytest.mark.asyncio
async def test_loco_never_touches_disk(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError(f"attribution must not write to disk: {args!r}")

    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            refuse(file, mode)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    # pathlib / soundfile-style writers go through io.open, which is a separate binding.
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(os, "open", refuse)
    monkeypatch.setattr(np, "save", refuse)
    monkeypatch.setattr(np, "savez", refuse)
    # The real orchestrator's rerun path (worker thread + timeout), not just the fake.
    orchestrator = _real_orchestrator()
    metrics = FusionMetrics()
    settings = Settings(onset_attribution_enabled=True)
    attributor = OnsetClusterAttributor(settings.onset_cluster_attribution_config(), orchestrator, lambda: metrics)
    signal = _signal()
    result = await attributor.attribute(_classified(signal), _single_array_context(), event_time_ns=0)
    assert result.classification.label == "American Robin"
    assert metrics.onset_attribution_loco_rerun_count == 1
    assert result.classification_signal is signal  # the snippet source stays the unmasked window


class _LabelStorageStub:
    async def upsert_label(self, *, name, category, source, created_ns):
        return f"label-{name}"


class _TaxonomyStub:
    def category_for_label(self, label):
        return "wildlife"

    def iff_for_category(self, category):
        return "friendly"


class _OracleClassifier:
    def __init__(self) -> None:
        self._oracle = DominantFrequencySpeciesClassifier(BANDS)

    def classify(self, samples, sample_rate_hz):
        return self._oracle.classify(samples, sample_rate_hz)


def _real_orchestrator():
    from minimappr.core.classification import ClassificationOrchestrator

    return ClassificationOrchestrator(
        classifier=_OracleClassifier(),
        storage=_LabelStorageStub(),
        taxonomy_provider=_TaxonomyStub(),
        environment_provider=None,
    )


@pytest.mark.asyncio
async def test_relabel_keeps_beamformed_alias_in_step_with_the_attributed_label() -> None:
    orchestrator = _real_orchestrator()
    metrics = FusionMetrics()
    attributor = OnsetClusterAttributor(
        Settings(onset_attribution_enabled=True).onset_cluster_attribution_config(), orchestrator, lambda: metrics
    )
    classified = _classified(_signal())
    result = await attributor.attribute(classified, _single_array_context(), event_time_ns=0)
    assert result.classification.label == "American Robin" and result.label_id == "label-American Robin"
    # beamformed_label / omni fields must not keep reporting the rejected interferer label.
    assert result.beamformed_classification is result.classification
    assert result.omni_classification is result.classification


@pytest.mark.asyncio
async def test_window_without_other_clusters_is_left_alone() -> None:
    attributor, metrics, orchestrator = _attributor()
    context = _single_array_context()
    context = dataclasses.replace(context, cluster_evidence={0: context.cluster_evidence[0]})
    classified = _classified(_signal())
    result = await attributor.attribute(classified, context, event_time_ns=0)
    assert result is classified and orchestrator.rerun_count == 0
    assert metrics.onset_attribution_single_cluster_count == 1


@pytest.mark.asyncio
async def test_failed_classification_is_not_attributed() -> None:
    attributor, metrics, orchestrator = _attributor()
    classified = _classified(_signal())
    classified.classification = ClassificationResult(
        label="unknown", confidence=0.0, scores={}, features={"reason": "classification_error"}
    )
    assert await attributor.attribute(classified, _single_array_context(), event_time_ns=0) is classified
    assert metrics.onset_attribution_window_count == 0


def test_measured_level_constraint_prefers_the_track_the_gradient_points_at() -> None:
    centres = {"a": np.array([0.0, 0.0, 1.5]), "b": np.array([20.0, 0.0, 1.5]), "c": np.array([10.0, 18.0, 1.5])}
    bird = np.array([3.0, 3.0, 3.0])
    snr = {node: 60.0 - 20.0 * np.log10(np.linalg.norm(bird - pos)) for node, pos in centres.items()}
    constraint = measured_level_constraint(
        node_band_snr_db=snr, node_centres_m=centres, audible_snr_db=3.0, level_margin_db=7.0
    )
    assert constraint is not None
    assert constraint.normalized_residual(bird) < 0.1
    assert constraint.normalized_residual(np.array([18.0, 2.0, 3.0])) > 1.0
    decision = choose_track_for_partial_constraints(
        timestamp_ns=10,
        detection_top_species_labels=["Song Sparrow"],
        constraints=[constraint],
        candidates=[
            PartialBindingCandidate("near-a", bird, 5, ["Song Sparrow"]),
            PartialBindingCandidate("near-b", np.array([18.0, 2.0, 3.0]), 5, ["Song Sparrow"]),
        ],
        config=PartialBindingConfig(),
    )
    assert decision is not None and decision.track_id == "near-a"
    # One node's reading is not a gradient: keep the stand-in.
    assert measured_level_constraint(
        node_band_snr_db={"a": 20.0}, node_centres_m=centres, audible_snr_db=3.0, level_margin_db=7.0
    ) is None
