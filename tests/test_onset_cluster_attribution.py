"""Field Mode W3-attribution: SNR-gradient decision, LOCO decision, noise-floor
masking, per-cluster level evidence and the band-distinct persistence promotion."""

from __future__ import annotations

import math

import numpy as np

from minimappr.config import Settings
from minimappr.core.onset_cluster_attribution import (
    AttributionThresholds,
    interferers_in_window,
    is_band_distinct_persistence_hold,
    loco_decision,
    snr_gradient_decision,
)
from minimappr.core.onset_cluster_attribution_evidence import (
    ClusterAttributionEvidence,
    ClusterOnsetArrival,
    build_cluster_attribution_evidence,
)
from minimappr.core.onset_cluster_masking import MaskingParameters, MaskRegionSeed, mask_regions_with_noise_floor
from minimappr.core.onset_detection import build_subband_history
from minimappr.core.onset_srp_types import OnsetFrame

SAMPLE_RATE_HZ = 24_000
THRESHOLDS = AttributionThresholds(
    audible_snr_db=3.0,
    dominance_margin_db=10.0,
    spreading_tolerance_db=6.0,
    loco_min_confidence=0.3,
    loco_label_margin=0.1,
    loco_collapse_ratio=0.5,
)
NODES = {"a": np.array([0.0, 0.0, 1.5]), "b": np.array([20.0, 0.0, 1.5]), "c": np.array([10.0, 18.0, 1.5])}


def _evidence(source_m, source_level_db, onset_sample=1000, band=(2000.0, 3000.0), measured_offsets=None):
    """Evidence obeying 1/r spreading exactly (plus optional per-node offsets)."""
    distances = {node: float(np.linalg.norm(np.asarray(source_m) - pos)) for node, pos in NODES.items()}
    snr = {node: source_level_db - 20.0 * math.log10(d) for node, d in distances.items()}
    for node, offset in (measured_offsets or {}).items():
        snr[node] += offset
    residual = float(np.sqrt(np.mean(np.square(list((measured_offsets or {"a": 0.0}).values())))))
    return ClusterAttributionEvidence(
        node_band_snr_db=snr,
        node_distances_m=distances,
        node_onset_arrivals={node: (ClusterOnsetArrival(onset_sample, *band),) for node in NODES},
        fitted_source_level_db=source_level_db,
        spreading_residual_db=residual,
    )


def _interferers(target_evidence, other_evidence, node="a"):
    return interferers_in_window(
        target_index=0,
        cluster_evidence={0: target_evidence, 1: other_evidence},
        classifying_node_id=node,
        window_start_sample=0,
        window_end_sample=SAMPLE_RATE_HZ,
        sample_rate_hz=SAMPLE_RATE_HZ,
        audible_snr_db=THRESHOLDS.audible_snr_db,
    )


def test_gradient_settles_when_target_dominates_its_node_as_spreading_predicts() -> None:
    near_a = _evidence([2.0, 2.0, 2.0], 60.0)
    near_b = _evidence([19.0, -2.0, 2.0], 60.0)
    interferers = _interferers(near_a, near_b)
    assert len(interferers) == 1
    decision = snr_gradient_decision(
        target=near_a, interferers=interferers, classifying_node_id="a", thresholds=THRESHOLDS
    )
    assert decision.settled, decision
    assert decision.dominance_db[1] > THRESHOLDS.dominance_margin_db


def test_gradient_refuses_when_interferer_is_louder_on_the_classifying_node() -> None:
    near_a = _evidence([2.0, 2.0, 2.0], 60.0)
    near_b = _evidence([19.0, -2.0, 2.0], 60.0)
    decision = snr_gradient_decision(
        target=near_b, interferers=_interferers(near_b, near_a), classifying_node_id="a", thresholds=THRESHOLDS
    )
    assert not decision.settled and decision.reason == "not_dominant"


def test_gradient_refuses_when_levels_do_not_fit_the_positions() -> None:
    # Target reads 15 dB hot on node a relative to its position (occlusion / mixed energy).
    target = _evidence([10.0, 8.0, 2.0], 50.0, measured_offsets={"a": 15.0, "b": -7.0, "c": -8.0})
    interferer = _evidence([19.0, -2.0, 2.0], 45.0)
    decision = snr_gradient_decision(
        target=target, interferers=_interferers(target, interferer), classifying_node_id="a", thresholds=THRESHOLDS
    )
    assert not decision.settled
    assert decision.reason in {"spreading_inconsistent", "dominance_not_predicted"}


def test_gradient_never_settles_a_single_array_window() -> None:
    target = ClusterAttributionEvidence({"a": 40.0}, {"a": 14.0}, {"a": (ClusterOnsetArrival(10, 2000, 2600),)}, 63.0, None)
    other = ClusterAttributionEvidence({"a": 12.0}, {"a": 20.0}, {"a": (ClusterOnsetArrival(500, 3900, 4500),)}, 38.0, None)
    decision = snr_gradient_decision(
        target=target, interferers=_interferers(target, other), classifying_node_id="a", thresholds=THRESHOLDS
    )
    assert not decision.settled and decision.reason == "no_cross_node_gradient"


def test_interferer_must_be_audible_and_inside_the_window() -> None:
    target = _evidence([2.0, 2.0, 2.0], 60.0)
    inaudible = _evidence([19.0, -2.0, 2.0], 10.0)
    assert _interferers(target, inaudible) == []
    late = _evidence([19.0, -2.0, 2.0], 60.0, onset_sample=2 * SAMPLE_RATE_HZ)
    assert _interferers(target, late) == []


def test_interferer_onsets_shared_with_the_target_note_are_not_masked() -> None:
    # A ground/trunk image of the target: same band, onset 10 ms behind the target's.
    target = _evidence([2.0, 2.0, 2.0], 60.0, onset_sample=1000)
    image = _evidence([2.5, 2.5, 0.3], 50.0, onset_sample=1240)
    common = dict(
        target_index=0, cluster_evidence={0: target, 1: image}, classifying_node_id="a", window_start_sample=0,
        window_end_sample=SAMPLE_RATE_HZ, sample_rate_hz=SAMPLE_RATE_HZ, audible_snr_db=3.0,
    )
    assert len(interferers_in_window(**common)) == 1  # legacy view: masks the target's own note
    assert interferers_in_window(
        **common, shared_onset_max_lag_samples=int(0.03 * SAMPLE_RATE_HZ), same_band_min_overlap_fraction=0.5
    ) == []


def test_loco_settles_target_label_and_reports_the_collapsed_interferer_label() -> None:
    decision = loco_decision(
        original_label="Scarlet Tanager",
        original_scores={"Scarlet Tanager": 0.82, "American Robin": 0.7},
        masked_label="American Robin",
        masked_confidence=0.88,
        masked_scores={"American Robin": 0.88, "Scarlet Tanager": 0.04},
        thresholds=THRESHOLDS,
    )
    assert decision.settled and decision.label == "American Robin"
    assert "Scarlet Tanager" in decision.collapsed_labels


def test_loco_refuses_rather_than_guess() -> None:
    abstained = loco_decision(
        original_label="A", original_scores={"A": 0.9}, masked_label="background", masked_confidence=0.05,
        masked_scores={"background": 0.05}, thresholds=THRESHOLDS,
    )
    assert not abstained.settled and abstained.reason == "masked_rerun_abstained"
    survived = loco_decision(
        original_label="A", original_scores={"A": 0.9}, masked_label="B", masked_confidence=0.9,
        masked_scores={"B": 0.9, "A": 0.85}, thresholds=THRESHOLDS,
    )
    assert not survived.settled and survived.reason == "masked_rerun_ambiguous"
    survived_clear = loco_decision(
        original_label="A", original_scores={"A": 0.9}, masked_label="B", masked_confidence=0.95,
        masked_scores={"B": 0.95, "A": 0.6}, thresholds=THRESHOLDS,
    )
    assert not survived_clear.settled and survived_clear.reason == "original_label_survived"
    weak = loco_decision(
        original_label="A", original_scores={"A": 0.9}, masked_label="A", masked_confidence=0.1,
        masked_scores={"A": 0.1}, thresholds=THRESHOLDS,
    )
    assert not weak.settled and weak.reason == "masked_rerun_not_confident"


def _band_power(signal, low_hz, high_hz, start_s, stop_s):
    segment = signal[int(start_s * SAMPLE_RATE_HZ) : int(stop_s * SAMPLE_RATE_HZ)]
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size))) ** 2
    frequencies = np.fft.rfftfreq(segment.size, 1.0 / SAMPLE_RATE_HZ)
    return float(np.mean(spectrum[(frequencies >= low_hz) & (frequencies <= high_hz)]))


def test_masking_keeps_target_and_replaces_interferer_with_noise_at_the_floor() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(SAMPLE_RATE_HZ) / SAMPLE_RATE_HZ
    noise = 0.002 * rng.standard_normal(t.size)
    target = 0.1 * np.sin(2 * np.pi * 2000 * t) * (t < 0.9)
    interferer = 0.1 * np.sin(2 * np.pi * 4000 * t) * ((t >= 0.4) & (t < 0.7))
    mixture = (noise + target + interferer).astype(np.float32)
    parameters = MaskingParameters(0.02, 20.0, 6.0, 0.05, 1.0, 150.0, 0.01)
    masked = mask_regions_with_noise_floor(
        mixture, sample_rate_hz=SAMPLE_RATE_HZ, seeds=[MaskRegionSeed(0.4, 3900.0, 4100.0)],
        parameters=parameters, rng=np.random.default_rng(1),
    )
    assert masked.samples.dtype == np.float32 and masked.samples.shape == mixture.shape
    assert 0.25 <= masked.region_seconds[0] <= 0.45  # tracks the 0.3 s note, not the cap
    target_ratio = _band_power(masked.samples, 1900, 2100, 0.1, 0.8) / _band_power(mixture, 1900, 2100, 0.1, 0.8)
    assert abs(10 * np.log10(target_ratio)) < 1.0
    noise_only = _band_power(noise, 3800, 4200, 0.45, 0.65)
    replaced = _band_power(masked.samples, 3800, 4200, 0.45, 0.65)
    assert abs(10 * np.log10(replaced / noise_only)) < 3.0  # at the floor ...
    assert replaced > 0.0  # ... and never a hole of zeros


def test_masking_ignores_a_note_that_ended_before_the_signal_started() -> None:
    rng = np.random.default_rng(2)
    mixture = 0.002 * rng.standard_normal(SAMPLE_RATE_HZ)
    parameters = MaskingParameters(0.02, 20.0, 6.0, 0.05, 1.0, 150.0, 0.01)
    masked = mask_regions_with_noise_floor(
        mixture, sample_rate_hz=SAMPLE_RATE_HZ, seeds=[MaskRegionSeed(-0.3, 3900.0, 4100.0)],
        parameters=parameters, rng=rng,
    )
    assert masked.masked_cell_fraction == 0.0


def test_evidence_levels_follow_the_nearest_node_and_fit_spreading() -> None:
    rng = np.random.default_rng(3)
    source = np.array([3.0, 4.0, 1.5])
    t = np.arange(int(0.6 * SAMPLE_RATE_HZ)) / SAMPLE_RATE_HZ
    histories = {}
    for node, position in NODES.items():
        distance = float(np.linalg.norm(source - position))
        call = (1.0 / distance) * np.sin(2 * np.pi * 2500 * t) * ((t > 0.2) & (t < 0.4))
        histories[node] = build_subband_history(
            call + 0.0005 * rng.standard_normal(t.size), sample_rate_hz=SAMPLE_RATE_HZ, hop_seconds=0.01,
            band_min_hz=500.0, band_max_hz=10000.0, band_count=32,
        )
    frame = OnsetFrame(0, "a", "a:ch0", int(0.2 * SAMPLE_RATE_HZ), 10.0, 2300.0, 2700.0, [])
    evidence = build_cluster_attribution_evidence(
        position_m=source, led_frames=[frame], node_histories=histories, node_positions_m=NODES,
        sound_speed_mps=343.0, sample_rate_hz=SAMPLE_RATE_HZ, frame_samples=int(0.08 * SAMPLE_RATE_HZ),
        audible_snr_db=3.0,
    )
    snr = evidence.node_band_snr_db
    assert snr["a"] > snr["b"] and snr["a"] > snr["c"]
    assert evidence.spreading_residual_db is not None and evidence.spreading_residual_db < 3.0
    later = evidence.node_onset_arrivals["b"][0].span_sample - evidence.node_onset_arrivals["a"][0].span_sample
    assert later > 0  # node b is farther, so it hears the onset later


def test_band_distinct_persistence_hold_is_promotable_but_echo_like_holds_are_not() -> None:
    accepted = [(2425.0, 4838.0)]
    assert is_band_distinct_persistence_hold(
        status_reasons=("not_persistent",), band_hz=(1804.0, 2197.0), accepted_bands_hz=accepted,
        same_band_min_overlap_fraction=0.5,
    )
    assert not is_band_distinct_persistence_hold(
        status_reasons=("not_persistent",), band_hz=(2600.0, 4000.0), accepted_bands_hz=accepted,
        same_band_min_overlap_fraction=0.5,
    )
    assert not is_band_distinct_persistence_hold(
        status_reasons=("level_inconsistent",), band_hz=(1804.0, 2197.0), accepted_bands_hz=accepted,
        same_band_min_overlap_fraction=0.5,
    )


def test_attribution_settings_default_off_and_validate() -> None:
    settings = Settings()
    config = settings.onset_cluster_attribution_config()
    assert config.enabled is False and config.loco_enabled is True
    assert config.loco_max_reruns_per_window >= 1
