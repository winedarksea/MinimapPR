"""Cross-node beamform + late fusion (BEAMFORMED_RENDER_CONTRACT Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.ambi_atob import SIRITH_MIC_POSITIONS_M
from minimappr.core.cross_node_beamform import (
    CrossNodeBeamConfig,
    CrossNodeBeamformer,
    NodeAudioCandidate,
    fuse_label_scores,
    position_source_trusted,
    select_nodes,
    world_to_node_local,
)
from minimappr.models import NodeOrientation, SyncGrade

SAMPLE_RATE_HZ = 16_000
SOUND_SPEED_MPS = 343.2
N_SAMPLES = int(SAMPLE_RATE_HZ * 0.25)

TARGET_POS = (5.0, 0.0, 1.0)
TARGET_FREQ = 800.0
INTERFERER_POS = (30.0, 25.0, 1.0)
INTERFERER_FREQ = 1900.0


def _node_candidate(
    node_id: str,
    node_position: tuple[float, float, float],
    *,
    sync_grade: SyncGrade = SyncGrade.GPS_PPS,
    position_trusted: bool = True,
    covers_event: bool = True,
) -> NodeAudioCandidate:
    """Synthesize tetra-array audio from both scene sources with true delays."""
    node_pos = np.asarray(node_position, dtype=np.float64)
    mics_local = np.asarray(SIRITH_MIC_POSITIONS_M, dtype=np.float64)
    t = np.arange(N_SAMPLES, dtype=np.float64) / SAMPLE_RATE_HZ
    rng = np.random.default_rng(hash(node_id) % (2**32))
    channels = np.zeros((mics_local.shape[0], N_SAMPLES), dtype=np.float64)
    for src_pos, freq, amp in (
        (TARGET_POS, TARGET_FREQ, 0.3),
        (INTERFERER_POS, INTERFERER_FREQ, 0.3),
    ):
        src = np.asarray(src_pos, dtype=np.float64)
        for m in range(mics_local.shape[0]):
            mic_world = node_pos + mics_local[m]
            distance = float(np.linalg.norm(src - mic_world))
            # 1/r amplitude falloff (normalized at 5 m) + true delay.
            gain = amp * 5.0 / max(distance, 1.0)
            channels[m] += gain * np.sin(
                2.0 * np.pi * freq * (t - distance / SOUND_SPEED_MPS)
            )
    channels += 0.01 * rng.standard_normal(channels.shape)
    return NodeAudioCandidate(
        node_id=node_id,
        node_position_m=node_pos,
        orientation=None,
        mic_positions_local_m=mics_local,
        channels=channels.astype(np.float32),
        sample_rate_hz=SAMPLE_RATE_HZ,
        sync_grade=sync_grade,
        position_trusted=position_trusted,
        covers_event=covers_event,
    )


def _tone_ratio_classifier() -> "callable":
    async def classify(samples: np.ndarray, sample_rate_hz: int) -> dict[str, float]:
        spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate_hz)
        band = lambda f0: float(np.sum(spectrum[(freqs > f0 - 50) & (freqs < f0 + 50)] ** 2))
        target_energy = band(TARGET_FREQ)
        interferer_energy = band(INTERFERER_FREQ)
        total = target_energy + interferer_energy + 1e-12
        return {
            "target": target_energy / total,
            "interferer": interferer_energy / total,
        }

    return classify


def test_select_nodes_filters_and_ranks() -> None:
    config = CrossNodeBeamConfig(enabled=True, max_range_m=75.0, max_nodes=2)
    near = _node_candidate("near", (2.0, 0.0, 0.0))
    mid = _node_candidate("mid", (20.0, 0.0, 0.0))
    far = _node_candidate("far", (200.0, 0.0, 0.0))
    free_running = _node_candidate("free", (3.0, 0.0, 0.0), sync_grade=SyncGrade.FREE)
    static_pos = _node_candidate("static", (3.0, 0.0, 0.0), position_trusted=False)
    no_audio = _node_candidate("gap", (3.0, 0.0, 0.0), covers_event=False)

    selected = select_nodes(
        [far, mid, near, free_running, static_pos, no_audio],
        TARGET_POS,
        config,
    )
    ids = [candidate.node_id for candidate, _ in selected]
    assert ids == ["near", "mid"]  # ranked closest-first, capped at max_nodes


def test_world_to_node_local_applies_yaw() -> None:
    # Node at origin yawed +90°: a source due east in world appears rotated
    # into the node frame by the inverse rotation.
    orientation = NodeOrientation(yaw_deg=90.0)
    local = world_to_node_local((10.0, 0.0, 0.0), np.zeros(3), orientation)
    assert abs(np.linalg.norm(local) - 10.0) < 1e-9  # rotation preserves range
    assert abs(local[2]) < 1e-9
    # Inverse of the local→world yaw: the point moves off the local x-axis.
    assert abs(abs(local[1]) - 10.0) < 1e-6 or abs(abs(local[0]) - 10.0) < 1e-6
    assert not (abs(local[0] - 10.0) < 1e-6 and abs(local[1]) < 1e-6)


async def test_two_node_scene_fuses_to_target_label() -> None:
    beamformer = CrossNodeBeamformer(
        CrossNodeBeamConfig(enabled=True, max_range_m=75.0, max_nodes=3)
    )
    node_a = _node_candidate("node-a", (2.0, 1.0, 0.0))  # near target
    node_b = _node_candidate("node-b", (28.0, 23.0, 0.0))  # near interferer

    result = await beamformer.classify_across_nodes(
        candidates=[node_a, node_b],
        source_position_m=TARGET_POS,
        classify_fn=_tone_ratio_classifier(),
    )
    assert result is not None
    assert result["label"] == "target"
    assert result["best_node_id"] == "node-a"
    assert result["contributing_node_ids"] == ["node-a", "node-b"]
    assert result["fusion_method"] == "max_late_fusion"
    assert result["node_count"] == 2
    assert result["canonical_beam"].size > 0
    assert result["canonical_alias_cutoff_hz"] > 3000.0  # tetra geometry


async def test_disabled_config_returns_none() -> None:
    beamformer = CrossNodeBeamformer(CrossNodeBeamConfig(enabled=False))
    result = await beamformer.classify_across_nodes(
        candidates=[_node_candidate("node-a", (2.0, 1.0, 0.0))],
        source_position_m=TARGET_POS,
        classify_fn=_tone_ratio_classifier(),
    )
    assert result is None


async def test_no_eligible_nodes_returns_none() -> None:
    beamformer = CrossNodeBeamformer(CrossNodeBeamConfig(enabled=True))
    result = await beamformer.classify_across_nodes(
        candidates=[
            _node_candidate("free", (2.0, 1.0, 0.0), sync_grade=SyncGrade.FREE),
            _node_candidate("static", (3.0, 0.0, 0.0), position_trusted=False),
        ],
        source_position_m=TARGET_POS,
        classify_fn=_tone_ratio_classifier(),
    )
    assert result is None


def test_fuse_label_scores_max_with_evidence() -> None:
    fused = fuse_label_scores(
        {
            "node-a": {"coyote": 0.9, "owl": 0.2},
            "node-b": {"coyote": 0.5, "owl": 0.7},
        }
    )
    assert fused is not None
    assert fused.label == "coyote"
    assert fused.confidence == pytest.approx(0.9)
    assert fused.best_node_id == "node-a"
    assert fused.contributing_node_ids == ["node-a", "node-b"]
    assert fused.per_node_confidence == {"node-a": 0.9, "node-b": 0.5}


def test_position_source_trust_markers() -> None:
    assert position_source_trusted("gps_fix")
    assert not position_source_trusted("gps_fallback")
    assert not position_source_trusted("fallback_static")
    assert not position_source_trusted("static")
    assert not position_source_trusted(None)
