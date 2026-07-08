from __future__ import annotations

from tests.spatial_audio.scene_harness import (
    build_weak_directional_foa,
    build_two_object_bed,
    encode_scene_profile,
    evaluate_foa_metrics,
    evaluate_object_subtraction_depth,
    synthesize_diffuse_noise,
    synthesize_plane_wave_scene,
)
from minimappr.spatial_audio.parametric import enhance_foa_parametric
from minimappr.spatial_audio.profiles import PARAMETRIC_V2


def test_scene_harness_records_plane_wave_metrics_without_clicks() -> None:
    scene = synthesize_plane_wave_scene()
    linear = encode_scene_profile(scene, "linear_v1")
    parametric = encode_scene_profile(scene, "parametric_v2")

    linear_metrics = evaluate_foa_metrics(
        linear,
        source_direction=scene.source_direction,
        sample_rate_hz=scene.sample_rate_hz,
    )
    parametric_metrics = evaluate_foa_metrics(
        parametric,
        source_direction=scene.source_direction,
        sample_rate_hz=scene.sample_rate_hz,
    )

    assert parametric_metrics.ideal_xyz_to_w_db_error < linear_metrics.ideal_xyz_to_w_db_error
    assert parametric_metrics.doa_error_deg <= 35.0
    assert parametric_metrics.virtual_speaker_directivity_db >= linear_metrics.virtual_speaker_directivity_db - 1.0
    assert parametric_metrics.click_count == 0


def test_scene_harness_records_weak_directional_improvement() -> None:
    weak_foa, direction, sample_rate_hz = build_weak_directional_foa()
    parametric = enhance_foa_parametric(weak_foa, sample_rate_hz, profile=PARAMETRIC_V2)
    weak_metrics = evaluate_foa_metrics(
        weak_foa,
        source_direction=direction,
        sample_rate_hz=sample_rate_hz,
    )
    parametric_metrics = evaluate_foa_metrics(
        parametric,
        source_direction=direction,
        sample_rate_hz=sample_rate_hz,
    )

    assert parametric_metrics.xyz_to_w_db - weak_metrics.xyz_to_w_db >= 6.0
    assert parametric_metrics.ideal_xyz_to_w_db_error < weak_metrics.ideal_xyz_to_w_db_error
    assert parametric_metrics.click_count == 0


def test_scene_harness_diffuse_noise_is_finite_and_click_free() -> None:
    channels = synthesize_diffuse_noise()
    scene = synthesize_plane_wave_scene()
    mixed_scene = scene.__class__(
        channels=(scene.channels * 0.75 + channels * 0.25).astype("float32"),
        source_direction=scene.source_direction,
        mono=scene.mono,
        sample_rate_hz=scene.sample_rate_hz,
    )
    parametric = encode_scene_profile(mixed_scene, "parametric_v2")
    metrics = evaluate_foa_metrics(
        parametric,
        source_direction=mixed_scene.source_direction,
        sample_rate_hz=mixed_scene.sample_rate_hz,
    )
    assert metrics.click_count == 0
    assert metrics.xyz_to_w_db > -12.0


def test_object_subtraction_depth_metric_passes_plan_gate() -> None:
    sample_rate_hz = 16_000
    bed_full, selected_bed, unselected_bed = build_two_object_bed(sample_rate_hz=sample_rate_hz)
    metrics = evaluate_object_subtraction_depth(
        bed_full=bed_full,
        object_bed=selected_bed,
        unselected_bed=unselected_bed,
        sample_rate_hz=sample_rate_hz,
    )
    assert metrics.subtraction_depth_db >= 20.0
    assert abs(metrics.unselected_retention_db) <= 1.0
    assert metrics.click_count == 0
