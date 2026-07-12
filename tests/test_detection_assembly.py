import numpy as np

from minimappr.core.assembly import _collapse_long_exact_zero_runs


from minimappr.core.assembly import _drone_head_retains_audio


def test_drone_head_retains_audio_for_positive_classes() -> None:
    assert _drone_head_retains_audio("drone_head", "drone")
    assert _drone_head_retains_audio("drone_head", "coyote")
    assert _drone_head_retains_audio("drone_head", "Coyote")


def test_drone_head_discards_audio_for_negative_labels() -> None:
    assert not _drone_head_retains_audio("drone_head", "unknown")
    assert not _drone_head_retains_audio("drone_head", "ambient")
    assert not _drone_head_retains_audio("drone_head", "no_drone")


def test_non_drone_head_source_always_retains() -> None:
    assert _drone_head_retains_audio("yamnet", "unknown")


def test_collapse_long_exact_zero_runs_removes_only_long_gaps() -> None:
    sample_rate_hz = 100
    # 1.0, 1.0, then 50 ms zero gap, then 1.0, then 10 ms zero gap, then 1.0
    signal = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    compacted = _collapse_long_exact_zero_runs(
        signal,
        sample_rate_hz=sample_rate_hz,
        min_gap_seconds=0.03,
    )

    # The 5-sample zero run (50 ms) is removed; the 1-sample run (10 ms) remains.
    assert compacted.tolist() == [1.0, 1.0, 1.0, 0.0, 1.0]


def test_collapse_long_exact_zero_runs_smooths_high_rate_splice() -> None:
    sample_rate_hz = 1_000
    signal = np.array(
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0],
        dtype=np.float32,
    )

    compacted = _collapse_long_exact_zero_runs(
        signal,
        sample_rate_hz=sample_rate_hz,
        min_gap_seconds=0.003,
        crossfade_seconds=0.002,
    )

    assert compacted.tolist() == [1.0, 0.0, -0.0, -1.0]
    assert np.max(np.abs(np.diff(compacted))) <= 1.0


def test_collapse_long_exact_zero_runs_keeps_signal_when_no_long_gap() -> None:
    sample_rate_hz = 100
    signal = np.array([0.25, 0.0, -0.1, 0.0, 0.4], dtype=np.float32)

    compacted = _collapse_long_exact_zero_runs(
        signal,
        sample_rate_hz=sample_rate_hz,
        min_gap_seconds=0.05,
    )

    np.testing.assert_allclose(compacted, signal)
