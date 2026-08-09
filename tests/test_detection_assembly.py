import numpy as np

from minimappr.core.assembly import _collapse_long_exact_zero_runs


from minimappr.core.assembly import _classifier_retains_audio


def test_drone_head_retains_audio_for_positive_classes() -> None:
    assert _classifier_retains_audio("drone_head", "drone")
    assert _classifier_retains_audio("drone_head", "coyote")
    assert _classifier_retains_audio("drone_head", "Coyote")


def test_drone_head_discards_audio_for_negative_labels() -> None:
    assert not _classifier_retains_audio("drone_head", "unknown")
    assert not _classifier_retains_audio("drone_head", "ambient")
    assert not _classifier_retains_audio("drone_head", "no_drone")


def test_classifier_without_negative_labels_always_retains() -> None:
    assert _classifier_retains_audio("yamnet", "unknown")
    assert _classifier_retains_audio("yamnet", "Silence")


def test_birdnet_discards_only_the_unknown_negative() -> None:
    assert not _classifier_retains_audio("birdnet", "unknown")
    assert not _classifier_retains_audio("birdnet", "Unknown")
    # Sub-threshold but genuinely labelled results are the review corpus.
    assert _classifier_retains_audio("birdnet", "house sparrow")


def test_label_prefix_wins_over_a_mismatched_winner_member() -> None:
    # A chained drone-head result can be attributed to the upstream member.
    assert not _classifier_retains_audio("yamnet", "drone_head:ambient")
    assert not _classifier_retains_audio("yamnet", "drone_head:unknown")
    assert _classifier_retains_audio("yamnet", "drone_head:coyote")
    assert not _classifier_retains_audio("yamnet", "birdnet:unknown")


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
