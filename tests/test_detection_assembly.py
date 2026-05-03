import numpy as np

from minimappr.core.assembly import _collapse_long_exact_zero_runs


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


def test_collapse_long_exact_zero_runs_keeps_signal_when_no_long_gap() -> None:
    sample_rate_hz = 100
    signal = np.array([0.25, 0.0, -0.1, 0.0, 0.4], dtype=np.float32)

    compacted = _collapse_long_exact_zero_runs(
        signal,
        sample_rate_hz=sample_rate_hz,
        min_gap_seconds=0.05,
    )

    np.testing.assert_allclose(compacted, signal)
