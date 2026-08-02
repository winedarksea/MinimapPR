"""Classification-stage admission ordering: significance scoring + priority queue.

The classification lane is the pipeline bottleneck and runs oversubscribed, so
these tests pin down *which* work survives a queue that cannot drain — including
the property that matters most operationally: a full queue sheds its least
significant item, never its best one.
"""

from __future__ import annotations

import asyncio

import pytest

from minimappr.config import Settings
from minimappr.core.classification_priority import (
    PriorityStageQueue,
    PriorityWeights,
    corroboration,
    priority_key,
    score_significance,
    signal_excess,
    track_affinity,
)


TRIGGER_RMS = 0.001


def _significance(**overrides) -> float:
    base = dict(
        localization_confidence=0.5,
        capability_tier="2d",
        sensor_count=4,
        signal_rms=TRIGGER_RMS * 2.0,
        trigger_rms=TRIGGER_RMS,
        position_m=(0.0, 0.0, 0.0),
        track_positions_m=(),
        track_radius_m=50.0,
    )
    base.update(overrides)
    return score_significance(**base)


class TestSignificanceTerms:
    def test_track_affinity_decays_to_zero_at_radius(self) -> None:
        assert track_affinity((0.0, 0.0, 0.0), [(0.0, 0.0, 0.0)], radius_m=50.0) == pytest.approx(1.0)
        assert track_affinity((25.0, 0.0, 0.0), [(0.0, 0.0, 0.0)], radius_m=50.0) == pytest.approx(0.5)
        assert track_affinity((50.0, 0.0, 0.0), [(0.0, 0.0, 0.0)], radius_m=50.0) == pytest.approx(0.0)
        assert track_affinity((999.0, 0.0, 0.0), [(0.0, 0.0, 0.0)], radius_m=50.0) == pytest.approx(0.0)

    def test_track_affinity_takes_the_nearest_track(self) -> None:
        tracks = [(400.0, 0.0, 0.0), (10.0, 0.0, 0.0), (200.0, 0.0, 0.0)]
        assert track_affinity((0.0, 0.0, 0.0), tracks, radius_m=50.0) == pytest.approx(0.8)

    def test_no_tracks_is_zero_not_an_error(self) -> None:
        assert track_affinity((0.0, 0.0, 0.0), [], radius_m=50.0) == 0.0
        assert track_affinity(None, [(0.0, 0.0, 0.0)], radius_m=50.0) == 0.0

    def test_signal_excess_is_zero_at_or_below_the_trigger_floor(self) -> None:
        assert signal_excess(TRIGGER_RMS, TRIGGER_RMS) == 0.0
        assert signal_excess(TRIGGER_RMS * 0.5, TRIGGER_RMS) == 0.0
        assert signal_excess(None, TRIGGER_RMS) == 0.0
        assert signal_excess(TRIGGER_RMS * 4.0, TRIGGER_RMS) > 0.0

    def test_signal_excess_saturates_rather_than_dominating(self) -> None:
        """A very loud event must not be able to crowd out everything else."""
        assert signal_excess(TRIGGER_RMS * 10_000.0, TRIGGER_RMS) == pytest.approx(1.0)

    def test_corroboration_rises_with_sensor_count(self) -> None:
        assert corroboration(1) == 0.0
        assert corroboration(4) == pytest.approx(1.0)
        assert corroboration(2) < corroboration(3) < corroboration(4)


class TestSignificanceScore:
    def test_track_association_outranks_an_isolated_event(self) -> None:
        """The headline behaviour: continuing evidence for a live track wins."""
        on_track = _significance(track_positions_m=[(0.0, 0.0, 0.0)])
        isolated = _significance(track_positions_m=[(5_000.0, 0.0, 0.0)])
        assert on_track > isolated

    def test_confidence_and_tier_both_raise_significance(self) -> None:
        assert _significance(localization_confidence=0.9) > _significance(localization_confidence=0.1)
        assert _significance(capability_tier="full_3d") > _significance(capability_tier="2d")
        assert _significance(capability_tier="2d") > _significance(capability_tier="classification_only")

    def test_unknown_tier_does_not_crash_or_win(self) -> None:
        assert 0.0 <= _significance(capability_tier="nonsense") <= 1.0
        assert _significance(capability_tier="nonsense") < _significance(capability_tier="full_3d")

    def test_degraded_audio_halves_the_score(self) -> None:
        assert _significance(audio_degraded=True) == pytest.approx(_significance() * 0.5)

    def test_missing_localization_still_scores(self) -> None:
        """A classification-only product must be orderable, not an exception."""
        score = _significance(
            localization_confidence=None,
            capability_tier=None,
            position_m=None,
        )
        assert 0.0 <= score <= 1.0

    def test_score_is_bounded(self) -> None:
        best = _significance(
            localization_confidence=1.0,
            capability_tier="full_3d",
            sensor_count=99,
            signal_rms=TRIGGER_RMS * 1e6,
            track_positions_m=[(0.0, 0.0, 0.0)],
        )
        worst = score_significance(
            localization_confidence=0.0,
            capability_tier="alerting_only",
            sensor_count=0,
            signal_rms=0.0,
            trigger_rms=TRIGGER_RMS,
            position_m=None,
        )
        assert best == pytest.approx(1.0, abs=1e-6)
        assert worst >= 0.0
        assert best > worst

    def test_zero_weights_do_not_divide_by_zero(self) -> None:
        score = _significance(weights=PriorityWeights(0.0, 0.0, 0.0, 0.0, 0.0))
        assert 0.0 <= score <= 1.0


class TestPriorityKey:
    def test_strong_and_newest_leads_the_whole_ordering(self) -> None:
        """Strongest-and-freshest first, then significance, then recency.

        The headline contract: an item that is both the most significant *and*
        the most recent must sort ahead of everything, including an equally
        significant but older item.
        """
        items = {
            "strong+newest": (0.90, 9_000),
            "strong+oldest": (0.90, 1_000),
            "middling+newest": (0.50, 9_000),
            "weak+newest": (0.10, 9_000),
        }
        order = sorted(
            items,
            key=lambda name: priority_key(
                significance=items[name][0], event_time_ns=items[name][1]
            ),
        )
        assert order == ["strong+newest", "strong+oldest", "middling+newest", "weak+newest"]

    def test_significance_dominates_recency(self) -> None:
        significant_but_old = priority_key(significance=0.9, event_time_ns=1_000)
        trivial_but_fresh = priority_key(significance=0.1, event_time_ns=9_999_999)
        assert significant_but_old < trivial_but_fresh

    def test_recency_breaks_ties_within_a_bucket(self) -> None:
        """Comparable significance ⇒ freshest audio first."""
        newer = priority_key(significance=0.51, event_time_ns=2_000)
        older = priority_key(significance=0.52, event_time_ns=1_000)
        # Both land in the same coarse bucket, so event time decides.
        assert newer[0] == older[0]
        assert newer < older


def _drain(queue: PriorityStageQueue) -> list:
    items = []
    while queue.qsize():
        items.append(queue.get_nowait())
    return items


def _queue(maxsize: int = 4) -> PriorityStageQueue:
    return PriorityStageQueue(
        maxsize=maxsize,
        key=lambda item: priority_key(significance=item[0], event_time_ns=item[1]),
    )


class TestPriorityStageQueue:
    def test_pops_most_significant_first(self) -> None:
        q = _queue()
        q.put_nowait((0.1, 100))
        q.put_nowait((0.9, 100))
        q.put_nowait((0.5, 100))
        assert [item[0] for item in _drain(q)] == [0.9, 0.5, 0.1]

    def test_equal_significance_pops_newest_first(self) -> None:
        q = _queue()
        q.put_nowait((0.5, 100))
        q.put_nowait((0.5, 300))
        q.put_nowait((0.5, 200))
        assert [item[1] for item in _drain(q)] == [300, 200, 100]

    def test_eviction_sheds_the_worst_not_the_best(self) -> None:
        """The whole point: a full queue must not discard its best work."""
        q = _queue(maxsize=3)
        q.put_nowait((0.9, 100))
        q.put_nowait((0.5, 100))
        q.put_nowait((0.1, 100))
        assert q.full()

        evicted = q.evict_least_significant()

        assert evicted == (0.1, 100)
        assert [item[0] for item in _drain(q)] == [0.9, 0.5]

    def test_eviction_keeps_the_heap_valid(self) -> None:
        q = _queue(maxsize=32)
        for i in range(20):
            q.put_nowait((round((i % 7) / 10.0, 1), 1_000 + i))
        for _ in range(8):
            q.evict_least_significant()
        popped = [item[0] for item in _drain(q)]
        assert popped == sorted(popped, reverse=True)

    def test_eviction_on_empty_queue_is_none(self) -> None:
        assert _queue().evict_least_significant() is None

    def test_shutdown_sentinel_sorts_first_and_is_never_evicted(self) -> None:
        q = _queue(maxsize=4)
        q.put_nowait((0.9, 100))
        q.put_nowait(None)
        q.put_nowait((0.1, 100))

        # The sentinel outranks even the most significant work, so a stop
        # request is never stuck behind a saturated queue.
        assert q.get_nowait() is None

        q.put_nowait(None)
        # Drain everything except the sentinel; eviction must then refuse.
        while q.qsize() > 1:
            q.evict_least_significant()
        assert q.evict_least_significant() is None
        assert q.get_nowait() is None

    @pytest.mark.asyncio
    async def test_join_and_task_done_accounting_survive_eviction(self) -> None:
        """queue.join() must not deadlock when items leave via eviction."""
        q = _queue(maxsize=2)
        q.put_nowait((0.9, 100))
        q.put_nowait((0.1, 100))

        assert q.evict_least_significant() is not None
        q.task_done()
        assert q.get_nowait() is not None
        q.task_done()

        await asyncio.wait_for(q.join(), timeout=1.0)


class TestPrioritySettings:
    def test_defaults(self) -> None:
        settings = Settings()
        assert settings.classification_priority_enabled is True
        assert settings.classification_priority_track_radius_m == pytest.approx(50.0)
        assert settings.classification_priority_buckets == 10

    def test_from_env_overrides(self, monkeypatch) -> None:
        monkeypatch.setenv("MINIMAPPR_CLASSIFICATION_PRIORITY_ENABLED", "false")
        monkeypatch.setenv("MINIMAPPR_CLASSIFICATION_PRIORITY_TRACK_RADIUS_M", "120")
        monkeypatch.setenv("MINIMAPPR_CLASSIFICATION_PRIORITY_BUCKETS", "4")
        monkeypatch.setenv("MINIMAPPR_CLASSIFICATION_PRIORITY_TRACK_WEIGHT", "0.9")

        settings = Settings.from_env()

        assert settings.classification_priority_enabled is False
        assert settings.classification_priority_track_radius_m == pytest.approx(120.0)
        assert settings.classification_priority_buckets == 4
        assert settings.classification_priority_track_weight == pytest.approx(0.9)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("classification_priority_track_radius_m", 0.0),
            ("classification_priority_track_cache_seconds", -1.0),
            ("classification_priority_buckets", 0),
            ("classification_priority_track_weight", -0.1),
        ],
    )
    def test_invalid_values_rejected(self, field: str, value: float) -> None:
        with pytest.raises(ValueError):
            Settings(**{field: value})
