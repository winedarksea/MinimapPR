"""Field Mode W3-attribution: noise-floor-relative ingest trigger (flag off by default)."""

from __future__ import annotations

import numpy as np

from minimappr.config import Settings
from minimappr.core.ingest_relative_trigger import NoiseFloorRelativeTrigger

FRAME_SECONDS = 1024 / 24_000
FLOOR_RMS = 5e-5


def _trigger(**overrides) -> NoiseFloorRelativeTrigger:
    return NoiseFloorRelativeTrigger(Settings(trigger_noise_floor_relative_enabled=True, **overrides).relative_trigger_config())


def _feed_noise(trigger, node="a", seconds=3.0, rms=FLOOR_RMS, seed=0) -> list[bool]:
    rng = np.random.default_rng(seed)
    frames = int(seconds / FRAME_SECONDS)
    return [trigger.observe(node, rms * float(rng.uniform(0.85, 1.15)), FRAME_SECONDS) for _ in range(frames)]


def test_faint_call_over_a_quiet_floor_triggers_far_below_any_absolute_gate() -> None:
    trigger = _trigger()
    assert not any(_feed_noise(trigger))  # noise alone never triggers
    faint_call_rms = FLOOR_RMS * 10 ** (12 / 20)  # 12 dB over the floor, ~2e-4 << trigger_rms 0.001
    assert trigger.observe("a", faint_call_rms, FRAME_SECONDS)
    assert faint_call_rms < Settings().trigger_rms


def test_call_does_not_raise_its_own_bar_but_lasting_ambience_is_followed() -> None:
    trigger = _trigger()
    _feed_noise(trigger)
    call = FLOOR_RMS * 10 ** (12 / 20)
    fired = [trigger.observe("a", call, FRAME_SECONDS) for _ in range(int(0.6 / FRAME_SECONDS))]
    assert all(fired)  # a 0.6 s call stays above the slowly rising floor throughout
    _feed_noise(trigger, seconds=60.0, rms=FLOOR_RMS * 10)  # wind picks up and stays
    assert not trigger.observe("a", FLOOR_RMS * 10 * 1.2, FRAME_SECONDS)


def test_floors_are_per_node_and_warm_up_first() -> None:
    trigger = _trigger(trigger_noise_floor_warmup_seconds=1.0)
    assert not trigger.observe("a", FLOOR_RMS, FRAME_SECONDS)
    assert not trigger.observe("a", FLOOR_RMS * 100, FRAME_SECONDS)  # still warming up
    _feed_noise(trigger, node="a")
    _feed_noise(trigger, node="b", rms=FLOOR_RMS * 20)
    level = FLOOR_RMS * 10 ** (12 / 20)
    assert trigger.observe("a", level, FRAME_SECONDS)
    assert not trigger.observe("b", level, FRAME_SECONDS)  # loud node's floor is above it


def test_digital_silence_does_not_make_bit_noise_loud() -> None:
    trigger = _trigger()
    _feed_noise(trigger, rms=0.0)
    assert not trigger.observe("a", 5e-6, FRAME_SECONDS)


def test_relative_trigger_defaults_off() -> None:
    assert Settings().trigger_noise_floor_relative_enabled is False


def _feed_gusty_noise(trigger, *, envelope_std_db: float, seconds: float = 300.0, call_every_frames: int = 0):
    """Log-normal envelope with ~0.3 s correlation (gusty wind / rain); optional 12 dB calls."""
    rng = np.random.default_rng(1)
    correlation = np.exp(-FRAME_SECONDS / 0.3)
    state = 0.0
    noise_fires = call_hits = calls = 0
    frames = int(seconds / FRAME_SECONDS)
    for index in range(frames):
        state = correlation * state + np.sqrt(1.0 - correlation**2) * rng.standard_normal()
        level = FLOOR_RMS * 10 ** (envelope_std_db * state / 20)
        is_call = call_every_frames and index > 100 and index % call_every_frames < 10
        if is_call:
            level = float(np.hypot(level, FLOOR_RMS * 10 ** (12 / 20)))
        fired = trigger.observe("a", level, FRAME_SECONDS)
        calls += bool(is_call)
        call_hits += bool(is_call) and fired
        noise_fires += (not is_call) and fired
    return noise_fires / frames, (call_hits / calls if calls else None)


def test_gusty_ambience_does_not_trigger_every_cooldown_but_calls_still_do() -> None:
    bare = _trigger(trigger_noise_floor_fluctuation_multiplier=0.0)
    bare_rate, _ = _feed_gusty_noise(bare, envelope_std_db=3.0)
    adaptive = _trigger()
    adaptive_rate, call_hit_rate = _feed_gusty_noise(adaptive, envelope_std_db=3.0, call_every_frames=200)
    assert bare_rate > 0.03  # floor hugs the envelope minima: gust peaks clear a bare 8 dB bar
    assert adaptive_rate < 0.005
    assert call_hit_rate is not None and call_hit_rate > 0.9


def test_steady_noise_keeps_the_bare_bar_for_faint_birds() -> None:
    trigger = _trigger()
    _feed_noise(trigger, seconds=30.0)
    assert trigger.observe("a", FLOOR_RMS * 10 ** (8.5 / 20), FRAME_SECONDS)


def test_non_finite_frame_does_not_poison_the_floor() -> None:
    trigger = _trigger()
    _feed_noise(trigger)
    floor_before = trigger.floor_rms("a")
    assert not trigger.observe("a", float("nan"), FRAME_SECONDS)
    assert not trigger.observe("a", float("inf"), FRAME_SECONDS)
    assert trigger.floor_rms("a") == floor_before
    assert trigger.observe("a", FLOOR_RMS * 10 ** (12 / 20), FRAME_SECONDS)


def test_node_churn_cannot_grow_floor_state_without_bound(monkeypatch) -> None:
    import minimappr.core.ingest_relative_trigger as relative_trigger_module

    monkeypatch.setattr(relative_trigger_module, "MAX_TRACKED_NODE_FLOORS", 8)
    trigger = _trigger()
    for index in range(50):
        trigger.observe(f"phone-{index}", FLOOR_RMS, FRAME_SECONDS)
        trigger.observe("fixed", FLOOR_RMS, FRAME_SECONDS)  # recently heard: never evicted
    assert trigger.tracked_node_count() == 8
    assert trigger.floor_rms("fixed") is not None
    assert trigger.floor_rms("phone-0") is None
