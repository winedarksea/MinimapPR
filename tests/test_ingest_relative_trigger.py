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
