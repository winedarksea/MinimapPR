"""Noise-floor-relative ingest trigger (Field Mode W3-attribution, flag
``trigger_noise_floor_relative_enabled``; default off = absolute gate only).

The absolute ``trigger_rms`` gate decides whether a faint distant bird ever
opens a localization window: a call 10 dB over a quiet site's floor can sit
far below any fixed RMS threshold that also keeps a noisy site calm (the W2
faint scene needed ``trigger_rms`` hand-set to ~3x its floor). Here each node
tracks its own floor and a frame also triggers when it stands
``snr_db`` above that floor. The absolute gate is kept (OR), so every window
that opens today still opens.

The floor follows the frame-energy envelope in dB with an asymmetric one-pole
smoother: it falls quickly (``fall_seconds``) into quiet gaps and rises slowly
(``rise_seconds``), so a call lasting a fraction of a second barely lifts it
while a lasting change of ambience is followed within tens of seconds.

Because the floor hugs the envelope's lower edge, fluctuating ambience (gusty
wind, rain) would otherwise put its own peaks ``snr_db`` over it and trigger
continuously (every cooldown). The bar is therefore raised by
``fluctuation_multiplier`` x the slow mean excess of non-triggering frames over
the floor: ~0 dB on steady noise (faint birds keep the bare ``snr_db`` bar),
several dB in gusts.

Short-block excess (``trigger_short_block_excess_enabled``, default off): a
call shorter than a frame, or straddling two, is diluted by the frame's quiet
remainder. When the caller also passes the frame's loudest short block
(``peak_short_block_rms``), that level is judged against the same floor with
its own fluctuation allowance (block maxima fluctuate more than frame means)
and OR'd with the frame decision, so every frame that triggers today still
does. The floor itself always follows the whole-frame level. Same semantics as
``minimappr-core/src/stage1/noise_floor_relative_trigger.rs``.

O(1) per frame, no I/O, no per-frame allocation beyond Python floats: safe on
the ingest hot path (``peak_short_block_rms`` is one O(N) numpy pass).
Non-finite levels are ignored so one corrupt frame cannot poison a node's
floor; per-node state is LRU-capped so node churn cannot grow it.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from minimappr.config import RelativeTriggerConfig

DECIBELS_PER_AMPLITUDE_DECADE = 20.0
# Least-recently-heard node floors are evicted beyond this (memory bound only;
# far above any real deployment's node count).
MAX_TRACKED_NODE_FLOORS = 1024
MINIMUM_TIME_CONSTANT_SECONDS = 1e-9


@dataclass(slots=True)
class _NodeFloorState:
    floor_db: float
    observed_seconds: float = 0.0
    # Slow mean of max(level - floor, 0) over non-triggering frames (fluctuation scale).
    mean_excess_db: float = 0.0
    # Same, for the loudest short block (only fed when the caller passes one).
    mean_short_block_excess_db: float = 0.0


def short_block_samples(sample_rate_hz: float, block_seconds: float) -> int:
    # floor(x + 0.5), not round(): Python rounds half to even, Rust half away from zero.
    return max(1, int(math.floor(block_seconds * sample_rate_hz + 0.5)))


def peak_short_block_rms(channels_first: np.ndarray, sample_rate_hz: float, block_seconds: float) -> float:
    """Largest RMS over any channel's non-overlapping ``block_seconds`` blocks.

    Blocks start at sample 0; a remainder is covered by one extra block ending
    at the last sample (overlapping its neighbour) so every block has full
    length and the same noise variance. A frame no longer than one block
    yields its strongest channel's whole-frame RMS.
    """
    if channels_first.ndim != 2 or channels_first.shape[0] == 0 or channels_first.shape[1] == 0:
        return 0.0
    squared = np.square(channels_first, dtype=np.float64)
    frame_samples = squared.shape[1]
    block_samples = short_block_samples(sample_rate_hz, block_seconds)
    if block_samples >= frame_samples:
        return float(math.sqrt(float(squared.mean(axis=1).max())))
    full_block_count = frame_samples // block_samples
    block_mean_squares = (
        squared[:, : full_block_count * block_samples].reshape(squared.shape[0], full_block_count, block_samples).mean(axis=2)
    )
    peak_mean_square = float(block_mean_squares.max())
    if frame_samples % block_samples:
        peak_mean_square = max(peak_mean_square, float(squared[:, frame_samples - block_samples :].mean(axis=1).max()))
    return math.sqrt(peak_mean_square)


class NoiseFloorRelativeTrigger:
    def __init__(self, config: RelativeTriggerConfig) -> None:
        self._config = config
        self._floor_by_node: OrderedDict[str, _NodeFloorState] = OrderedDict()

    def floor_rms(self, node_id: str) -> float | None:
        state = self._floor_by_node.get(node_id)
        return None if state is None else 10.0 ** (state.floor_db / DECIBELS_PER_AMPLITUDE_DECADE)

    def tracked_node_count(self) -> int:
        return len(self._floor_by_node)

    def observe(
        self,
        node_id: str,
        frame_rms: float,
        frame_seconds: float,
        peak_short_block_rms: float | None = None,
    ) -> bool:
        """Update the node's floor with this frame; True when the frame (or,
        when given, its loudest short block) stands ``snr_db`` (+ its
        fluctuation allowance) above the floor as it was *before* the frame
        (so a call never raises its own bar)."""
        if not math.isfinite(frame_rms) or not math.isfinite(frame_seconds):
            return False
        if peak_short_block_rms is not None and not math.isfinite(peak_short_block_rms):
            return False
        config = self._config
        level_db = DECIBELS_PER_AMPLITUDE_DECADE * math.log10(max(frame_rms, config.minimum_rms))
        state = self._floor_by_node.get(node_id)
        if state is None:
            self._floor_by_node[node_id] = _NodeFloorState(floor_db=level_db, observed_seconds=frame_seconds)
            if len(self._floor_by_node) > MAX_TRACKED_NODE_FLOORS:
                self._floor_by_node.popitem(last=False)
            return False
        self._floor_by_node.move_to_end(node_id)
        excess_db = level_db - state.floor_db
        warmed_up = state.observed_seconds >= config.warmup_seconds
        required_db = config.snr_db + config.fluctuation_multiplier * state.mean_excess_db
        triggered = warmed_up and frame_rms > config.minimum_rms and excess_db >= required_db
        short_block_excess_db: float | None = None
        if peak_short_block_rms is not None:
            short_block_level_db = DECIBELS_PER_AMPLITUDE_DECADE * math.log10(
                max(peak_short_block_rms, config.minimum_rms)
            )
            short_block_excess_db = short_block_level_db - state.floor_db
            short_block_required_db = config.snr_db + config.fluctuation_multiplier * state.mean_short_block_excess_db
            triggered = triggered or (
                warmed_up
                and peak_short_block_rms > config.minimum_rms
                and short_block_excess_db >= short_block_required_db
            )
        step_seconds = max(frame_seconds, 0.0)
        time_constant = config.fall_seconds if excess_db < 0.0 else config.rise_seconds
        smoothing = 1.0 - math.exp(-step_seconds / max(time_constant, MINIMUM_TIME_CONSTANT_SECONDS))
        state.floor_db += smoothing * excess_db
        if not triggered:
            excess_smoothing = 1.0 - math.exp(-step_seconds / max(config.rise_seconds, MINIMUM_TIME_CONSTANT_SECONDS))
            state.mean_excess_db += excess_smoothing * (max(excess_db, 0.0) - state.mean_excess_db)
            if short_block_excess_db is not None:
                state.mean_short_block_excess_db += excess_smoothing * (
                    max(short_block_excess_db, 0.0) - state.mean_short_block_excess_db
                )
        state.observed_seconds += frame_seconds
        return triggered
