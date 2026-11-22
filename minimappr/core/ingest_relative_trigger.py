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
O(1) per frame and no I/O: safe on the ingest hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from minimappr.config import RelativeTriggerConfig

DECIBELS_PER_AMPLITUDE_DECADE = 20.0


@dataclass(slots=True)
class _NodeFloorState:
    floor_db: float
    observed_seconds: float = 0.0


class NoiseFloorRelativeTrigger:
    def __init__(self, config: RelativeTriggerConfig) -> None:
        self._config = config
        self._floor_by_node: dict[str, _NodeFloorState] = {}

    def floor_rms(self, node_id: str) -> float | None:
        state = self._floor_by_node.get(node_id)
        return None if state is None else 10.0 ** (state.floor_db / DECIBELS_PER_AMPLITUDE_DECADE)

    def observe(self, node_id: str, frame_rms: float, frame_seconds: float) -> bool:
        """Update the node's floor with this frame; True when the frame stands
        ``snr_db`` above the floor as it was *before* the frame (so a call never
        raises its own bar)."""
        config = self._config
        level_db = DECIBELS_PER_AMPLITUDE_DECADE * math.log10(max(frame_rms, config.minimum_rms))
        state = self._floor_by_node.get(node_id)
        if state is None:
            self._floor_by_node[node_id] = _NodeFloorState(floor_db=level_db, observed_seconds=frame_seconds)
            return False
        triggered = (
            state.observed_seconds >= config.warmup_seconds
            and frame_rms > config.minimum_rms
            and level_db - state.floor_db >= config.snr_db
        )
        time_constant = config.fall_seconds if level_db < state.floor_db else config.rise_seconds
        smoothing = 1.0 - math.exp(-max(frame_seconds, 0.0) / max(time_constant, 1e-9))
        state.floor_db += smoothing * (level_db - state.floor_db)
        state.observed_seconds += frame_seconds
        return triggered
