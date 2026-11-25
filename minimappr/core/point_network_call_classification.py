"""Point-network clip classification (plan item P7).

A point network is a solve whose every contributing node gave exactly one
sensor (single-mic or omni point nodes): no node can beamform a clean clip.
Before P7 both paths classified a mix of all of them anyway: the Rust render's
band-split DAS (`birdnet_render.rs`) and Python's ``BandSplitDasRenderer`` only
steer below the geometry's alias cutoff, c / (2 x baseline), which is a few Hz
for nodes metres apart, so the clip was the *unaligned* mean of every node:
the far nodes' noise diluted the call, and each node's copy landed at its own
propagation delay, smearing one call into several.

Here the clip is one point node's own audio, re-timed to that node:

- each contributing sensor's clip is the clip the path would have classified
  (render span, or the Python classification window) shifted by the solve's
  arrival-time difference between that sensor and the alignment reference
  (the node the solve window is centred on), from the solved position;
- a node where another localized caller overlaps the clip at a comparable
  level is masked (``point_network_overlapping_callers``); another solve of
  the same call is not another caller (``point_network_same_caller``);
- the clearest unmasked node wins by in-band SNR (event-span level over the
  clip's noise floor, both over the frames the node received; a node that
  received less than the configured share of the clip is not a candidate),
  ties going to the node nearest the source;
- provenance records the node, why, the alignment offset and the runner-up;
- if every node is masked, the least-masked node's clip is classified but the
  species is left uncertain: label ``unknown`` with confidence 0 and the
  unattributed scores kept (the vocabulary of
  ``onset_cluster_attribution_stage``), never a guess the clip may not hold;
- if the clip classifier fails (timeout, error) and the path already holds a
  label of its own (the sidecar's), that label stands: a backend failure is
  not an abstention.

The position was already observed (P5); this only decides the label. Audio
comes from the in-memory per-sensor buffer (mirrored from the sidecar's
``raw_audio_frame`` events on the Rust path), so nothing reaches disk.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from minimappr.classifiers.base import ClassificationResult
from minimappr.core.onset_cluster_attribution_stage import FAILED_CLASSIFICATION_REASONS
from minimappr.core.point_network_clip_config import PointNetworkClipConfig, point_network_clip_config_from_settings
from minimappr.core.point_network_clip_signal import FramedInBandEnergy, framed_in_band_energy
from minimappr.core.point_network_localized_calls import (
    NANOSECONDS_PER_SECOND,
    LocalizedCall,
    LocalizedCallRegistry,
    seconds_to_nanoseconds,
)
from minimappr.core.point_network_overlapping_callers import (
    CallerReferenceLevel,
    CompetingCaller,
    competing_callers_at_node,
    nearest_position_id,
)
from minimappr.core.point_network_same_caller import is_same_caller

if TYPE_CHECKING:
    from minimappr.core.classification import ClassifiedResult
    from minimappr.core.fusion_node import FusionNode

POINT_NETWORK_CLIP_FEATURE_KEY = "point_network_clip"
POINT_NETWORK_CLIP_CLASSIFICATION_PATH = "point_network_clip"
SELECTION_REASON_HIGHEST_IN_BAND_SNR = "highest_in_band_snr"
SELECTION_REASON_LEAST_MASKED = "every_node_masked_least_masked_chosen"
UNAVAILABLE_REASON_NO_BUFFERED_CLIP = "no_point_node_clip_buffered"
UNAVAILABLE_REASON_NO_COVERED_CLIP = "no_point_node_clip_sufficiently_received"
UNAVAILABLE_REASON_CLIP_CLASSIFIER_FAILED = "clip_classifier_failed"
UNCERTAINTY_REASON_OVERLAPPING_CALLER = "overlapping_localized_caller"
# The abstention label (``taxonomy.ABSTENTION_LABELS``) the onset attributor
# also uses when it cannot settle which caller owns a window.
SPECIES_UNCERTAIN_LABEL = "unknown"
FEATURE_DECIMALS = 4

DefaultClassification = Callable[[], Awaitable["ClassifiedResult | None"]]


@dataclass(frozen=True, slots=True)
class PointNetworkCallRequest:
    """One point-network solve's call, ready for clip selection."""

    call: LocalizedCall
    # Contributing sensors only (solve order), each with its node and site position.
    sensor_node_ids: dict[str, str]
    sensor_positions_m: dict[str, np.ndarray]
    # The clip the path would otherwise classify, on the alignment reference's timeline.
    base_clip_start_ns: int
    clip_seconds: float
    sample_rate_hz: int
    event_time_ns: int


def is_point_network(sensor_node_ids: dict[str, str]) -> bool:
    """At least two nodes, each contributing exactly one sensor."""
    node_ids = list(sensor_node_ids.values())
    return len(node_ids) >= 2 and len(set(node_ids)) == len(node_ids)


@dataclass(slots=True)
class _PointNodeClip:
    sensor_id: str
    node_id: str
    distance_to_source_m: float
    alignment_offset_seconds: float
    clip_start_ns: int
    samples: np.ndarray | None = None
    coverage: np.ndarray | None = None
    energy: FramedInBandEnergy | None = None
    signal_level_db: float | None = None
    noise_floor_db: float | None = None
    competing_callers: list[CompetingCaller] = field(default_factory=list)

    @property
    def snr_db(self) -> float | None:
        if self.signal_level_db is None or self.noise_floor_db is None:
            return None
        return self.signal_level_db - self.noise_floor_db

    @property
    def worst_competitor_margin_db(self) -> float:
        return max((caller.level_relative_to_call_db for caller in self.competing_callers), default=float("-inf"))

    def ranking_key(self) -> tuple:
        # Unmasked nodes first, by SNR; among masked ones, where the call dominates most.
        masked = bool(self.competing_callers)
        return (masked, self.worst_competitor_margin_db if masked else 0.0, -self.snr_db, self.distance_to_source_m)


@dataclass(slots=True)
class PointNetworkClipSelection:
    request: PointNetworkCallRequest
    chosen: _PointNodeClip
    runner_up: _PointNodeClip | None
    candidates: list[_PointNodeClip]

    @property
    def species_uncertain(self) -> bool:
        return bool(self.chosen.competing_callers)

    def provenance(self) -> dict[str, Any]:
        chosen, runner_up = self.chosen, self.runner_up
        return {
            "node_id": chosen.node_id,
            "sensor_id": chosen.sensor_id,
            "selection_reason": SELECTION_REASON_LEAST_MASKED if self.species_uncertain else SELECTION_REASON_HIGHEST_IN_BAND_SNR,
            "in_band_snr_db": _rounded(chosen.snr_db),
            "runner_up_node_id": None if runner_up is None else runner_up.node_id,
            "runner_up_in_band_snr_db": None if runner_up is None else _rounded(runner_up.snr_db),
            "alignment_reference_sensor_id": self.request.call.alignment_reference_sensor_id,
            "alignment_offset_seconds": float(chosen.alignment_offset_seconds),
            "clip_start_time_ns": int(chosen.clip_start_ns),
            "clip_seconds": float(self.request.clip_seconds),
            "sample_rate_hz": int(self.request.sample_rate_hz),
            "in_band_snr_db_by_node": {
                candidate.node_id: _rounded(candidate.snr_db) for candidate in self.candidates if candidate.snr_db is not None
            },
            "unbuffered_node_ids": sorted(c.node_id for c in self.candidates if c.samples is None),
            "covered_frame_ratio_by_node": {
                c.node_id: _rounded(c.energy.covered_frame_ratio) for c in self.candidates if c.energy is not None
            },
            "masked_node_ids": sorted(c.node_id for c in self.candidates if c.competing_callers),
            "species_uncertain": self.species_uncertain,
            "overlapping_callers": [
                {
                    "call_id": caller.call_id,
                    "separation_m": round(caller.separation_m, FEATURE_DECIMALS),
                    "level_relative_to_call_db": round(caller.level_relative_to_call_db, FEATURE_DECIMALS),
                }
                for caller in self.chosen.competing_callers
            ],
        }


class PointNetworkCallClassifier:
    """Classifies point-network solves from the clearest node's aligned clip.

    ``host`` is read per call (settings, buffer, orchestrator, metrics), so a
    runtime settings change or a replaced orchestrator applies immediately.
    """

    def __init__(self, host: "FusionNode") -> None:
        self._host = host
        config = self.config
        self._calls = LocalizedCallRegistry(
            retention_seconds=config.localized_call_retention_seconds, max_records=config.localized_call_max_records
        )

    @property
    def config(self) -> PointNetworkClipConfig:
        return point_network_clip_config_from_settings(self._host.settings)

    def register_localized_call(self, call: LocalizedCall | None) -> None:
        """Every localized solve (array or point) is remembered: any caller can
        overlap a later point-network clip."""
        if call is not None:
            self._calls.register(call)

    async def classify(
        self,
        request: PointNetworkCallRequest | None,
        *,
        default_classification: DefaultClassification,
        keep_default_on_classifier_failure: bool = False,
    ) -> "ClassifiedResult | None":
        """Classify ``request``'s clip, or run ``default_classification`` when the
        solve is not a point network (``request is None``) or P7 is off.

        ``keep_default_on_classifier_failure``: the default is a label the path
        already holds (the sidecar's), so a failed clip classification returns
        it instead of an abstention. Off where the default is another inference
        on the same backend, which would only fail again.
        """
        config = self.config
        if request is None or not config.enabled:
            return await default_classification()
        self.register_localized_call(request.call)
        selection, unavailable_reason = await self._select_clip(request, config)
        if selection is None:
            return await self._default_with_clip_unavailable(request, default_classification, unavailable_reason)
        return await self._classify_selection(
            selection,
            default_classification=default_classification if keep_default_on_classifier_failure else None,
        )

    async def _default_with_clip_unavailable(
        self,
        request: PointNetworkCallRequest,
        default_classification: DefaultClassification,
        unavailable_reason: str,
        **provenance: Any,
    ) -> "ClassifiedResult | None":
        self._host._metrics.point_network_clip_unavailable_count += 1
        classified = await default_classification()
        if classified is not None:
            classified.classification.features[POINT_NETWORK_CLIP_FEATURE_KEY] = {
                "unavailable_reason": unavailable_reason,
                "candidate_node_ids": sorted(set(request.sensor_node_ids.values())),
                **provenance,
            }
        return classified

    async def _select_clip(
        self, request: PointNetworkCallRequest, config: PointNetworkClipConfig
    ) -> tuple[PointNetworkClipSelection | None, str]:
        """The selection, or None with the reason no node's clip could be used."""
        clip_sample_count = max(1, int(round(request.clip_seconds * request.sample_rate_hz)))
        clip_duration_ns = int(round(clip_sample_count * NANOSECONDS_PER_SECOND / request.sample_rate_hz))
        candidates = [self._aligned_candidate(request, sensor_id) for sensor_id in request.sensor_node_ids]
        buffer = self._host.buffer
        for candidate in candidates:
            # Ending-at keeps the clip's first sample at clip_start_ns exactly;
            # a clip the buffer only partly holds is not this node's clip.
            windows = await buffer.get_synchronized_window_ending_at_with_coverage(
                sensor_ids=[candidate.sensor_id],
                end_time_ns=candidate.clip_start_ns + clip_duration_ns,
                window_seconds=request.clip_seconds,
                sample_rate_hz=request.sample_rate_hz,
            )
            window = windows.get(candidate.sensor_id)
            if window is not None and window[0].size == clip_sample_count:
                candidate.samples, candidate.coverage = window
        buffered = [candidate for candidate in candidates if candidate.samples is not None]
        if not buffered:
            return None, UNAVAILABLE_REASON_NO_BUFFERED_CLIP
        await asyncio.to_thread(self._measure_levels, buffered, request, config)
        measured = [candidate for candidate in buffered if candidate.snr_db is not None]
        if not measured:
            return None, UNAVAILABLE_REASON_NO_COVERED_CLIP
        await self._mark_competing_callers(measured, request, config, clip_duration_ns)
        ranked = sorted(measured, key=_PointNodeClip.ranking_key)
        selection = PointNetworkClipSelection(
            request=request,
            chosen=ranked[0],
            runner_up=ranked[1] if len(ranked) > 1 else None,
            candidates=candidates,
        )
        return selection, ""

    @staticmethod
    def _aligned_candidate(request: PointNetworkCallRequest, sensor_id: str) -> _PointNodeClip:
        position_m = request.sensor_positions_m[sensor_id]
        alignment_offset_seconds = request.call.arrival_offset_seconds_at(position_m)
        return _PointNodeClip(
            sensor_id=sensor_id,
            node_id=request.sensor_node_ids[sensor_id],
            distance_to_source_m=float(np.linalg.norm(request.call.source_position_m - position_m)),
            alignment_offset_seconds=alignment_offset_seconds,
            clip_start_ns=request.base_clip_start_ns + seconds_to_nanoseconds(alignment_offset_seconds),
        )

    @staticmethod
    def _measure_levels(
        candidates: list[_PointNodeClip], request: PointNetworkCallRequest, config: PointNetworkClipConfig
    ) -> None:
        for candidate in candidates:
            candidate.energy = framed_in_band_energy(candidate.samples, candidate.coverage, request.sample_rate_hz, config)
            if candidate.energy.covered_frame_ratio < config.min_clip_covered_frame_ratio:
                continue  # Mostly gap: its level and floor rest on too few frames.
            span_start_ns, span_end_ns = request.call.event_span_at(request.sensor_positions_m[candidate.sensor_id])
            start_sample, end_sample = _span_samples_in_clip(
                candidate.clip_start_ns, span_start_ns, span_end_ns, request.sample_rate_hz, candidate.samples.size
            )
            candidate.signal_level_db = candidate.energy.level_db(
                start_sample=start_sample, end_sample=end_sample, quantile=config.signal_energy_quantile
            )
            candidate.noise_floor_db = candidate.energy.noise_floor_db(config.noise_floor_energy_quantile)

    async def _mark_competing_callers(
        self,
        measured: list[_PointNodeClip],
        request: PointNetworkCallRequest,
        config: PointNetworkClipConfig,
        clip_duration_ns: int,
    ) -> None:
        nearest = min(measured, key=lambda candidate: candidate.distance_to_source_m)
        call_level = CallerReferenceLevel(
            call=request.call,
            reference_position_m=request.sensor_positions_m[nearest.sensor_id],
            level_db=nearest.signal_level_db,
        )
        clip_spans_ns = {
            candidate.sensor_id: (candidate.clip_start_ns, candidate.clip_start_ns + clip_duration_ns)
            for candidate in measured
        }
        other_levels = await self._other_caller_levels(request, config, clip_spans_ns)
        if not other_levels:
            return
        for candidate in measured:
            candidate.competing_callers = competing_callers_at_node(
                call_level=call_level,
                other_levels=other_levels,
                node_position_m=request.sensor_positions_m[candidate.sensor_id],
                node_noise_floor_db=candidate.noise_floor_db,
                clip_span_ns=clip_spans_ns[candidate.sensor_id],
                config=config,
            )

    async def _other_caller_levels(
        self,
        request: PointNetworkCallRequest,
        config: PointNetworkClipConfig,
        clip_spans_ns: dict[str, tuple[int, int]],
    ) -> list[CallerReferenceLevel]:
        """Each distinct other caller overlapping a candidate clip, with its level
        at its nearest node of this network."""
        levels: list[CallerReferenceLevel] = []
        for other in self._calls.calls_other_than(request.call.call_id):
            if is_same_caller(request.call, other, config):
                continue  # Another solve of this call, or the next note of one perched bird.
            if not _overlaps_any_clip(other, request, clip_spans_ns):
                continue
            sensor_id = nearest_position_id(other.source_position_m, request.sensor_positions_m)
            position_m = request.sensor_positions_m[sensor_id]
            span_start_ns, span_end_ns = other.event_span_at(position_m)
            windows = await self._host.buffer.get_synchronized_window_ending_at_with_coverage(
                sensor_ids=[sensor_id],
                end_time_ns=span_end_ns,
                window_seconds=(span_end_ns - span_start_ns) / NANOSECONDS_PER_SECOND,
                sample_rate_hz=request.sample_rate_hz,
            )
            window = windows.get(sensor_id)
            if window is None or window[0].size == 0:
                continue  # Not buffered here: its level cannot be measured.
            samples, coverage = window
            energy = await asyncio.to_thread(framed_in_band_energy, samples, coverage, request.sample_rate_hz, config)
            level_db = energy.level_db(start_sample=0, end_sample=samples.size, quantile=config.signal_energy_quantile)
            if level_db is not None:
                levels.append(CallerReferenceLevel(call=other, reference_position_m=position_m, level_db=level_db))
        return levels

    async def _classify_selection(
        self, selection: PointNetworkClipSelection, *, default_classification: DefaultClassification | None
    ) -> "ClassifiedResult | None":
        request = selection.request
        orchestrator = self._host._classification_orchestrator
        classified = await orchestrator.classify_omni_only(
            reference_signal=selection.chosen.samples.astype(np.float32, copy=False),
            sample_rate_hz=request.sample_rate_hz,
            event_time_ns=request.event_time_ns,
            path=POINT_NETWORK_CLIP_CLASSIFICATION_PATH,
        )
        # Timeouts too, not only errors (``_classification_backend_failed`` sees errors only).
        failure_reason = classified.classification.features.get("reason")
        if default_classification is not None and failure_reason in FAILED_CLASSIFICATION_REASONS:
            return await self._default_with_clip_unavailable(
                request,
                default_classification,
                UNAVAILABLE_REASON_CLIP_CLASSIFIER_FAILED,
                clip_node_id=selection.chosen.node_id,
                failure_reason=failure_reason,
            )
        self._host._metrics.point_network_clip_classified_count += 1
        provenance = selection.provenance()
        if not selection.species_uncertain:
            classified.classification.features[POINT_NETWORK_CLIP_FEATURE_KEY] = provenance
            return classified
        self._host._metrics.point_network_clip_species_uncertain_count += 1
        original = classified.classification
        provenance.update(
            uncertainty_reason=UNCERTAINTY_REASON_OVERLAPPING_CALLER,
            unattributed_label=original.label,
            unattributed_confidence=round(float(original.confidence), FEATURE_DECIMALS),
            unattributed_scores={label: round(float(score), FEATURE_DECIMALS) for label, score in original.scores.items()},
        )
        features = dict(original.features)
        features[POINT_NETWORK_CLIP_FEATURE_KEY] = provenance
        uncertain = ClassificationResult(label=SPECIES_UNCERTAIN_LABEL, confidence=0.0, scores={}, features=features)
        return await orchestrator.relabel_classified_result(classified, uncertain, event_time_ns=request.event_time_ns)


def _overlaps_any_clip(
    other: LocalizedCall, request: PointNetworkCallRequest, clip_spans_ns: dict[str, tuple[int, int]]
) -> bool:
    for sensor_id, (clip_start_ns, clip_end_ns) in clip_spans_ns.items():
        span_start_ns, span_end_ns = other.event_span_at(request.sensor_positions_m[sensor_id])
        if span_end_ns > clip_start_ns and span_start_ns < clip_end_ns:
            return True
    return False


def _span_samples_in_clip(
    clip_start_ns: int, span_start_ns: int, span_end_ns: int, sample_rate_hz: int, clip_samples: int
) -> tuple[int, int]:
    """``[start, end)`` sample indices of a timeline span, clamped to the clip."""
    start = int(round((span_start_ns - clip_start_ns) * sample_rate_hz / NANOSECONDS_PER_SECOND))
    end = int(round((span_end_ns - clip_start_ns) * sample_rate_hz / NANOSECONDS_PER_SECOND))
    return max(0, start), min(clip_samples, end)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), FEATURE_DECIMALS)


__all__ = [
    "POINT_NETWORK_CLIP_CLASSIFICATION_PATH",
    "POINT_NETWORK_CLIP_FEATURE_KEY",
    "SELECTION_REASON_HIGHEST_IN_BAND_SNR",
    "SPECIES_UNCERTAIN_LABEL",
    "UNAVAILABLE_REASON_CLIP_CLASSIFIER_FAILED",
    "UNAVAILABLE_REASON_NO_BUFFERED_CLIP",
    "UNAVAILABLE_REASON_NO_COVERED_CLIP",
    "UNCERTAINTY_REASON_OVERLAPPING_CALLER",
    "PointNetworkCallClassifier",
    "PointNetworkCallRequest",
    "PointNetworkClipSelection",
    "is_point_network",
]
