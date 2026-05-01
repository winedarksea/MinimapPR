"""Internal Rust DSP manifest delivery types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from minimappr.api.binary_ingest import parse_binary_ingest_payload
from minimappr.api.journal_reader import JournalPayloadHandle
from minimappr.models import (
    ClassificationResult,
    EnvironmentSampleIn,
    NodeSpec,
    StoreForwardIngestRequest,
    TimeQuality,
)


@dataclass(slots=True)
class LocalizedClassifierRenderRequest:
    manifest_id: str
    node: NodeSpec
    event_time_ns: int
    sample_rate_hz: int
    decoded_audio: np.ndarray
    localization_position_m: tuple[float, float, float]
    localization_confidence: float
    localization_gdop: float = float("inf")
    localization_method: str = "rust_dsp_worker"
    source_type: str = "raw_sensor"
    time_quality: TimeQuality = TimeQuality.FREE_RUNNING
    source_observation_ids: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    render_kind: str | None = None
    fallback_reason: str | None = None
    authoritative_classification: ClassificationResult | None = None


@dataclass(frozen=True, slots=True)
class JournalCursorUpdate:
    journal_id: str
    stream_key: str
    journal_epoch: int
    journal_sequence: int


@dataclass(frozen=True, slots=True)
class LocalizedRenderManifestBundle:
    request: LocalizedClassifierRenderRequest
    cursor_updates: tuple[JournalCursorUpdate, ...]


def manifest_source_key(manifest_payload: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    source_handles = manifest_payload.get("source_handles")
    if not isinstance(source_handles, list):
        return ()
    key_parts: list[tuple[Any, ...]] = []
    for handle_payload in source_handles:
        if not isinstance(handle_payload, dict):
            return ()
        key_parts.append(
            (
                int(handle_payload.get("journal_epoch") or 0),
                str(handle_payload.get("stream_key") or ""),
                str(handle_payload.get("segment_id") or ""),
                int(handle_payload.get("payload_offset_bytes") or handle_payload.get("body_offset_bytes") or 0),
                int(handle_payload.get("payload_length_bytes") or handle_payload.get("body_length_bytes") or 0),
            )
        )
    return tuple(key_parts)


def load_localized_render_manifest_bundle(
    *,
    manifest_payload: dict[str, Any],
    journal_streams_dir: Path,
    paired_classifier_manifest_payload: dict[str, Any] | None = None,
) -> LocalizedRenderManifestBundle:
    source_handles_payload = manifest_payload.get("source_handles")
    if not isinstance(source_handles_payload, list) or not source_handles_payload:
        raise ValueError("Rust DSP manifest omitted source_handles")

    derived_handle_payload = manifest_payload.get("derived_handle")
    if derived_handle_payload is None and paired_classifier_manifest_payload is not None:
        derived_handle_payload = paired_classifier_manifest_payload.get("derived_handle")
    if not isinstance(derived_handle_payload, dict):
        raise ValueError("Rust DSP manifest omitted derived_handle")

    classifier_render_payload = manifest_payload.get("classifier_render")
    if classifier_render_payload is None and paired_classifier_manifest_payload is not None:
        classifier_render_payload = paired_classifier_manifest_payload.get("classifier_render")
    if not isinstance(classifier_render_payload, dict):
        raise ValueError("Rust DSP manifest omitted classifier_render payload")

    birdnet_payload = manifest_payload.get("birdnet")
    if birdnet_payload is None and paired_classifier_manifest_payload is not None:
        birdnet_payload = paired_classifier_manifest_payload.get("birdnet")
    birdnet_payload = birdnet_payload if isinstance(birdnet_payload, dict) else {}
    authoritative_classification = _load_authoritative_classification(birdnet_payload)
    manifest_id = str(manifest_payload.get("manifest_id") or "")
    if not manifest_id and paired_classifier_manifest_payload is not None:
        manifest_id = str(paired_classifier_manifest_payload.get("manifest_id") or "")

    source_handle = JournalPayloadHandle.from_mapping(source_handles_payload[0], journal_streams_dir=journal_streams_dir)
    node, event_time_ns, source_type, time_quality, environment = _load_source_context(source_handle)
    derived_handle = JournalPayloadHandle.from_mapping(derived_handle_payload, journal_streams_dir=journal_streams_dir)
    decoded_audio = _decode_classifier_render_audio(derived_handle, classifier_render_payload)

    localization_payload = manifest_payload.get("localization")
    if isinstance(localization_payload, dict) and isinstance(localization_payload.get("position_m"), list):
        localization_position_m = tuple(float(value) for value in localization_payload["position_m"])
        localization_confidence = float(localization_payload.get("confidence") or 0.0)
        localization_method = str(
            localization_payload.get("resolved_algorithm")
            or localization_payload.get("attempted_algorithm")
            or "rust_dsp_worker"
        )
    else:
        if node.position_m is None:
            raise ValueError("Standalone classifier render manifest requires node.position_m")
        localization_position_m = tuple(float(value) for value in node.position_m)
        localization_confidence = 0.0
        localization_method = "rust_classifier_render_fallback"

    cursor_updates = _resolve_cursor_updates(source_handles_payload, journal_streams_dir)
    return LocalizedRenderManifestBundle(
        request=LocalizedClassifierRenderRequest(
            manifest_id=manifest_id,
            node=node,
            event_time_ns=event_time_ns,
            sample_rate_hz=int(classifier_render_payload.get("sample_rate_hz") or 0),
            decoded_audio=decoded_audio,
            localization_position_m=localization_position_m,
            localization_confidence=localization_confidence,
            localization_method=localization_method,
            source_type=source_type,
            time_quality=time_quality,
            environment=environment,
            render_kind=str(
                classifier_render_payload.get("render_kind")
                or birdnet_payload.get("spatial_blend_mode")
                or ""
            )
            or None,
            fallback_reason=str(
                birdnet_payload.get("fallback_reason")
                or classifier_render_payload.get("fallback_reason")
                or ""
            )
            or None,
            authoritative_classification=authoritative_classification,
        ),
        cursor_updates=cursor_updates,
    )


def _load_authoritative_classification(
    birdnet_payload: dict[str, Any],
) -> ClassificationResult | None:
    label = str(birdnet_payload.get("label") or "").strip()
    if not label:
        return None

    confidence = birdnet_payload.get("label_confidence")
    if confidence is None:
        confidence = birdnet_payload.get("confidence")
    try:
        parsed_confidence = float(confidence)
    except (TypeError, ValueError):
        parsed_confidence = 0.0
    parsed_confidence = min(1.0, max(0.0, parsed_confidence))

    parsed_scores: dict[str, float] = {}
    scores_payload = birdnet_payload.get("scores")
    if isinstance(scores_payload, dict):
        for raw_label, raw_score in scores_payload.items():
            try:
                parsed_scores[str(raw_label)] = float(raw_score)
            except (TypeError, ValueError):
                continue

    features: dict[str, Any] = {}
    classifier_source_node = birdnet_payload.get("classifier_source_node")
    if classifier_source_node is not None:
        features["rust_classifier_source_node"] = str(classifier_source_node)

    return ClassificationResult(
        label=label,
        confidence=parsed_confidence,
        scores=parsed_scores,
        features=features,
    )


def _load_source_context(
    source_handle: JournalPayloadHandle,
) -> tuple[NodeSpec, int, str, TimeQuality, dict[str, Any]]:
    raw_payload = source_handle.read_bytes()
    if raw_payload.startswith(b"MMB1"):
        parsed_payload = parse_binary_ingest_payload(raw_payload)
        buffered_frame = parsed_payload.buffered_frames[0]
        return (
            parsed_payload.node,
            int(buffered_frame.frame.toa_ns or buffered_frame.frame.start_time_ns),
            buffered_frame.frame.source_type,
            buffered_frame.frame.time_quality,
            _environment_summary(buffered_frame.environment),
        )

    try:
        decoded_payload = json.loads(raw_payload.decode("utf-8"))
        parsed_payload = StoreForwardIngestRequest.model_validate(decoded_payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Unsupported Rust DSP source payload format: {exc}") from exc

    buffered_frame = parsed_payload.buffered_frames[0]
    return (
        parsed_payload.node,
        int(buffered_frame.frame.toa_ns or buffered_frame.frame.start_time_ns),
        buffered_frame.frame.source_type,
        buffered_frame.frame.time_quality,
        _environment_summary(buffered_frame.environment),
    )


def _decode_classifier_render_audio(
    derived_handle: JournalPayloadHandle,
    classifier_render_payload: dict[str, Any],
) -> np.ndarray:
    sample_format = str(classifier_render_payload.get("sample_format") or "")
    if sample_format != "pcm16le":
        raise ValueError(f"Unsupported classifier render sample format {sample_format!r}")

    channels = int(classifier_render_payload.get("channels") or 1)
    if channels < 1:
        raise ValueError("Classifier render channels must be >= 1")

    raw_payload = derived_handle.read_bytes()
    decoded = np.frombuffer(raw_payload, dtype="<i2").astype(np.float32) / 32768.0
    if decoded.size % channels != 0:
        raise ValueError("Classifier render payload length is not aligned to the channel count")
    if channels == 1:
        return decoded
    return decoded.reshape((-1, channels)).mean(axis=1)


def _resolve_cursor_updates(
    source_handles_payload: list[dict[str, Any]],
    journal_streams_dir: Path,
) -> tuple[JournalCursorUpdate, ...]:
    latest_by_stream: dict[str, JournalCursorUpdate] = {}
    for handle_payload in source_handles_payload:
        handle = JournalPayloadHandle.from_mapping(handle_payload, journal_streams_dir=journal_streams_dir)
        cursor_update = _lookup_cursor_update(handle, journal_streams_dir)
        existing = latest_by_stream.get(cursor_update.stream_key)
        if existing is None or _cursor_update_is_newer(existing, cursor_update):
            latest_by_stream[cursor_update.stream_key] = cursor_update
    return tuple(latest_by_stream.values())


def _lookup_cursor_update(
    handle: JournalPayloadHandle,
    journal_streams_dir: Path,
) -> JournalCursorUpdate:
    index_path = journal_streams_dir / handle.stream_key / "segments" / f"{handle.segment_id}.index.jsonl"
    with index_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            payload = json.loads(line)
            payload_offset = int(payload.get("payload_offset_bytes") or payload.get("body_offset_bytes") or 0)
            payload_length = int(payload.get("payload_length_bytes") or payload.get("body_length_bytes") or 0)
            if int(payload.get("journal_epoch") or 0) != handle.journal_epoch:
                continue
            if payload_offset != handle.payload_offset_bytes:
                continue
            if payload_length != handle.payload_length_bytes:
                continue
            return JournalCursorUpdate(
                journal_id=str(payload["journal_id"]),
                stream_key=str(payload.get("stream_key") or handle.stream_key),
                journal_epoch=int(payload.get("journal_epoch") or 0),
                journal_sequence=int(payload["journal_sequence"]),
            )
    raise ValueError(
        f"Unable to resolve journal cursor update for {handle.stream_key}/{handle.segment_id}@{handle.payload_offset_bytes}"
    )


def _cursor_update_is_newer(existing: JournalCursorUpdate, candidate: JournalCursorUpdate) -> bool:
    if candidate.journal_epoch != existing.journal_epoch:
        return candidate.journal_epoch > existing.journal_epoch
    return candidate.journal_sequence > existing.journal_sequence


def _environment_summary(environment: EnvironmentSampleIn | None) -> dict[str, Any]:
    if environment is None:
        return {}
    return {
        "temperature_c": environment.temperature_c,
        "humidity_fraction": environment.humidity_fraction,
        "pressure_pa": environment.pressure_pa,
        "wind_speed_mps": environment.wind_speed_mps,
        "wind_dir_deg": environment.wind_dir_deg,
        "solar_lux": environment.solar_lux,
        "metadata": dict(environment.metadata),
    }