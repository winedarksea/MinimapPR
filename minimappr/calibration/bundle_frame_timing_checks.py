"""Per-frame timing and clock checks for bundle validation.

Reads the raw ``frame_timing/{node_id}.json`` payload (``calibration/frame_timing.py``)
rather than ``NodeFrameTimingTable``, whose constructor raises on the first
overlap: the protocol report names every defect with its own code instead.

Clock evidence mirrors what the sidecar measures (``docs/localization_solve_contract.md``
section 7): a step is a frame-to-frame packet-time advance that disagrees with
the sample advance by more than the step threshold; the timeline residual is
packet time minus the nominal-rate timeline anchored at the first frame. The
sidecar anchors at its buffer's first packet and takes a windowed percentile,
so this is the capture-wide view of the same quantity. Drift is reported, never
corrected (not in this round).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from minimappr.calibration.bundle_node_checks import WavHeader
from minimappr.calibration.bundle_validation_findings import (
    BundleFindingCode as Code,
    BundleValidationConfig,
    BundleValidationFinding as Finding,
)
from minimappr.calibration.frame_timing import FRAME_TIMING_SCHEMA_VERSION
from minimappr.models import TimeQuality

NANOSECONDS_PER_SECOND = 1_000_000_000
PARTS_PER_MILLION = 1.0e6
_KNOWN_TIME_QUALITY_VALUES = frozenset(quality.value for quality in TimeQuality)


@dataclass(frozen=True, slots=True)
class _RawFrame:
    index: int
    sample_offset: int
    sample_count: int
    start_time_ns: int
    time_quality: str | None
    sequence: int | None


def _parse_frames(payload: Mapping[str, Any]) -> list[_RawFrame]:
    return [
        _RawFrame(
            index=index,
            sample_offset=int(entry["sample_offset"]),
            sample_count=int(entry["sample_count"]),
            start_time_ns=int(entry["start_time_ns"]),
            time_quality=entry.get("time_quality"),
            sequence=None if entry.get("sequence") is None else int(entry["sequence"]),
        )
        for index, entry in enumerate(payload["frames"])
    ]


def _structure_findings(node_id: str, frames: list[_RawFrame], wav_header: WavHeader | None) -> list[Finding]:
    empty_frame_indices = [frame.index for frame in frames if frame.sample_count <= 0]
    backwards_frame_indices: list[int] = []
    overlapping_frame_indices: list[int] = []
    for previous, frame in zip(frames, frames[1:]):
        if frame.sample_offset < previous.sample_offset:
            backwards_frame_indices.append(frame.index)
        elif frame.sample_offset < previous.sample_offset + previous.sample_count:
            overlapping_frame_indices.append(frame.index)
    outside_frame_indices = (
        [
            frame.index
            for frame in frames
            if frame.sample_offset < 0 or frame.sample_offset + frame.sample_count > wav_header.total_samples
        ]
        if wav_header is not None
        else []
    )
    findings: list[Finding] = []
    for code, indices, what in (
        (Code.FRAME_TIMING_EMPTY_FRAME, empty_frame_indices, "frame(s) with no samples"),
        (Code.FRAME_TIMING_NOT_MONOTONIC, backwards_frame_indices, "frame(s) start before the previous frame"),
        (Code.FRAME_TIMING_OVERLAP, overlapping_frame_indices, "frame(s) overlap the previous frame"),
        (Code.FRAME_TIMING_OUTSIDE_AUDIO, outside_frame_indices, "frame(s) lie outside the node's WAV"),
    ):
        if indices:
            findings.append(
                Finding(
                    code,
                    f"{len(indices)} {what}; first at frame index {indices[0]}",
                    node_id=node_id,
                    details={"frame_count": len(indices), "first_frame_index": indices[0]},
                )
            )
    return findings


def _time_quality_findings(node_id: str, frames: list[_RawFrame]) -> tuple[list[Finding], Counter]:
    counts = Counter("unreported" if frame.time_quality is None else frame.time_quality for frame in frames)
    findings: list[Finding] = []
    if counts.get("unreported"):
        findings.append(
            Finding(
                Code.TIME_QUALITY_UNREPORTED,
                f"{counts['unreported']} frame(s) carry no time_quality; the clock class must be reported per frame",
                node_id=node_id,
            )
        )
    unknown_values = sorted(value for value in counts if value != "unreported" and value not in _KNOWN_TIME_QUALITY_VALUES)
    if unknown_values:
        findings.append(Finding(Code.TIME_QUALITY_UNKNOWN_VALUE, f"unknown time_quality value(s) {unknown_values}", node_id=node_id))
    if counts.get(TimeQuality.FREE_RUNNING.value):
        findings.append(
            Finding(
                Code.TIME_QUALITY_FREE_RUNNING,
                f"{counts[TimeQuality.FREE_RUNNING.value]} frame(s) free_running: cross-node pairs are kept "
                "but carry the wide free-running uncertainty",
                node_id=node_id,
            )
        )
    transitions = sum(1 for previous, frame in zip(frames, frames[1:]) if previous.time_quality != frame.time_quality)
    if transitions:
        findings.append(
            Finding(
                Code.TIME_QUALITY_CHANGED,
                f"time_quality changed {transitions} time(s) during the capture: {dict(counts)}",
                node_id=node_id,
                details={"transition_count": transitions},
            )
        )
    return findings, counts


def _clock_findings(
    node_id: str, frames: list[_RawFrame], sample_rate_hz: int, quality_counts: Counter, config: BundleValidationConfig
) -> list[Finding]:
    offsets = np.asarray([frame.sample_offset for frame in frames], dtype=np.float64)
    packet_seconds = np.asarray([frame.start_time_ns - frames[0].start_time_ns for frame in frames], dtype=np.float64)
    packet_seconds /= NANOSECONDS_PER_SECOND
    nominal_seconds = (offsets - offsets[0]) / sample_rate_hz
    residual_seconds = packet_seconds - nominal_seconds
    step_seconds = np.diff(residual_seconds)
    step_mask = np.abs(step_seconds) > config.timestamp_step_threshold_seconds
    step_frame_indices = [frames[position + 1].index for position in np.flatnonzero(step_mask)]
    findings: list[Finding] = []
    if step_frame_indices:
        largest_step_seconds = float(step_seconds[step_mask][np.argmax(np.abs(step_seconds[step_mask]))])
        findings.append(
            Finding(
                Code.TIMESTAMP_STEP_DETECTED,
                f"{len(step_frame_indices)} timestamp step(s) over {config.timestamp_step_threshold_seconds * 1e3:.1f} ms; "
                f"largest {largest_step_seconds * 1e3:+.1f} ms at frame index {step_frame_indices[0]}",
                node_id=node_id,
                details={"step_frame_indices": step_frame_indices, "largest_step_seconds": largest_step_seconds},
            )
        )
    max_abs_residual_seconds = float(np.max(np.abs(residual_seconds)))
    if max_abs_residual_seconds > config.cross_node_max_timeline_residual_seconds:
        findings.append(
            Finding(
                Code.TIMELINE_RESIDUAL_OVER_LIMIT,
                f"packet time strays {max_abs_residual_seconds * 1e3:.1f} ms from the nominal timeline "
                f"(limit {config.cross_node_max_timeline_residual_seconds * 1e3:.1f} ms): the sidecar drops this "
                "node's cross-node pairs while it is over the limit",
                node_id=node_id,
                details={"max_abs_timeline_residual_seconds": max_abs_residual_seconds},
            )
        )
    # Drift: slope of the residual once detected steps are taken out.
    step_free_residual = residual_seconds - np.concatenate(([0.0], np.cumsum(np.where(step_mask, step_seconds, 0.0))))
    drift_ppm = (
        float(np.polyfit(nominal_seconds, step_free_residual, 1)[0] * PARTS_PER_MILLION)
        if len(frames) >= 2 and nominal_seconds[-1] > 0.0
        else None
    )
    findings.append(
        Finding(
            Code.NODE_CLOCK_SUMMARY,
            f"{len(frames)} frames, time_quality {dict(quality_counts)}, {len(step_frame_indices)} step(s), "
            f"max |residual| {max_abs_residual_seconds * 1e3:.2f} ms, drift "
            + ("n/a" if drift_ppm is None else f"{drift_ppm:+.1f} ppm"),
            node_id=node_id,
            details={
                "frame_count": len(frames),
                "time_quality_counts": dict(quality_counts),
                "timestamp_step_count": len(step_frame_indices),
                "max_abs_timeline_residual_seconds": max_abs_residual_seconds,
                "final_timeline_residual_seconds": float(residual_seconds[-1]),
                "drift_ppm": drift_ppm,
            },
        )
    )
    return findings


def _gap_findings(
    node_id: str, frames: list[_RawFrame], sample_rate_hz: int, wav_header: WavHeader | None, config: BundleValidationConfig
) -> list[Finding]:
    internal_gap_samples = [
        frame.sample_offset - (previous.sample_offset + previous.sample_count)
        for previous, frame in zip(frames, frames[1:])
        if frame.sample_offset > previous.sample_offset + previous.sample_count
    ]
    timeline_samples = wav_header.total_samples if wav_header is not None else frames[-1].sample_offset + frames[-1].sample_count
    edge_gap_samples = frames[0].sample_offset + max(0, timeline_samples - (frames[-1].sample_offset + frames[-1].sample_count))
    lost_samples = sum(internal_gap_samples) + edge_gap_samples
    gap_fraction = lost_samples / timeline_samples if timeline_samples > 0 else 0.0
    sequence_jumps = sum(
        1
        for previous, frame in zip(frames, frames[1:])
        if previous.sequence is not None and frame.sequence is not None and frame.sequence != previous.sequence + 1
    )
    details = {
        "gap_count": len(internal_gap_samples),
        "lost_seconds": lost_samples / sample_rate_hz,
        "longest_gap_seconds": max(internal_gap_samples, default=0) / sample_rate_hz,
        "gap_fraction": gap_fraction,
        "sequence_jump_count": sequence_jumps,
    }
    findings = [
        Finding(
            Code.NODE_GAP_STATISTICS,
            f"{details['gap_count']} gap(s), {details['lost_seconds']:.3f} s lost ({gap_fraction:.1%}), "
            f"longest {details['longest_gap_seconds']:.3f} s, {sequence_jumps} sequence jump(s)",
            node_id=node_id,
            details=details,
        )
    ]
    if gap_fraction > config.maximum_node_gap_fraction:
        findings.append(
            Finding(
                Code.NODE_GAP_FRACTION_HIGH,
                f"{gap_fraction:.1%} of the node's audio was lost (limit {config.maximum_node_gap_fraction:.1%})",
                node_id=node_id,
                details={"gap_fraction": gap_fraction},
            )
        )
    return findings


def check_node_frame_timing(
    node: Mapping[str, Any],
    timing_payload: Mapping[str, Any],
    wav_header: WavHeader | None,
    config: BundleValidationConfig,
) -> list[Finding]:
    """Checks one readable frame timing payload; absence is reported by the caller."""
    node_id = str(node.get("node_id"))
    try:
        if int(timing_payload.get("schema_version", 0)) != FRAME_TIMING_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {timing_payload.get('schema_version')!r}")
        table_node_id = str(timing_payload["node_id"])
        sample_rate_hz = int(timing_payload["sample_rate_hz"])
        frames = _parse_frames(timing_payload)
    except (KeyError, TypeError, ValueError) as exc:
        return [Finding(Code.FRAME_TIMING_UNREADABLE, f"frame timing table is malformed: {exc}", node_id=node_id)]
    findings: list[Finding] = []
    if table_node_id != node_id:
        findings.append(Finding(Code.FRAME_TIMING_NODE_MISMATCH, f"frame timing table describes node {table_node_id!r}", node_id=node_id))
    if sample_rate_hz != node.get("sample_rate_hz") or (wav_header is not None and sample_rate_hz != wav_header.sample_rate_hz):
        findings.append(
            Finding(
                Code.NODE_SAMPLE_RATE_INCONSISTENT,
                f"frame timing says {sample_rate_hz} Hz, manifest {node.get('sample_rate_hz')} Hz",
                node_id=node_id,
            )
        )
    if not frames or sample_rate_hz <= 0:
        return findings + [Finding(Code.FRAME_TIMING_UNREADABLE, "frame timing table has no frames or no rate", node_id=node_id)]
    structure_findings = _structure_findings(node_id, frames, wav_header)
    quality_findings, quality_counts = _time_quality_findings(node_id, frames)
    findings += structure_findings + quality_findings
    if structure_findings:
        # Steps and gaps measured on a broken table would be noise.
        return findings
    findings += _clock_findings(node_id, frames, sample_rate_hz, quality_counts, config)
    findings += _gap_findings(node_id, frames, sample_rate_hz, wav_header, config)
    return findings
