"""Tables for the faint-call admission sweep (markdown, pasted into
``docs/faint_call_admission_results.md``) and its JSON dump."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from tests.faint_admission_offline_replays import OmniScanReplayResult, TriggerNoiseReplayResult
from tests.faint_admission_sweep import AdmissionCellResult, CallAdmissionRow, cell_result_to_json
from tests.faint_admission_sweep_fixtures import ALIGNMENT_STARTUP, SWEEP_SNR_STEPS_DB

PERCENT = 100.0
NOT_MEASURED = "-"


def _percent(rows: Sequence[CallAdmissionRow], stage: str) -> str:
    if not rows:
        return NOT_MEASURED
    return f"{PERCENT * sum(bool(getattr(row, stage)) for row in rows) / len(rows):.0f}%"


def _optional(value: float | None, digits: int = 1) -> str:
    return NOT_MEASURED if value is None else f"{value:.{digits}f}"


def funnel_summary_table(
    cells: Sequence[AdmissionCellResult], noise_replays: Sequence[TriggerNoiseReplayResult]
) -> str:
    replay_by_key = {(replay.candidate, replay.background): replay for replay in noise_replays}
    lines = [
        "| candidate | background | calls | received | admitted | solved | classified | stored localized | periodic omni coverage "
        "| false trig/h (app) | false trig/h (offline steady) | startup false trig (offline, first 30 s) "
        "| trigger us/frame (Python) | run wall s | app CPU s | sidecar CPU s | sidecar drain wait s | load 1m start/end |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        rows = cell.rows
        replay = replay_by_key.get((cell.candidate, cell.background))
        lines.append(
            f"| {cell.candidate} | {cell.background} | {len(rows)} | {_percent(rows, 'received')} | {_percent(rows, 'admitted')} "
            f"| {_percent(rows, 'solved')} | {_percent(rows, 'classified')} | {_percent(rows, 'stored')} "
            f"| {_percent(rows, 'periodic_omni_coverage')} "
            f"| {cell.false_triggers_per_hour:.0f} ({cell.false_triggers} in {cell.false_trigger_exposure_seconds:.0f} s, {cell.startup_false_triggers} before 30 s) "
            f"| {NOT_MEASURED if replay is None else f'{replay.steady_false_triggers_per_hour:.0f}'} "
            f"| {NOT_MEASURED if replay is None else replay.startup_false_triggers} "
            f"| {NOT_MEASURED if replay is None else f'{replay.microseconds_per_frame:.0f}'} "
            f"| {cell.wall_seconds:.0f} | {cell.test_process_cpu_seconds:.0f} | {_optional(cell.sidecar_cpu_seconds, 0)} "
            f"| {_optional(cell.sidecar_drain_wait_seconds, 0)}{'' if cell.sidecar_caught_up in (None, True) else ' (not caught up)'} "
            f"| {cell.load_average_1m_start:.1f}/{cell.load_average_1m_end:.1f} |"
        )
    return "\n".join(lines)


def admission_by_kind_table(cells: Sequence[AdmissionCellResult]) -> str:
    """Admitted calls per kind and alignment, and the faintest step admitted."""
    lines = [
        "| candidate | background | kind | alignment | admitted (of steps) | faintest admitted dB | solved | stored |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        groups: dict[tuple[str, str], list[CallAdmissionRow]] = defaultdict(list)
        for row in cell.rows:
            groups[(row.kind, row.alignment)].append(row)
        for (kind, alignment), rows in groups.items():
            admitted = [row for row in rows if row.admitted]
            faintest = min((row.in_band_snr_db for row in admitted), default=None)
            lines.append(
                f"| {cell.candidate} | {cell.background} | {kind} | {alignment} | {len(admitted)}/{len(rows)} "
                f"| {_optional(faintest, 0)} | {sum(row.solved for row in rows)}/{len(rows)} "
                f"| {sum(row.stored for row in rows)}/{len(rows)} |"
            )
    return "\n".join(lines)


def omni_replay_table(
    omni_results: Sequence[tuple[str, OmniScanReplayResult]], call_count: int, clearest_gain_db_by_kind: dict[str, float]
) -> str:
    lines = [
        "| background | gate | scan phase | calls classified | scans | gate passes with no call (per hour) |",
        "|---|---|---|---|---|---|",
    ]
    for background, result in omni_results:
        per_hour = result.scans_passing_gate_without_call * 3600.0 / result.capture_seconds
        lines.append(
            f"| {background} | {result.gate_mode} | {result.phase} | {len(result.classified_call_indices)}/{call_count} "
            f"| {result.scans} | {result.scans_passing_gate_without_call} ({per_hour:.0f}) |"
        )
    gains = ", ".join(f"{kind} {gain:+.1f} dB" for kind, gain in clearest_gain_db_by_kind.items())
    lines.append("")
    lines.append(f"Median clearest-channel over channel-mean power during the call (quiet site, top step): {gains}.")
    return "\n".join(lines)


def schedule_note() -> str:
    return (
        f"Steps {', '.join(f'{step:g}' for step in SWEEP_SNR_STEPS_DB)} dB in-band SNR per kind and alignment, "
        f"plus two `{ALIGNMENT_STARTUP}` syllables."
    )


def write_sweep_report(
    path: Path,
    cells: Sequence[AdmissionCellResult],
    noise_replays: Sequence[TriggerNoiseReplayResult],
    omni_results: Sequence[tuple[str, OmniScanReplayResult]],
) -> None:
    payload = {
        "cells": [cell_result_to_json(cell) for cell in cells],
        "noise_replays": [
            {**asdict(replay), "steady_false_triggers_per_hour": replay.steady_false_triggers_per_hour}
            for replay in noise_replays
        ],
        "omni_replays": [
            {**asdict(result), "background": background, "classified_call_indices": sorted(result.classified_call_indices)}
            for background, result in omni_results
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
