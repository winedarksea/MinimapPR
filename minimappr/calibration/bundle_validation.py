"""Validate a calibration capture bundle against the field capture protocol.

``docs/field_capture_protocol.md`` is the protocol; this module checks a
bundle zip (``calibration/bundle.py``) against it and returns structured
findings with stable codes (``bundle_validation_findings.py``). It never
raises on a bad bundle and never blocks extraction or export: run it on site
before leaving, from the CLI or ``GET /api/v1/calibration/{session_id}/validation``.

    python -m minimappr.calibration.bundle_validation path/to/bundle.zip [--json]

Exit status: 1 when the report has errors (or warnings with ``--strict``), else 0.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import wave
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from minimappr.calibration.bundle import SCHEMA_VERSION as BUNDLE_SCHEMA_VERSION
from minimappr.calibration.bundle_frame_timing_checks import check_node_frame_timing
from minimappr.calibration.bundle_ground_truth_checks import (
    check_ground_truth,
    reference_node_ids,
    valid_ground_truth_events,
)
from minimappr.calibration.bundle_node_checks import (
    WavHeader,
    capture_window_ns,
    check_cluster_sample_rates,
    check_node_audio,
    check_node_survey,
    check_site_frame,
    node_position_in_site_frame,
)
from minimappr.calibration.bundle_validation_findings import (
    BundleFindingCode as Code,
    BundleValidationConfig,
    BundleValidationFinding as Finding,
    BundleValidationReport,
    FindingSeverity,
)
from minimappr.calibration.pipeline import BUNDLE_KIND

NANOSECONDS_PER_SECOND = 1_000_000_000
PCM16_SAMPLE_WIDTH_BYTES = 2
_SEVERITY_ORDER = {FindingSeverity.ERROR: 0, FindingSeverity.WARNING: 1, FindingSeverity.INFO: 2}


def _read_json_member(bundle_zip: zipfile.ZipFile, member_name: str) -> Any:
    return json.loads(bundle_zip.read(member_name))


def _read_wav_header(bundle_zip: zipfile.ZipFile, member_name: Any) -> WavHeader | None:
    if not isinstance(member_name, str) or member_name not in bundle_zip.namelist():
        return None
    try:
        with bundle_zip.open(member_name) as raw, wave.open(raw) as wav:
            if wav.getsampwidth() != PCM16_SAMPLE_WIDTH_BYTES:
                return None
            return WavHeader(wav.getnchannels(), wav.getframerate(), wav.getnframes())
    except (wave.Error, EOFError):
        return None


def _node_frame_timing_findings(
    bundle_zip: zipfile.ZipFile, node: Mapping[str, Any], wav_header: WavHeader | None, config: BundleValidationConfig
) -> list[Finding]:
    node_id = str(node.get("node_id"))
    timing_file = node.get("frame_timing_file")
    if not timing_file:
        return [
            Finding(
                Code.FRAME_TIMING_MISSING,
                "no frame_timing_file: per-frame packet time and time_quality are unknown, so steps cannot be seen",
                node_id=node_id,
            )
        ]
    try:
        payload = _read_json_member(bundle_zip, timing_file)
    except (KeyError, ValueError) as exc:
        return [Finding(Code.FRAME_TIMING_UNREADABLE, f"{timing_file}: {exc}", node_id=node_id)]
    if not isinstance(payload, dict):
        return [Finding(Code.FRAME_TIMING_UNREADABLE, f"{timing_file} is not a JSON object", node_id=node_id)]
    return check_node_frame_timing(node, payload, wav_header, config)


def _capture_findings(window_ns: tuple[int, int] | None, nodes: Sequence[Mapping], event_count: int, config: BundleValidationConfig) -> list[Finding]:
    if window_ns is None:
        return []
    duration_seconds = (window_ns[1] - window_ns[0]) / NANOSECONDS_PER_SECOND
    findings: list[Finding] = []
    if duration_seconds < config.minimum_capture_duration_seconds:
        findings.append(
            Finding(
                Code.CAPTURE_TOO_SHORT,
                f"capture is {duration_seconds:.1f} s (protocol minimum {config.minimum_capture_duration_seconds:.0f} s)",
                details={"duration_seconds": duration_seconds},
            )
        )
    sample_rates = sorted({node.get("sample_rate_hz") for node in nodes if isinstance(node.get("sample_rate_hz"), int)})
    findings.append(
        Finding(
            Code.CAPTURE_SUMMARY,
            f"{duration_seconds:.1f} s, {len(nodes)} node(s), {event_count} ground-truth event(s), rates {sample_rates}",
            details={
                "duration_seconds": duration_seconds,
                "node_count": len(nodes),
                "ground_truth_event_count": event_count,
                "sample_rates_hz": sample_rates,
            },
        )
    )
    return findings


def _validate_open_bundle(bundle_zip: zipfile.ZipFile, config: BundleValidationConfig) -> list[Finding]:
    names = set(bundle_zip.namelist())
    try:
        manifest = _read_json_member(bundle_zip, "manifest.json")
    except (KeyError, ValueError) as exc:
        return [Finding(Code.BUNDLE_UNREADABLE, f"manifest.json: {exc}")]
    if not isinstance(manifest, dict) or manifest.get("kind") != BUNDLE_KIND:
        return [Finding(Code.BUNDLE_UNREADABLE, f"manifest.json is not a {BUNDLE_KIND}")]
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        return [Finding(Code.BUNDLE_UNREADABLE, f"unsupported manifest schema_version {manifest.get('schema_version')!r}")]
    nodes = manifest.get("nodes")
    if not isinstance(nodes, list) or not nodes or not all(isinstance(node, dict) for node in nodes):
        return [Finding(Code.BUNDLE_UNREADABLE, "manifest.json lists no nodes")]

    ground_truth_payload: Any = None
    findings: list[Finding] = []
    if "ground_truth.json" in names:
        try:
            ground_truth_payload = _read_json_member(bundle_zip, "ground_truth.json")
        except ValueError as exc:
            findings.append(Finding(Code.GROUND_TRUTH_INVALID, f"ground_truth.json: {exc}"))
        if ground_truth_payload is not None and not isinstance(ground_truth_payload, dict):
            findings.append(Finding(Code.GROUND_TRUTH_INVALID, "ground_truth.json is not a JSON object"))
            ground_truth_payload = None
    raw_events = valid_ground_truth_events(ground_truth_payload)
    reference_ids = reference_node_ids(raw_events)

    site_findings, frame = check_site_frame(manifest)
    findings += site_findings
    window_ns = capture_window_ns(manifest)
    localizing_node_positions_m: dict[str, np.ndarray] = {}
    for node in nodes:
        node_id = str(node.get("node_id"))
        wav_header = _read_wav_header(bundle_zip, node.get("audio_file"))
        findings += check_node_survey(node, frame, config)
        findings += check_node_audio(node, wav_header, window_ns, config)
        findings += _node_frame_timing_findings(bundle_zip, node, wav_header, config)
        position_m = node_position_in_site_frame(node, frame)
        if position_m is not None and node_id not in reference_ids:
            localizing_node_positions_m[node_id] = position_m
    findings += check_cluster_sample_rates(nodes, reference_ids)
    if not any(finding.code is Code.GROUND_TRUTH_INVALID for finding in findings):
        findings += check_ground_truth(
            ground_truth_payload,
            window_ns=window_ns,
            frame=frame,
            bundle_node_ids={str(node.get("node_id")) for node in nodes},
            localizing_node_positions_m=localizing_node_positions_m,
            config=config,
        )
    findings += _capture_findings(window_ns, nodes, len(raw_events), config)
    return findings


def validate_calibration_bundle(bundle_path: Path | str, config: BundleValidationConfig | None = None) -> BundleValidationReport:
    resolved_config = config if config is not None else BundleValidationConfig()
    path = Path(bundle_path)
    try:
        with zipfile.ZipFile(path) as bundle_zip:
            findings = _validate_open_bundle(bundle_zip, resolved_config)
    except (OSError, zipfile.BadZipFile) as exc:
        findings = [Finding(Code.BUNDLE_UNREADABLE, f"cannot open bundle: {exc}")]
    ordered = sorted(findings, key=lambda finding: _SEVERITY_ORDER[finding.severity])
    return BundleValidationReport(bundle_path=str(path), findings=tuple(ordered))


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    defaults = BundleValidationConfig()
    parser = argparse.ArgumentParser(description="Validate a calibration capture bundle against the field capture protocol.")
    parser.add_argument("bundle", type=Path, help="calibration bundle zip")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--strict", action="store_true", help="exit 1 on warnings as well as errors")
    parser.add_argument("--minimum-events-per-range-rung", type=int, default=defaults.minimum_events_per_range_rung)
    parser.add_argument("--minimum-capture-duration-seconds", type=float, default=defaults.minimum_capture_duration_seconds)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    config = dataclasses.replace(
        BundleValidationConfig(),
        minimum_events_per_range_rung=arguments.minimum_events_per_range_rung,
        minimum_capture_duration_seconds=arguments.minimum_capture_duration_seconds,
    )
    report = validate_calibration_bundle(arguments.bundle, config)
    if arguments.json:
        print(json.dumps(report.to_json_payload(), indent=2))
    else:
        for finding in report.findings:
            print(finding.one_line())
        payload = report.to_json_payload()
        print(f"{payload['error_count']} error(s), {payload['warning_count']} warning(s), {payload['info_count']} info")
    failed = report.has_errors or (arguments.strict and report.has_warnings)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
