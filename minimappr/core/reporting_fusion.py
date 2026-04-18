"""Reporting-window canonicalization for localized and omni detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from minimappr.interfaces import StorageBackend


ReportingAction = Literal["insert", "upgrade_existing", "enrich_existing", "suppress"]
ReportingModality = Literal["localized", "omni"]


@dataclass(slots=True)
class ReportingDecision:
    action: ReportingAction
    report_window_start_ns: int
    report_window_end_ns: int
    reporting_modality: ReportingModality
    branch_evidence: dict[str, Any]
    existing_detection: dict[str, Any] | None = None
    suppression_reason: str | None = None


class ReportingFusionPolicy:
    """Choose the canonical detection for a label inside a reporting window."""

    def __init__(
        self,
        *,
        storage: StorageBackend,
        reporting_window_seconds: float,
    ) -> None:
        self._storage = storage
        self._reporting_window_ns = max(1, int(reporting_window_seconds * 1_000_000_000))

    def report_window_bounds(self, event_time_ns: int) -> tuple[int, int]:
        start_ns = (event_time_ns // self._reporting_window_ns) * self._reporting_window_ns
        return start_ns, start_ns + self._reporting_window_ns

    async def decide(
        self,
        *,
        event_time_ns: int,
        source_node_id: str | None,
        label: str,
        reporting_modality: ReportingModality,
        branch_details: dict[str, Any],
    ) -> ReportingDecision:
        window_start_ns, window_end_ns = self.report_window_bounds(event_time_ns)
        existing = await self._storage.find_detection_for_reporting_window(
            source_node_id=source_node_id,
            label=label,
            report_window_start_ns=window_start_ns,
            report_window_end_ns=window_end_ns,
        )
        merged_branch_evidence = self._merged_branch_evidence(
            existing_detection=existing,
            reporting_modality=reporting_modality,
            branch_details=branch_details,
        )
        if existing is None:
            return ReportingDecision(
                action="insert",
                report_window_start_ns=window_start_ns,
                report_window_end_ns=window_end_ns,
                reporting_modality=reporting_modality,
                branch_evidence=merged_branch_evidence,
            )

        existing_modality = str(existing.get("reporting_modality") or "localized").strip().lower()
        if existing_modality == "omni" and reporting_modality == "localized":
            return ReportingDecision(
                action="upgrade_existing",
                report_window_start_ns=window_start_ns,
                report_window_end_ns=window_end_ns,
                reporting_modality="localized",
                branch_evidence=merged_branch_evidence,
                existing_detection=existing,
            )

        if existing_modality == "localized" and reporting_modality == "omni":
            omni_details = dict(branch_details)
            omni_details["suppressed"] = True
            omni_details["suppression_reason"] = "localized_detection_already_canonical"
            merged_branch_evidence["omni"] = omni_details
            return ReportingDecision(
                action="enrich_existing",
                report_window_start_ns=window_start_ns,
                report_window_end_ns=window_end_ns,
                reporting_modality="localized",
                branch_evidence=merged_branch_evidence,
                existing_detection=existing,
                suppression_reason="localized_detection_already_canonical",
            )

        return ReportingDecision(
            action="suppress",
            report_window_start_ns=window_start_ns,
            report_window_end_ns=window_end_ns,
            reporting_modality=existing_modality if existing_modality in {"localized", "omni"} else reporting_modality,
            branch_evidence=merged_branch_evidence,
            existing_detection=existing,
            suppression_reason="report_window_duplicate",
        )

    @staticmethod
    def _merged_branch_evidence(
        *,
        existing_detection: dict[str, Any] | None,
        reporting_modality: ReportingModality,
        branch_details: dict[str, Any],
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if existing_detection is not None:
            existing_feature_summary = existing_detection.get("feature_summary", {})
            if isinstance(existing_feature_summary, dict):
                current = existing_feature_summary.get("branch_evidence", {})
                if isinstance(current, dict):
                    merged = {str(key): value for key, value in current.items()}
        merged[reporting_modality] = dict(branch_details)
        return merged
