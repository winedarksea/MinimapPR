"""Reporting-window canonicalization for localized and omni detections."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from minimappr.interfaces import StorageBackend, TaxonomyProvider


def _position_tuple(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict) and {"x", "y", "z"} <= value.keys():
        return (float(value["x"]), float(value["y"]), float(value["z"]))
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return None


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
        omni_suppression_scope: str = "site",
        omni_suppression_max_distance_m: float = 0.0,
        taxonomy_provider: TaxonomyProvider | None = None,
    ) -> None:
        self._storage = storage
        self._reporting_window_ns = max(1, int(reporting_window_seconds * 1_000_000_000))
        self._omni_suppression_scope = omni_suppression_scope.strip().lower()
        self._omni_suppression_max_distance_m = max(0.0, omni_suppression_max_distance_m)
        self._taxonomy_provider = taxonomy_provider

    def _canonical_label(self, label: str) -> str:
        # Only the site-wide lookup canonicalizes, which resolves an incoming alias
        # against a stored canonical row. The reverse (a row persisted under an
        # alias) is still missed, because rows persist the raw label; closing that
        # gap requires canonicalizing at persist time.
        resolver = getattr(self._taxonomy_provider, "canonical_label", None)
        if callable(resolver):
            return str(resolver(label))
        return label

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
        source_position_m: tuple[float, float, float] | None = None,
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
            site_decision = await self._site_wide_omni_suppression(
                label=self._canonical_label(label),
                reporting_modality=reporting_modality,
                branch_details=branch_details,
                window_start_ns=window_start_ns,
                window_end_ns=window_end_ns,
                source_position_m=source_position_m,
            )
            if site_decision is not None:
                return site_decision
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

    async def _site_wide_omni_suppression(
        self,
        *,
        label: str,
        reporting_modality: ReportingModality,
        branch_details: dict[str, Any],
        window_start_ns: int,
        window_end_ns: int,
        source_position_m: tuple[float, float, float] | None,
    ) -> ReportingDecision | None:
        """Omni→localized suppression check across all nodes (scope="site").

        Only the suppression check goes site-wide: an omni report whose label
        already has a localized detection anywhere on the site in this
        reporting window enriches that detection instead of inserting a
        duplicate. Insert/upgrade paths keep per-node keying.
        """
        if reporting_modality != "omni" or self._omni_suppression_scope != "site":
            return None
        site_existing = await self._storage.find_detection_for_reporting_window(
            source_node_id=None,
            label=label,
            report_window_start_ns=window_start_ns,
            report_window_end_ns=window_end_ns,
            any_node=True,
        )
        if site_existing is None:
            return None
        existing_modality = str(site_existing.get("reporting_modality") or "localized").strip().lower()
        if existing_modality != "localized":
            return None
        if self._omni_suppression_max_distance_m > 0.0 and source_position_m is not None:
            existing_position = _position_tuple(site_existing.get("position_m"))
            if existing_position is not None:
                distance = math.dist(
                    existing_position,
                    tuple(float(v) for v in source_position_m[:3]),
                )
                if distance > self._omni_suppression_max_distance_m:
                    return None
        omni_details = dict(branch_details)
        omni_details["suppressed"] = True
        omni_details["suppression_reason"] = "localized_detection_site_wide"
        merged = self._merged_branch_evidence(
            existing_detection=site_existing,
            reporting_modality=reporting_modality,
            branch_details=omni_details,
        )
        return ReportingDecision(
            action="enrich_existing",
            report_window_start_ns=window_start_ns,
            report_window_end_ns=window_end_ns,
            reporting_modality="localized",
            branch_evidence=merged,
            existing_detection=site_existing,
            suppression_reason="localized_detection_site_wide",
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
