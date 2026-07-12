"""Structured diagnostics helpers for operator and debug APIs."""

from __future__ import annotations

import platform
import tempfile
from pathlib import Path
from typing import Any

from minimappr.classifiers.chaining import ChainedClassifier
from minimappr.config import Settings
from minimappr.interfaces import StorageBackend


class DiagnosticsService:
    """Build stable, machine-readable diagnostics for API clients."""

    def __init__(
        self,
        *,
        settings: Settings,
        storage: StorageBackend,
        fusion_node,
        classifier,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._fusion_node = fusion_node
        self._classifier = classifier

    def replace_classifier(self, classifier) -> None:
        self._classifier = classifier

    async def config_snapshot(self) -> dict[str, Any]:
        classifier_runtime = self._classifier_runtime()
        return {
            "runtime": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "classification_audio_source": self._settings.classification_audio_source,
                "classifier_backend_resolved": self._settings.resolved_classifier_backend(),
                "classifier": classifier_runtime,
                "beamforming": {
                    "enabled": self._settings.beamformed_classification_enabled,
                    "beamformer_type": self._settings.beamformer_type,
                    "min_sensor_count": self._settings.beamformed_classification_min_sensor_count,
                    "confidence_margin": self._settings.beamformed_classification_confidence_margin,
                },
                "localization": {
                    "algorithm": self._settings.localization_algorithm,
                    "strategy": self._settings.localization_strategy,
                    "band_min_hz": self._settings.localization_band_min_hz,
                    "band_max_hz": self._settings.localization_band_max_hz,
                    "window_seconds": self._settings.localization_window_seconds,
                    "classification_window_seconds": self._settings.classification_window_seconds,
                    "reporting_window_seconds": self._settings.reporting_window_seconds,
                    "max_tau_seconds": self._settings.localization_max_tau_seconds,
                    "max_tau_s": self._settings.localization_max_tau_s,
                    "skip_localization_for_classification": (
                        self._settings.skip_localization_for_classification
                    ),
                },
                "paths": {
                    "db_path": str(self._settings.db_path),
                    "snippet_dir": str(self._settings.snippet_dir),
                    "large_artifact_dir": str(self._settings.large_artifact_dir),
                    "rules_config_path": str(self._settings.rules_config_path),
                    "taxonomy_config_path": str(self._settings.taxonomy_config_path),
                    "model_chain_config_path": str(self._settings.model_chain_config_path),
                },
            },
            "thresholds": {
                "trigger_rms": self._settings.trigger_rms,
                "trigger_cooldown_seconds": self._settings.trigger_cooldown_seconds,
                "sensor_energy_threshold_multiplier": self._settings.sensor_energy_threshold_multiplier,
                "fallback_localization_confidence": self._settings.fallback_localization_confidence,
                "yamnet_min_confidence": self._settings.yamnet_min_confidence,
                "yamnet_input_target_rms": self._settings.yamnet_input_target_rms,
                "yamnet_max_input_gain": self._settings.yamnet_max_input_gain,
                "birdnet_trigger_min_confidence": self._settings.birdnet_trigger_min_confidence,
            },
            "queues": {
                "fusion_worker_count": self._settings.fusion_worker_count,
                "event_queue_size": self._settings.fusion_event_queue_size,
                "localization_queue_size": self._settings.fusion_localization_queue_size,
                "classification_queue_size": self._settings.fusion_classification_queue_size,
                "rules_queue_size": self._settings.fusion_rules_queue_size,
                "drop_on_backpressure": self._settings.drop_on_backpressure,
                "offline_replay_mode": self._settings.fusion_offline_replay_mode,
            },
            "retention": {
                "snippet_retention_seconds": self._settings.snippet_retention_seconds,
                "ephemeral_seconds": self._settings.retention_ephemeral_seconds,
                "short_seconds": self._settings.retention_short_seconds,
                "long_seconds": self._settings.retention_long_seconds,
                "experiment_seconds": self._settings.retention_experiment_seconds,
            },
        }

    async def event_snapshot(self, event_id: str) -> dict[str, Any] | None:
        detection = await self._storage.get_detection_by_event_id(event_id)
        if detection is None:
            return None

        observations = await self._storage.list_observations_by_ids(detection.get("source_observation_ids", []))
        node = None
        source_node_id = detection.get("source_node_id")
        if isinstance(source_node_id, str) and source_node_id:
            node = await self._storage.get_node_by_id(source_node_id)

        track_id = detection.get("track_id")
        track_updates = await self._storage.list_track_updates(
            track_id=track_id,
            detection_id=detection["id"],
            event_id=detection["event_id"],
            limit=10,
        )
        alerts = await self._storage.list_alerts(
            limit=20,
            detection_id=detection["id"],
            track_id=track_id,
        )
        fusion_status = await self._fusion_node.status()

        window_seconds = float(self._settings.localization_window_seconds)
        half_window_ns = int(window_seconds * 0.5 * 1_000_000_000)

        return {
            "event_id": detection["event_id"],
            "detection_id": detection["id"],
            "ingest": {
                "node": node,
                "observations": observations,
            },
            "selection": {
                "selected_sensor_ids": detection.get("source_sensors", []),
                "reference_sensor": detection.get("reference_sensor"),
                "window": {
                    "center_time_ns": detection["timestamp_ns"],
                    "window_seconds": window_seconds,
                    "start_time_ns": detection["timestamp_ns"] - half_window_ns,
                    "end_time_ns": detection["timestamp_ns"] + half_window_ns,
                },
            },
            "localization": {
                "position_m": detection.get("position_m"),
                "position_geo": detection.get("position_geo"),
                "confidence": detection.get("confidence"),
                "gdop": detection.get("gdop"),
                "tdoa_s": detection.get("tdoa_s", {}),
                "method": detection.get("feature_summary", {}).get("localization_method"),
                "capability_tier": detection.get("feature_summary", {}).get("capability_tier"),
                "environment": detection.get("feature_summary", {}).get("environment"),
            },
            "classification": {
                "label": detection.get("label"),
                "label_category": detection.get("label_category"),
                "iff_category": detection.get("iff_category"),
                "confidence": detection.get("label_confidence"),
                "path": detection.get("feature_summary", {}).get("classification_path"),
                "scores": detection.get("classifier_scores", {}),
                "features": detection.get("feature_summary", {}),
            },
            "reporting": {
                "modality": detection.get("reporting_modality"),
                "report_window_start_ns": detection.get("report_window_start_ns"),
                "report_window_end_ns": detection.get("report_window_end_ns"),
                "branch_evidence": detection.get("feature_summary", {}).get("branch_evidence", {}),
            },
            "tracking": {
                "track_id": track_id,
                "updates": track_updates,
            },
            "alerts": alerts,
            "provenance": {
                "source_node_id": source_node_id,
                "source_observation_ids": detection.get("source_observation_ids", []),
                "zone_ids": detection.get("zone_ids", []),
                "snippet_path": detection.get("snippet_path"),
            },
            "runtime": {
                "classifier": self._classifier_runtime(),
                "fusion_metrics": fusion_status.get("metrics", {}),
            },
        }

    async def selftest(self) -> dict[str, Any]:
        fusion_status = await self._fusion_node.status()
        checks = [
            self._path_writable_check("db_parent_writable", self._settings.db_path.parent),
            self._path_writable_check("snippet_dir_writable", self._settings.snippet_dir),
            self._path_writable_check("large_artifact_dir_writable", self._settings.large_artifact_dir),
            {
                "name": "classifier_active",
                "ok": self._classifier is not None,
                "detail": self._classifier_runtime(),
            },
            {
                "name": "fusion_workers_running",
                "ok": bool(
                    fusion_status["workers"].get("localization_running", 0)
                    and fusion_status["workers"].get("classification_running", 0)
                    and fusion_status["workers"].get("rules_running", 0)
                ),
                "detail": fusion_status["workers"],
            },
            {
                "name": "rules_config_present",
                "ok": self._settings.rules_config_path.exists(),
                "detail": str(self._settings.rules_config_path),
            },
            {
                "name": "taxonomy_config_present",
                "ok": self._settings.taxonomy_config_path.exists(),
                "detail": str(self._settings.taxonomy_config_path),
            },
        ]
        passed = sum(1 for check in checks if check["ok"])
        return {
            "ok": passed == len(checks),
            "summary": {
                "passed": passed,
                "total": len(checks),
            },
            "checks": checks,
            "runtime": {
                "python_version": platform.python_version(),
                "classifier": self._classifier_runtime(),
            },
        }

    def _classifier_runtime(self) -> dict[str, Any]:
        classifier = self._classifier
        base = classifier
        stages: list[dict[str, str]] = []
        if isinstance(classifier, ChainedClassifier):
            base = classifier._base
            stages = [
                {
                    "stage_id": stage.stage_id,
                    "classifier_class": stage.classifier.__class__.__name__,
                }
                for stage in classifier._stages
            ]
        return {
            "requested_backend": self._settings.classifier_backend,
            "active_classifier_class": classifier.__class__.__name__,
            "base_classifier_class": base.__class__.__name__,
            "chain_stages": stages,
        }

    def _path_writable_check(self, name: str, path: Path) -> dict[str, Any]:
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path, prefix="diag-", delete=True):
                pass
            return {"name": name, "ok": True, "detail": str(path)}
        except OSError as exc:
            return {"name": name, "ok": False, "detail": f"{path}: {type(exc).__name__}: {exc}"}
