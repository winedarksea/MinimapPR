"""FusionNode seam for onset-SRP clustering (flag `localization_onset_clustering_enabled`).

Replaces the legacy single 80 ms window solve for a trigger with:

* an analysis span covering every node, [event - pre, event + post + max
  inter-node delay], waited for with a non-blocking poll (later-arriving nodes
  are no longer dropped because the trigger node's mics covered first);
* `localize_onset_clusters` on a worker thread (off the event loop);
* one LocalizedCandidate per accepted cluster, whose classification window
  starts at that cluster's own onset on its nearest node (the legacy trailing
  window ends at the trigger and held almost none of a short call);
* a short memory of consumed onsets so overlapping spans from later triggers do
  not surface the same note twice.

Audio stays in the in-memory ring buffers; nothing is written to disk.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from minimappr.config import OnsetClusterAttributionConfig, OnsetSrpClusteringConfig
from minimappr.core.localization import speed_of_sound_mps
from minimappr.core.onset_cluster_attribution import ClusterAttributionContext, is_band_distinct_persistence_hold
from minimappr.core.onset_cluster_echo_checks import SAME_BAND_MIN_OVERLAP_FRACTION
from minimappr.core.onset_classification_node import classification_capable_cluster
from minimappr.core.onset_srp_clustering import localize_onset_clusters
from minimappr.core.onset_srp_grid import spatial_cluster_radius_m
from minimappr.core.onset_srp_types import (
    CLUSTER_STATUS_ACCEPTED,
    CLUSTER_STATUS_ECHO,
    CLUSTER_STATUS_TENTATIVE,
    LocalizedOnsetCluster,
    OnsetClusteringResult,
)
from minimappr.core.sensor_span_coverage_wait import await_sensor_span_coverage
from minimappr.models import LocalizationResult

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from minimappr.core.fusion_node import EventCandidate, FusionNode, LocalizedCandidate

NANOSECONDS_PER_SECOND = 1_000_000_000
BUFFER_POLL_INTERVAL_SECONDS = 0.01
LOCALIZATION_METHOD_NAME = "onset_srp"
MIN_SENSORS_FOR_FULL_3D_TIER = 4
# The span wait may use at most this share of the localization stage timeout,
# leaving the rest for clustering before asyncio.wait_for cancels the stage.
SPAN_WAIT_STAGE_TIMEOUT_FRACTION = 0.5


class OnsetSrpLocalizationStage:
    def __init__(
        self, config: OnsetSrpClusteringConfig, attribution_config: OnsetClusterAttributionConfig | None = None
    ) -> None:
        self.config = config
        # Field Mode W3-attribution: None or disabled keeps the W2 behaviour exactly.
        self.attribution_config = attribution_config if attribution_config and attribution_config.enabled else None
        self._consumed_onset_ns_by_node: dict[str, deque[int]] = {}

    # ------------------------------------------------------------------ memory
    def _consumed_samples(self, span_start_ns: int, sample_rate_hz: int) -> dict[str, tuple[int, ...]]:
        return {
            node: tuple(int(round((t - span_start_ns) * sample_rate_hz / NANOSECONDS_PER_SECOND)) for t in times)
            for node, times in self._consumed_onset_ns_by_node.items()
        }

    def _remember(self, result: OnsetClusteringResult, span_start_ns: int, sample_rate_hz: int, now_ns: int) -> None:
        ttl_ns = int(self.config.dedupe_ttl_seconds * NANOSECONDS_PER_SECOND)
        for node, samples in result.consumed_onset_samples_by_node.items():
            remembered = self._consumed_onset_ns_by_node.setdefault(node, deque())
            remembered.extend(span_start_ns + int(s * NANOSECONDS_PER_SECOND / sample_rate_hz) for s in samples)
        for node in list(self._consumed_onset_ns_by_node):
            remembered = self._consumed_onset_ns_by_node[node]
            while remembered and remembered[0] < now_ns - ttl_ns:
                remembered.popleft()
            # Drop idle nodes too, so removed/renamed nodes do not linger forever.
            if not remembered:
                del self._consumed_onset_ns_by_node[node]

    # ------------------------------------------------------------------ buffer
    async def _await_span(self, fusion: FusionNode, sensor_ids: list[str], start_ns: int, end_ns: int) -> list[str]:
        """Poll until every sensor covers the span; returns the sensors that do.

        The budget is a *stall* budget: waiting continues while some uncovered
        sensor's buffer is still advancing and stops after
        `buffer_wait_max_seconds` without progress. A fixed wall-clock deadline
        dropped whole windows ("onset_span_uncovered") whenever ingest ran slower
        than real time (CPU-starved replay), making results load-dependent. A
        stalled node still releases the wait after the budget, and the total is
        capped below the localization stage timeout.
        """
        stall_budget_s = max(0.0, self.config.buffer_wait_max_seconds)
        result = await await_sensor_span_coverage(
            buffer=fusion.buffer,
            sensor_ids=sensor_ids,
            span_start_ns=start_ns,
            span_end_ns=end_ns,
            stall_budget_seconds=stall_budget_s,
            total_cap_seconds=max(
                stall_budget_s, SPAN_WAIT_STAGE_TIMEOUT_FRACTION * fusion.classifier_config.stage_timeout_seconds
            ),
            poll_interval_seconds=BUFFER_POLL_INTERVAL_SECONDS,
        )
        return result.covered_sensor_ids

    # ------------------------------------------------------------------- stage
    async def localize_candidate(self, fusion: FusionNode, candidate: EventCandidate) -> list[LocalizedCandidate]:
        sensor_positions = await fusion.registry.sensor_positions()
        sensor_ids = sorted(sensor_positions)
        if not sensor_ids:
            fusion._record_silent_drop(stage="localization", reason="no_sensors", candidate_id=candidate.id)
            return []
        node_ids = await fusion.registry.sensor_node_ids(sensor_ids)
        conditions = fusion.environment_provider.get_conditions(
            tuple(float(v) for v in np.mean(np.vstack(list(sensor_positions.values())), axis=0))
        )
        sound_speed = speed_of_sound_mps(conditions.temperature_c, conditions.humidity_fraction)
        points = np.vstack([sensor_positions[s] for s in sensor_ids])
        max_delay_s = float(np.max(np.linalg.norm(points[:, None] - points[None, :], axis=2))) / sound_speed
        start_ns = candidate.event_time_ns - int(self.config.analysis_pre_trigger_seconds * NANOSECONDS_PER_SECOND)
        end_ns = candidate.event_time_ns + int(
            (self.config.analysis_post_trigger_seconds + max_delay_s) * NANOSECONDS_PER_SECOND
        )
        covered = await self._await_span(fusion, sensor_ids, start_ns, end_ns)
        if not covered:
            fusion._record_silent_drop(stage="localization", reason="onset_span_uncovered", candidate_id=candidate.id)
            return []
        window_seconds = (end_ns - start_ns) / NANOSECONDS_PER_SECOND
        windows = await fusion.buffer.get_synchronized_window(
            sensor_ids=covered,
            center_time_ns=(start_ns + end_ns) // 2,
            window_seconds=window_seconds,
            sample_rate_hz=candidate.sample_rate_hz,
        )
        if not windows:
            fusion._record_silent_drop(stage="localization", reason="onset_span_uncovered", candidate_id=candidate.id)
            return []
        length = min(window.size for window in windows.values())
        signals = {sid: np.asarray(window[:length], dtype=np.float64) for sid, window in windows.items()}
        result = await asyncio.to_thread(
            localize_onset_clusters,
            sensor_signals=signals,
            sensor_positions={sid: sensor_positions[sid] for sid in signals},
            sensor_node_ids={sid: node_ids.get(sid, sid) for sid in signals},
            sample_rate_hz=candidate.sample_rate_hz,
            sound_speed_mps=sound_speed,
            config=self.config,
            interpolation_factor=fusion.localization_config.gcc_phat_interp_factor,
            far_field_default_range_m=fusion.localization_config.localization_far_field_default_range_m,
            consumed_onset_samples_by_node=self._consumed_samples(start_ns, candidate.sample_rate_hz),
            attribution_audible_snr_db=(
                self.attribution_config.audible_snr_db if self.attribution_config is not None else None
            ),
        )
        self._remember(result, start_ns, candidate.sample_rate_hz, candidate.event_time_ns)
        metrics = fusion._metrics
        metrics.onset_clustering_window_count += 1
        for cluster in result.clusters:
            if cluster.status == CLUSTER_STATUS_ACCEPTED:
                metrics.onset_clustering_accepted_cluster_count += 1
            elif cluster.status == CLUSTER_STATUS_TENTATIVE:
                metrics.onset_clustering_tentative_cluster_count += 1
            elif cluster.status == CLUSTER_STATUS_ECHO:
                metrics.onset_clustering_echo_cluster_count += 1
        products = []
        environment_summary = {
            "temperature_c": conditions.temperature_c,
            "humidity_fraction": conditions.humidity_fraction,
            "pressure_pa": conditions.pressure_pa,
            "wind_speed_mps": conditions.wind_speed_mps,
            "wind_dir_deg": conditions.wind_dir_deg,
            "metadata": dict(conditions.metadata or {}),
        }
        accepted = [(i, c) for i, c in enumerate(result.clusters) if c.status == CLUSTER_STATUS_ACCEPTED]
        accepted += self._band_distinct_tentative_clusters(result, accepted, metrics)
        min_classification_sensors = fusion.localization_config.beamformed_classification_min_sensor_count
        window_clusters = list(result.clusters)
        for index, (window_cluster_index, cluster) in enumerate(accepted):
            cluster = classification_capable_cluster(cluster, node_ids, sensor_positions, min_classification_sensors)
            window_clusters[window_cluster_index] = cluster
            product = await self._product_for_cluster(
                fusion,
                candidate=candidate,
                cluster=cluster,
                cluster_index=index,
                span_start_ns=start_ns,
                span_end_ns=end_ns,
                node_ids=node_ids,
                sensor_positions=sensor_positions,
                environment_summary=environment_summary,
            )
            if product is not None:
                self._attach_attribution_context(
                    product, candidate.id, window_clusters, window_cluster_index, np.mean(points, axis=0),
                    start_ns, end_ns, fusion.localization_config.classification_window_seconds,
                )
                products.append(product)
        self._attach_sibling_count(products)
        return products

    @staticmethod
    def _attach_sibling_count(products: list[LocalizedCandidate]) -> None:
        """Closeout B2: how many classified siblings window label reconciliation waits for."""
        siblings = [p for p in products if p.onset_attribution_context is not None]
        for product in siblings:
            product.onset_attribution_context = dataclasses.replace(
                product.onset_attribution_context, sibling_product_count=len(siblings)
            )

    def _band_distinct_tentative_clusters(self, result, accepted, metrics) -> list[tuple[int, LocalizedOnsetCluster]]:
        """Field Mode W3-attribution: persistence-held single-array clusters that cannot be an echo."""
        if self.attribution_config is None or not self.attribution_config.promote_band_distinct_tentative:
            return []
        accepted_bands = [(cluster.band_min_hz, cluster.band_max_hz) for _, cluster in accepted]
        promoted = [
            (index, cluster)
            for index, cluster in enumerate(result.clusters)
            if cluster.status == CLUSTER_STATUS_TENTATIVE
            and is_band_distinct_persistence_hold(
                status_reasons=cluster.status_reasons,
                band_hz=(cluster.band_min_hz, cluster.band_max_hz),
                accepted_bands_hz=accepted_bands,
                same_band_min_overlap_fraction=SAME_BAND_MIN_OVERLAP_FRACTION,
            )
        ]
        metrics.onset_attribution_promoted_tentative_count += len(promoted)
        return promoted

    def _classification_end_ns(self, onset_ns: int, span_end_ns: int, classification_seconds: float) -> int:
        pre_roll_ns = int(self.config.frame_pre_roll_seconds * NANOSECONDS_PER_SECOND)
        return min(onset_ns - pre_roll_ns + int(classification_seconds * NANOSECONDS_PER_SECOND), span_end_ns)

    def _colocated(self, cluster_a, cluster_b, centroid_m) -> bool:
        """Within one clustering radius: spatially the same source (a split or a duplicate),
        so masking one would also erase the other; never treated as an interferer."""
        position_a = np.asarray(cluster_a.position_m)
        position_b = np.asarray(cluster_b.position_m)
        range_m = min(float(np.linalg.norm(position_a - centroid_m)), float(np.linalg.norm(position_b - centroid_m)))
        radius_m = spatial_cluster_radius_m(range_m, self.config.cluster_radius_m, self.config.cluster_range_fraction)
        return float(np.linalg.norm(position_a - position_b)) <= radius_m

    def _attach_attribution_context(
        self, product, window_id, clusters, window_cluster_index, centroid_m, span_start_ns, span_end_ns,
        classification_seconds,
    ) -> None:
        """Field Mode W3-attribution: what the classifier worker needs to attribute labels."""
        cluster = clusters[window_cluster_index]
        if self.attribution_config is None or cluster.attribution_evidence is None:
            return
        window_evidence = {
            index: other.attribution_evidence
            for index, other in enumerate(clusters)
            if other.attribution_evidence is not None
            and (
                index == window_cluster_index
                or (other.status != CLUSTER_STATUS_ECHO and not self._colocated(other, cluster, centroid_m))
            )
        }
        sample_rate_hz = product.candidate.sample_rate_hz
        classification_signal_samples = next(iter(product.classification_selected_windows.values())).size
        end_ns = self._classification_end_ns(product.candidate.event_time_ns, span_end_ns, classification_seconds)
        end_sample = int(round((end_ns - span_start_ns) * sample_rate_hz / NANOSECONDS_PER_SECOND))
        product.onset_attribution_context = ClusterAttributionContext(
            window_id=window_id,
            target_index=window_cluster_index,
            cluster_evidence=window_evidence,
            classifying_node_id=cluster.loudest_node_id,
            classification_start_sample=end_sample - classification_signal_samples,
            classification_end_sample=end_sample,
            sample_rate_hz=sample_rate_hz,
            shared_onset_max_lag_samples=int(round(self.config.echo_max_lag_seconds * sample_rate_hz)),
        )
        # Read by partial binding (assembly_track_resolution) as measured per-node levels.
        product.extra_classification_features["onset_cluster_node_band_snr_db"] = {
            node: round(snr, 2) for node, snr in cluster.attribution_evidence.node_band_snr_db.items()
        }
        product.extra_classification_features["onset_cluster_audible_snr_db"] = self.attribution_config.audible_snr_db

    async def _product_for_cluster(
        self,
        fusion: FusionNode,
        *,
        candidate: EventCandidate,
        cluster: LocalizedOnsetCluster,
        cluster_index: int,
        span_start_ns: int,
        span_end_ns: int,
        node_ids: dict[str, str],
        sensor_positions: dict[str, np.ndarray],
        environment_summary: dict[str, Any],
    ) -> LocalizedCandidate | None:
        sample_rate_hz = candidate.sample_rate_hz
        onset_ns = span_start_ns + int(min(cluster.onset_samples) * NANOSECONDS_PER_SECOND / sample_rate_hz)
        node_sensors = sorted(sid for sid, node in node_ids.items() if node == cluster.loudest_node_id)
        classification_seconds = fusion.localization_config.classification_window_seconds
        classification_end_ns = self._classification_end_ns(onset_ns, span_end_ns, classification_seconds)
        windows = await fusion.buffer.get_synchronized_window_ending_at(
            sensor_ids=node_sensors,
            end_time_ns=classification_end_ns,
            window_seconds=classification_seconds,
            sample_rate_hz=sample_rate_hz,
        )
        if not windows:
            fusion._record_silent_drop(
                stage="localization", reason="onset_classification_window_missing", candidate_id=candidate.id
            )
            return None
        selected_ids = sorted(windows)
        reference_sensor = cluster.reference_sensor if cluster.reference_sensor in windows else selected_ids[0]
        cluster_candidate = dataclasses.replace(
            candidate, id=f"{candidate.id}:onset{cluster_index}", event_time_ns=onset_ns
        )
        energies = {sid: float(np.sqrt(np.mean(np.square(windows[sid])))) for sid in selected_ids}
        product = fusion._build_reference_sensor_candidate(
            candidate=cluster_candidate,
            energies=energies,
            sensor_positions=sensor_positions,
            selected_ids=selected_ids,
            selected_windows=windows,
            classification_windows=windows,
            environment_summary=environment_summary,
        )
        gdop = cluster.gdop if np.isfinite(cluster.gdop) else float("inf")
        localization = LocalizationResult(
            position_m=cluster.position_m,
            confidence=float(np.clip(cluster.confidence, 0.0, 1.0)),
            gdop=max(gdop, 0.0),
            reference_sensor=reference_sensor,
            tdoa_s=cluster.tdoa_s,
            position_covariance_m2=cluster.covariance_m2,
            residual_rms_seconds=cluster.residual_rms_seconds,
            range_projection_mode=cluster.range_projection_mode,
            range_observability=cluster.range_observability,
            attempted_algorithm=LOCALIZATION_METHOD_NAME,
            resolved_algorithm=LOCALIZATION_METHOD_NAME,
        )
        centroid = np.mean(np.vstack([sensor_positions[s] for s in selected_ids]), axis=0)
        branch = fusion._build_localization_branch(
            localization=localization,
            selected_windows=windows,
            classification_windows=windows,
            capability_tier="full_3d" if len(selected_ids) >= MIN_SENSORS_FOR_FULL_3D_TIER else "2d",
            contributing_centroid_m=centroid,
        )
        if branch is None:
            return None
        branch.localization_method = LOCALIZATION_METHOD_NAME
        product.localization_branch = branch
        product.extra_classification_features = {
            "onset_cluster_status": cluster.status,
            "onset_cluster_band_hz": [cluster.band_min_hz, cluster.band_max_hz],
            "onset_cluster_frame_count": cluster.frame_count,
            "onset_cluster_independent_onsets": cluster.independent_onset_count,
            "onset_cluster_contributing_nodes": list(cluster.contributing_node_ids),
            "onset_cluster_refined_by_least_squares": cluster.refined_by_least_squares,
            "onset_cluster_envelope_fraction": cluster.used_envelope_fraction,
        }
        return product
