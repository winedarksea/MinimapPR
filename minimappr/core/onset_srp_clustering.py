"""Onset-weighted short-frame SRP with spatial clustering (Field Mode server track S2).

Pipeline for one analysis window (all nodes on the shared sample timeline):

1. per node: sub-band flux history -> note onsets (noise-floor relative);
2. per onset: template on the node's loudest channel vs delay-compensated search
   windows on every other sensor -> band-limited soft-SNR GCC-PHAT curves
   (envelope correlation for narrowband frames);
3. per frame: SRP over a direction grid (single compact array) or a polar
   spatial grid (multi-node) -> a few peaks, weighted by onset strength, peak
   height and map contrast;
4. greedy weighted clustering of all frame peaks -> candidate birds;
5. echo / persistence checks, per-cluster LM refine, multi-node level check.

Pure and synchronous: callers run it off the event loop (asyncio.to_thread).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from minimappr.config import OnsetSrpClusteringConfig
from minimappr.core.far_field_model_selection import COMPACT_ARRAY_MAX_APERTURE_M
from minimappr.core.onset_cluster_attribution_evidence import build_cluster_attribution_evidence
from minimappr.core.onset_cluster_echo_checks import (
    apply_onset_echo_and_persistence_checks,
    level_consistency_holds,
)
from minimappr.core.onset_cluster_refinement import refine_cluster_position
from minimappr.core.onset_detection import SubbandHistory, build_subband_history, detect_onsets
from minimappr.core.onset_frame_correlation import (
    SensorNoiseSpectrum,
    correlate_template_against_search,
    estimate_noise_spectrum,
)
from minimappr.core.onset_srp_grid import (
    CandidateGrid,
    LagCache,
    angular_separation_deg,
    build_direction_grid,
    build_spatial_grid,
    pick_frame_peaks,
    spatial_cluster_radius_m,
    steered_response_power,
)
from minimappr.core.onset_srp_types import (
    CLUSTER_STATUS_ACCEPTED,
    CLUSTER_STATUS_ECHO,
    CLUSTER_STATUS_TENTATIVE,
    FramePeak,
    LocalizedOnsetCluster,
    OnsetCluster,
    OnsetClusteringResult,
    OnsetFrame,
)

# Extra lag (samples) beyond the geometric maximum when sizing search windows.
SEARCH_LAG_MARGIN_SAMPLES = 2


def _group_sensors_by_node(sensor_node_ids: dict[str, str], sensor_ids: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for sensor_id in sensor_ids:
        groups[sensor_node_ids.get(sensor_id, sensor_id)].append(sensor_id)
    return {node: sorted(members) for node, members in groups.items()}


def _array_aperture_m(sensor_positions: dict[str, np.ndarray], sensor_ids: list[str]) -> float:
    points = np.vstack([sensor_positions[sensor_id] for sensor_id in sensor_ids])
    return float(np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)))


def _build_frame(
    *,
    frame_index: int,
    node_id: str,
    node_sensors: list[str],
    onset,
    signals: dict[str, np.ndarray],
    sensor_positions: dict[str, np.ndarray],
    noise_spectra: dict[str, SensorNoiseSpectrum],
    sample_rate_hz: int,
    sound_speed_mps: float,
    config: OnsetSrpClusteringConfig,
    interpolation_factor: int,
) -> OnsetFrame | None:
    frame_samples = int(round(config.frame_seconds * sample_rate_hz))
    start = onset.sample_index - int(round(config.frame_pre_roll_seconds * sample_rate_hz))
    total = next(iter(signals.values())).size
    if start < 0 or start + frame_samples > total:
        return None
    reference_sensor = max(node_sensors, key=lambda sid: float(np.sum(signals[sid][start : start + frame_samples] ** 2)))
    template = signals[reference_sensor][start : start + frame_samples]
    curves = []
    for sensor_id in sorted(signals):
        if sensor_id == reference_sensor:
            continue
        baseline_m = float(np.linalg.norm(sensor_positions[sensor_id] - sensor_positions[reference_sensor]))
        max_lag = int(np.ceil(baseline_m / sound_speed_mps * sample_rate_hz)) + SEARCH_LAG_MARGIN_SAMPLES
        if start - max_lag < 0 or start + frame_samples + max_lag > total:
            continue
        curve = correlate_template_against_search(
            sensor_a=reference_sensor,
            sensor_b=sensor_id,
            template=template,
            search=signals[sensor_id][start - max_lag : start + frame_samples + max_lag],
            max_lag_samples=max_lag,
            sample_rate_hz=sample_rate_hz,
            band_min_hz=onset.band_min_hz,
            band_max_hz=onset.band_max_hz,
            template_noise=noise_spectra[reference_sensor],
            search_noise=noise_spectra[sensor_id],
            interpolation_factor=interpolation_factor,
            narrowband_bandwidth_hz=config.narrowband_bandwidth_hz,
            allow_envelope=baseline_m > COMPACT_ARRAY_MAX_APERTURE_M,
        )
        if curve is not None:
            curves.append(curve)
    if not curves:
        return None
    return OnsetFrame(
        frame_index=frame_index,
        template_node_id=node_id,
        reference_sensor=reference_sensor,
        onset_sample=onset.sample_index,
        strength=onset.strength,
        band_min_hz=onset.band_min_hz,
        band_max_hz=onset.band_max_hz,
        curves=curves,
    )


def _is_consumed(sample: int, consumed: tuple[int, ...], tolerance_samples: int) -> bool:
    return any(abs(sample - other) <= tolerance_samples for other in consumed)


def _peak_belongs(peak: FramePeak, cluster: OnsetCluster, grid: CandidateGrid, config) -> bool:
    if grid.is_direction_grid or peak.range_m >= grid.ranges_m.max():
        return angular_separation_deg(peak.direction, cluster.weighted_direction()) <= config.cluster_angle_deg
    centroid = cluster.weighted_position_m()
    radius = spatial_cluster_radius_m(
        min(peak.range_m, float(np.linalg.norm(centroid - grid.centroid_m))),
        config.cluster_radius_m,
        config.cluster_range_fraction,
    )
    return float(np.linalg.norm(peak.position_m - centroid)) <= radius


def cluster_frame_peaks(peaks: list[FramePeak], grid: CandidateGrid, config) -> list[OnsetCluster]:
    """Greedy weighted clustering, heaviest peak first; strongest cluster first."""
    clusters: list[OnsetCluster] = []
    for peak in sorted(peaks, key=lambda item: -item.weight):
        home = next((cluster for cluster in clusters if _peak_belongs(peak, cluster, grid, config)), None)
        if home is None:
            clusters.append(OnsetCluster(peaks=[peak]))
        else:
            home.peaks.append(peak)
    clusters.sort(key=lambda cluster: -cluster.weight)
    if not clusters:
        return []
    floor = config.cluster_min_weight_fraction * clusters[0].weight
    return [cluster for cluster in clusters if cluster.weight >= floor][: config.cluster_max_count]


def localize_onset_clusters(
    *,
    sensor_signals: dict[str, np.ndarray],
    sensor_positions: dict[str, np.ndarray],
    sensor_node_ids: dict[str, str],
    sample_rate_hz: int,
    sound_speed_mps: float,
    config: OnsetSrpClusteringConfig,
    interpolation_factor: int,
    far_field_default_range_m: float,
    consumed_onset_samples_by_node: dict[str, tuple[int, ...]] | None = None,
    attribution_audible_snr_db: float | None = None,
) -> OnsetClusteringResult:
    """`attribution_audible_snr_db` set (W3-attribution on) attaches per-node level evidence to every cluster."""
    sensor_ids = sorted(sensor_signals)
    signals = {sid: np.asarray(sensor_signals[sid], dtype=np.float64) for sid in sensor_ids}
    positions = {sid: np.asarray(sensor_positions[sid], dtype=np.float64) for sid in sensor_ids}
    nodes = _group_sensors_by_node(sensor_node_ids, sensor_ids)
    consumed = consumed_onset_samples_by_node or {}
    frame_samples = int(round(config.frame_seconds * sample_rate_hz))
    tolerance = int(round(config.min_separation_seconds * sample_rate_hz))

    histories: dict[str, SubbandHistory] = {}
    frames: dict[int, OnsetFrame] = {}
    noise_spectra = {
        sid: estimate_noise_spectrum(signals[sid], sample_rate_hz=sample_rate_hz, segment_samples=frame_samples)
        for sid in sensor_ids
    }
    onset_count = 0
    for node_id, node_sensors in sorted(nodes.items()):
        node_mix = np.mean(np.vstack([signals[sid] for sid in node_sensors]), axis=0)
        histories[node_id] = build_subband_history(
            node_mix,
            sample_rate_hz=sample_rate_hz,
            hop_seconds=config.hop_seconds,
            band_min_hz=config.band_min_hz,
            band_max_hz=config.band_max_hz,
            band_count=config.subband_count,
        )
        onsets = detect_onsets(
            histories[node_id],
            sample_rate_hz=sample_rate_hz,
            flux_threshold_mad_multiplier=config.flux_threshold_mad_multiplier,
            min_separation_seconds=config.min_separation_seconds,
            max_onsets=config.max_onsets_per_window,
            active_band_snr_db=config.active_band_snr_db,
            frame_seconds=config.frame_seconds,
        )
        onset_count += len(onsets)
        for onset in onsets:
            if _is_consumed(onset.sample_index, consumed.get(node_id, ()), tolerance):
                continue
            frame = _build_frame(
                frame_index=len(frames),
                node_id=node_id,
                node_sensors=node_sensors,
                onset=onset,
                signals=signals,
                sensor_positions=positions,
                noise_spectra=noise_spectra,
                sample_rate_hz=sample_rate_hz,
                sound_speed_mps=sound_speed_mps,
                config=config,
                interpolation_factor=interpolation_factor,
            )
            if frame is not None:
                frames[frame.frame_index] = frame

    consumed_now = {
        node: tuple(sorted(frame.onset_sample for frame in frames.values() if frame.template_node_id == node))
        for node in nodes
    }
    if not frames:
        return OnsetClusteringResult((), onset_count, 0, consumed_now)

    node_centroids = {node: np.mean(np.vstack([positions[s] for s in members]), axis=0) for node, members in nodes.items()}
    layout_centroid = np.mean(np.vstack(list(node_centroids.values())), axis=0)
    compact = _array_aperture_m(positions, sensor_ids) <= COMPACT_ARRAY_MAX_APERTURE_M
    if compact:
        grid = build_direction_grid(
            layout_centroid,
            azimuth_step_deg=config.srp_azimuth_step_deg,
            elevation_step_deg=config.srp_elevation_step_deg,
            far_range_m=far_field_default_range_m,
        )
    else:
        grid = build_spatial_grid(
            layout_centroid,
            azimuth_step_deg=config.srp_azimuth_step_deg,
            elevation_step_deg=config.srp_elevation_step_deg,
            min_range_m=config.srp_grid_min_range_m,
            max_range_m=config.srp_grid_max_range_m,
            ring_count=config.srp_grid_ring_count,
        )
    lag_cache = LagCache(grid, positions, sound_speed_mps)
    max_strength = max(frame.strength for frame in frames.values())
    peaks: list[FramePeak] = []
    for frame in frames.values():
        frame.weight = frame.strength / max_strength
        power = steered_response_power(frame.curves, lag_cache)
        for rank, grid_peak in enumerate(
            pick_frame_peaks(
                power,
                grid,
                max_peaks=config.srp_peaks_per_frame,
                min_relative_height=config.srp_peak_min_relative_height,
                suppression_radius_m=config.cluster_radius_m,
                suppression_range_fraction=config.cluster_range_fraction,
                suppression_angle_deg=config.cluster_angle_deg,
            )
        ):
            peaks.append(
                FramePeak(
                    frame_index=frame.frame_index,
                    position_m=grid.points_m[grid_peak.point_index],
                    direction=grid.directions[grid_peak.point_index],
                    range_m=float(grid.ranges_m[grid_peak.point_index]),
                    value=grid_peak.value,
                    weight=frame.weight * grid_peak.value * grid_peak.contrast,
                    is_frame_primary=rank == 0,
                )
            )

    clusters = cluster_frame_peaks(peaks, grid, config)
    apply_onset_echo_and_persistence_checks(
        clusters,
        frames,
        sample_rate_hz=sample_rate_hz,
        echo_min_lag_seconds=config.echo_min_lag_seconds,
        echo_max_lag_seconds=config.echo_max_lag_seconds,
        echo_min_independent_fraction=config.echo_min_independent_fraction,
        persistence_min_independent_onsets=config.persistence_min_independent_onsets,
        apply_persistence=grid.is_direction_grid,
    )
    minimum_std_m = sound_speed_mps / (sample_rate_hz * max(1, interpolation_factor))
    localized = [
        _finalize_cluster(
            cluster,
            frames=frames,
            grid=grid,
            positions=positions,
            nodes=nodes,
            node_centroids=node_centroids,
            histories=histories,
            sample_rate_hz=sample_rate_hz,
            sound_speed_mps=sound_speed_mps,
            config=config,
            interpolation_factor=interpolation_factor,
            far_field_default_range_m=far_field_default_range_m,
            frame_samples=frame_samples,
            minimum_std_m=minimum_std_m,
            attribution_audible_snr_db=attribution_audible_snr_db,
        )
        for cluster in clusters
    ]
    return OnsetClusteringResult(tuple(localized), onset_count, len(frames), consumed_now)


def _finalize_cluster(
    cluster: OnsetCluster,
    *,
    frames,
    grid: CandidateGrid,
    positions,
    nodes,
    node_centroids,
    histories,
    sample_rate_hz,
    sound_speed_mps,
    config,
    interpolation_factor,
    far_field_default_range_m,
    frame_samples,
    minimum_std_m,
    attribution_audible_snr_db=None,
) -> LocalizedOnsetCluster:
    bearing_only = grid.is_direction_grid or all(peak.range_m >= grid.ranges_m.max() for peak in cluster.peaks)
    seed = (
        grid.centroid_m + cluster.weighted_direction() * far_field_default_range_m
        if bearing_only
        else cluster.weighted_position_m()
    )
    refined = refine_cluster_position(
        cluster=cluster,
        frames=frames,
        sensor_positions=positions,
        seed_position_m=seed,
        centroid_m=grid.centroid_m,
        seed_is_bearing_only=bearing_only,
        sound_speed_mps=sound_speed_mps,
        sample_rate_hz=sample_rate_hz,
        interpolation_factor=interpolation_factor,
        far_field_default_range_m=far_field_default_range_m,
        minimum_position_std_m=minimum_std_m,
    )
    led = sorted(cluster.led_frame_indices) or sorted(cluster.frame_indices)
    led_frames = [frames[index] for index in led]
    if (
        cluster.status == CLUSTER_STATUS_ACCEPTED
        and len(nodes) > 1
        and not level_consistency_holds(
            position_m=refined.position_m,
            led_frames=led_frames,
            node_histories=histories,
            node_positions_m=node_centroids,
            sound_speed_mps=sound_speed_mps,
            sample_rate_hz=sample_rate_hz,
            frame_samples=frame_samples,
            margin_db=config.level_consistency_margin_db,
        )
    ):
        cluster.status = CLUSTER_STATUS_TENTATIVE
        cluster.status_reasons.append("level_inconsistent")
    contributing = sorted({frame.template_node_id for frame in led_frames})
    loudest = _classification_node_for_cluster(led_frames, node_centroids, refined.position_m, bearing_only)
    attribution_evidence = (
        None
        if attribution_audible_snr_db is None
        else build_cluster_attribution_evidence(
            position_m=refined.position_m,
            led_frames=led_frames,
            node_histories=histories,
            node_positions_m=node_centroids,
            sound_speed_mps=sound_speed_mps,
            sample_rate_hz=sample_rate_hz,
            frame_samples=frame_samples,
            audible_snr_db=attribution_audible_snr_db,
        )
    )
    envelope_curves = [curve.used_envelope for index in cluster.frame_indices for curve in frames[index].curves]
    return LocalizedOnsetCluster(
        position_m=tuple(float(v) for v in refined.position_m),
        covariance_m2=refined.covariance_m2,
        confidence=refined.confidence,
        gdop=refined.gdop,
        range_projection_mode=refined.range_projection_mode,
        range_observability=refined.range_observability,
        residual_rms_seconds=refined.residual_rms_seconds,
        reference_sensor=refined.reference_sensor,
        tdoa_s=refined.tdoa_s,
        status=cluster.status,
        status_reasons=tuple(cluster.status_reasons),
        onset_samples=tuple(frames[index].onset_sample for index in led),
        template_node_ids=tuple(frames[index].template_node_id for index in led),
        band_min_hz=float(min(frame.band_min_hz for frame in led_frames)),
        band_max_hz=float(max(frame.band_max_hz for frame in led_frames)),
        weight=cluster.weight,
        frame_count=len(cluster.frame_indices),
        independent_onset_count=cluster.independent_onset_count,
        refined_by_least_squares=refined.refined_by_least_squares,
        used_envelope_fraction=float(np.mean(envelope_curves)) if envelope_curves else 0.0,
        contributing_node_ids=tuple(contributing),
        loudest_node_id=loudest,
        attribution_evidence=attribution_evidence,
    )


def _classification_node_for_cluster(led_frames, node_centroids, position_m, bearing_only) -> str:
    """Node to classify from: nearest the fix when range is known, else the most
    frequent template node (the one whose onsets led the cluster)."""
    if not bearing_only and len(node_centroids) > 1:
        return min(node_centroids, key=lambda node: float(np.linalg.norm(node_centroids[node] - position_m)))
    counts: dict[str, int] = defaultdict(int)
    for frame in led_frames:
        counts[frame.template_node_id] += 1
    return max(counts, key=counts.get)


__all__ = [
    "CLUSTER_STATUS_ACCEPTED",
    "CLUSTER_STATUS_ECHO",
    "CLUSTER_STATUS_TENTATIVE",
    "cluster_frame_peaks",
    "localize_onset_clusters",
]
