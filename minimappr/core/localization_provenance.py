"""Site coordinate-frame identity and the stored per-solve localization record.

Two jobs, both about naming *which* solve and *which* frame a stored position
belongs to:

- ``site_frame_id_for``: the one stable id for Python's site ENU frame. The
  geometry push (``core/sidecar_site_geometry.py``) sends it as
  ``site_frame_id``; the sidecar echoes it as ``coordinate_frame.frame_id``;
  the render handoff accepts a solve's positions only when the echo equals the
  id computed here. Same origin + projection always gives the same id, in any
  process, so a re-anchored origin or a changed projection can never be
  mistaken for the frame a solve was computed in.
- ``stored_localization_solve_record`` / ``ManifestLocalizationProvenanceRecorder``:
  the ``feature_summary["localization_solve"]`` block on a stored detection
  (field names pinned by ``tests/test_mixed_cluster_acceptance_e2e.py``).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Protocol

from minimappr.api.rust_localization_solve_contract import CoordinateProjection, RustLocalizationSolve

if TYPE_CHECKING:
    from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
    from minimappr.core.fusion_stage_carriers import LocalizedCandidate
    from minimappr.core.geo import LocalCoordinateFrame
    from minimappr.core.localized_render_handoff_hooks import HandoffSolveGeometry

SITE_FRAME_ID_PREFIX = "site-"
# 16 hex chars (64 bits) keeps ids short while making a collision between two
# distinct origins practically impossible.
SITE_FRAME_ID_DIGEST_HEX_CHARS = 16
# Canonical rounding before hashing: 1e-9 deg is ~0.1 mm on the ground and
# 1 mm in altitude, far below any meaningful origin change, so float noise from
# a database round trip cannot change the id.
SITE_FRAME_ID_DEGREE_DECIMALS = 9
SITE_FRAME_ID_ALTITUDE_DECIMALS = 3
SITE_FRAME_ID_CANONICAL_SEPARATOR = "|"

# LocalCoordinateFrame.mode -> contract projection (contract doc section 3.4).
CONTRACT_PROJECTION_BY_COORDINATE_FRAME_MODE: dict[str, CoordinateProjection] = {
    "flat": CoordinateProjection.FLAT_WGS84_SERIES,
    "geodetic": CoordinateProjection.GEODETIC_ECEF_ENU,
}

LOCALIZATION_SOLVE_FEATURE_KEY = "localization_solve"
STORED_PAIR_KEY_SEPARATOR = "|"
STORED_POSITION_SOURCE_SOLVE = "solve"
STORED_POSITION_SOURCE_REPORTING_NODE = "reporting_node"


class GeodeticOriginLike(Protocol):
    lat: float
    lon: float
    alt_m: float


def _canonical_rounded(value: float, decimals: int) -> str:
    # ``+ 0.0`` folds -0.0 into 0.0 so both spell the same id.
    return f"{round(float(value), decimals) + 0.0:.{decimals}f}"


def site_frame_id_for(origin: GeodeticOriginLike, projection: str) -> str:
    """Stable id of the site ENU frame anchored at ``origin`` under ``projection``.

    ``projection`` is a contract spelling (``flat_wgs84_series`` or
    ``geodetic_ecef_enu``); ``contract_projection_for_coordinate_frame_mode``
    maps a ``LocalCoordinateFrame.mode`` onto it.
    """
    contract_projection = CoordinateProjection(projection)
    canonical = SITE_FRAME_ID_CANONICAL_SEPARATOR.join(
        (
            contract_projection.value,
            _canonical_rounded(origin.lat, SITE_FRAME_ID_DEGREE_DECIMALS),
            _canonical_rounded(origin.lon, SITE_FRAME_ID_DEGREE_DECIMALS),
            _canonical_rounded(origin.alt_m, SITE_FRAME_ID_ALTITUDE_DECIMALS),
        )
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{SITE_FRAME_ID_PREFIX}{digest[:SITE_FRAME_ID_DIGEST_HEX_CHARS]}"


def contract_projection_for_coordinate_frame_mode(mode: str) -> CoordinateProjection:
    try:
        return CONTRACT_PROJECTION_BY_COORDINATE_FRAME_MODE[mode.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"LocalCoordinateFrame mode {mode!r} has no contract projection") from exc


def site_frame_id_for_coordinate_frame(coordinate_frame: "LocalCoordinateFrame") -> str:
    return site_frame_id_for(
        coordinate_frame.origin,
        contract_projection_for_coordinate_frame_mode(coordinate_frame.mode),
    )


def stored_localization_solve_record(
    solve: RustLocalizationSolve,
    *,
    stored_position_source: str,
) -> dict[str, Any]:
    """The ``feature_summary["localization_solve"]`` block for one solve.

    Lags are re-keyed to ``"<a>|<b>"`` with ``a < b`` and valued
    ``arrival(b) - arrival(a)``; Rust ``lag_seconds`` is
    ``arrival(ch_a) - arrival(ch_b)``, so a pair already in sorted order is
    negated and a reversed pair is kept as is.
    """
    provenance = solve.provenance
    sensors = provenance.sensors
    contributing_indices = solve.contributing_sensor_indices()
    pair_arrival_lag_seconds_by_sensor_pair: dict[str, float] = {}
    cross_node_pair_count = 0
    for pair in solve.admitted_pair_tdoas:
        sensor_a, sensor_b = sensors[pair.ch_a], sensors[pair.ch_b]
        if sensor_a.node_id != sensor_b.node_id:
            cross_node_pair_count += 1
        if sensor_a.sensor_id < sensor_b.sensor_id:
            key = f"{sensor_a.sensor_id}{STORED_PAIR_KEY_SEPARATOR}{sensor_b.sensor_id}"
            pair_arrival_lag_seconds_by_sensor_pair[key] = -pair.lag_seconds
        else:
            key = f"{sensor_b.sensor_id}{STORED_PAIR_KEY_SEPARATOR}{sensor_a.sensor_id}"
            pair_arrival_lag_seconds_by_sensor_pair[key] = pair.lag_seconds
    return {
        "solve_id": provenance.solve_id,
        "attempt_kind": provenance.attempt_kind.value,
        "buffer_scope": provenance.buffer_scope.value,
        "cluster_id": provenance.cluster_id,
        "reporting_node_id": provenance.reporting_node_id,
        "contributing_sensor_ids": [sensors[index].sensor_id for index in contributing_indices],
        "contributing_node_ids": sorted({sensors[index].node_id for index in contributing_indices}),
        "admitted_pair_count": len(solve.admitted_pair_tdoas),
        "cross_node_pair_count": cross_node_pair_count,
        "pair_arrival_lag_seconds_by_sensor_pair": pair_arrival_lag_seconds_by_sensor_pair,
        "coordinate_frame_id": provenance.coordinate_frame.frame_id,
        "coordinate_frame_kind": provenance.coordinate_frame.kind.value,
        "layout_revision": provenance.layout_revision,
        "geometry_push_revision": provenance.geometry_push_revision,
        "window_center_time_ns": provenance.window_center_time_ns,
        "stored_position_source": stored_position_source,
    }


class ManifestLocalizationProvenanceRecorder:
    """Seam implementation: stamps the solve record onto the localized product.

    Recorded for omni outcomes too (``stored_position_source`` says the stored
    position is then the reporting node's), so a degraded detection still says
    which solve was rejected.
    """

    def record_handoff_provenance(
        self,
        *,
        payload: "LocalizedClassifierRenderRequest",
        geometry: "HandoffSolveGeometry",
        localized_product: "LocalizedCandidate",
        reporting_modality: str,
    ) -> None:
        del geometry
        if payload.localization_solve is None:
            return
        localized_product.extra_classification_features[LOCALIZATION_SOLVE_FEATURE_KEY] = (
            stored_localization_solve_record(
                payload.localization_solve,
                stored_position_source=(
                    STORED_POSITION_SOURCE_SOLVE
                    if reporting_modality == "localized"
                    else STORED_POSITION_SOURCE_REPORTING_NODE
                ),
            )
        )
