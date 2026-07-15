"""Pipeline Flow DAG builder — read-only structural + status graph.

Turns the running configuration (Settings, registered nodes, classifier
routing, rules) plus live status (FusionMetrics / sidecar DSP status) into a
:class:`PipelineGraph` for the ``/settings/pipeline`` view.

Every new pipeline stage or routing concept MUST be registered here and in
``config_groups.py``; ``tests/test_config_structured.py`` enforces config-key
coverage so the DAG and the Settings surface cannot silently drift apart.

Design constraints:
  * Pure/dependency-injected — no ``request.app.state`` access here. The
    endpoint gathers inputs and hands them in.
  * Read-only over ``FusionMetrics`` (a ``@dataclass(slots=True)`` — do not add
    fields to it).
  * ``fusion_status=None`` / sidecar unreachable → ``fusion_available=False``,
    health ``"unknown"``, still a valid 200 graph.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from ..classifiers.routing import (
    CONTEXT_DETECTION_TRIGGER,
    CONTEXT_LOCALIZED_RENDER,
    CONTEXT_OMNI_CONTINUOUS,
    RoutingConfig,
)
from ..config import Settings, TrackingConfig
from ..core.config_groups import EXPOSED_CONFIG_KEYS
from ..core.rules import RuleDef
from ..models import (
    NodeCapability,
    NodeSpec,
    PipelineGraph,
    PipelineGraphColumn,
    PipelineGraphEdge,
    PipelineGraphLane,
    PipelineGraphNode,
    PipelineParam,
    PipelineStageKind,
    PipelineStageStatus,
)

# Column vocabulary — shared with config_groups.py stage ids.
COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("sources", "Sources", 0),
    ("preprocess", "Preprocess", 1),
    ("gates", "Gates", 2),
    ("localize", "Localization", 3),
    ("beamform", "Beamform", 4),
    ("classify", "Classifiers", 5),
    ("track", "Tracking", 6),
    ("alert", "Rules & Alerts", 7),
)

_HealthT = str  # "ok" | "warn" | "danger" | "idle" | "off" | "unknown"


def _fmt(value: Any, *, suffix: str = "", digits: int | None = None) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        if digits is not None:
            return f"{value:.{digits}f}{suffix}"
        text = f"{value:g}"
        return f"{text}{suffix}"
    return f"{value}{suffix}"


def _param(label: str, value: Any, config_key: str | None = None, **kw: Any) -> PipelineParam:
    # Only keep config_key if it is actually exposed by the flat config API, so
    # deep links never point at a non-existent key (enforced by tests).
    if config_key is not None and config_key not in EXPOSED_CONFIG_KEYS:
        config_key = None
    return PipelineParam(label=label, value=_fmt(value, **kw), config_key=config_key)


def _metrics(fusion_status: dict | None) -> dict[str, Any]:
    if not isinstance(fusion_status, dict):
        return {}
    m = fusion_status.get("metrics")
    return m if isinstance(m, dict) else {}


def _capabilities(node: NodeSpec) -> set[str]:
    return {c.value if isinstance(c, NodeCapability) else str(c) for c in (node.capabilities or [])}


def build_pipeline_graph(
    *,
    settings: Settings,
    nodes: Iterable[NodeSpec],
    routing: RoutingConfig,
    rules: Iterable[RuleDef],
    fusion_status: dict | None,
    sidecar_dsp_status: dict | None,
    active_pipeline: str,
    now_ns: int,
) -> PipelineGraph:
    nodes = list(nodes)
    rules = list(rules)
    fusion_available = fusion_status is not None
    metrics = _metrics(fusion_status)
    unknown_health: _HealthT = "unknown" if not fusion_available else "ok"

    graph_nodes: list[PipelineGraphNode] = []
    graph_edges: list[PipelineGraphEdge] = []
    lanes: list[PipelineGraphLane] = []
    columns = [PipelineGraphColumn(id=i, title=t, order=o) for (i, t, o) in COLUMNS]

    def add_node(**kw: Any) -> PipelineGraphNode:
        n = PipelineGraphNode(**kw)
        graph_nodes.append(n)
        return n

    def add_edge(
        source: str,
        target: str,
        *,
        kind: str = "audio",
        label: str = "",
        active: bool = True,
        edge_id: str | None = None,
    ) -> None:
        graph_edges.append(
            PipelineGraphEdge(
                # Context routes can deliberately connect the same pair of
                # cards more than once. Their routing identity must survive
                # even though they share endpoints.
                id=edge_id or f"{source}->{target}",
                source=source,
                target=target,
                kind=kind,
                label=label,
                active=active,
            )
        )

    # ── Site lane header ────────────────────────────────────────────────────
    lanes.append(PipelineGraphLane(id="site", title="Site (cross-node)", node_type=None, health=None, link=None, order=0))

    audio_node_ids: list[str] = []
    gate_ids: list[str] = []
    doa_ids: list[str] = []
    ble_present = False

    # ── Per audio-node lanes ────────────────────────────────────────────────
    for order, node in enumerate(sorted(nodes, key=lambda n: n.id), start=1):
        caps = _capabilities(node)
        node_id = node.id
        node_type = node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type)
        has_audio = "audio" in caps or not caps  # legacy nodes without caps assumed audio
        lane_link = f"/settings/nodes/{node_id}"
        lanes.append(
            PipelineGraphLane(
                id=node_id,
                title=node_id,
                node_type=node_type,
                health=None,
                link=lane_link,
                order=order,
            )
        )

        if has_audio:
            audio_node_ids.append(node_id)
            mic_count = max(1, len(node.sensor_offsets_m))
            overrides: dict = settings.node_audio_overrides.get(node_id) or {}

            # Source
            src_id = f"src:{node_id}:audio"
            add_node(
                id=src_id,
                stage=PipelineStageKind.SOURCE,
                column="sources",
                lane=node_id,
                title=node_id,
                subtitle=f"{node_type} · {mic_count} mic{'s' if mic_count != 1 else ''}",
                modality="audio",
                enabled=True,
                node_type=node_type,
                params=[_param("Mics", mic_count), _param("Type", node_type)],
                status=_node_audio_status(node_id, sidecar_dsp_status, fusion_available),
                link=lane_link,
            )

            # Preprocess
            pre_id = f"pre:{node_id}"
            if overrides.get("stages"):
                pre_params = [_param("Stages", len(overrides["stages"]))]
                for i, st in enumerate(overrides["stages"]):
                    if isinstance(st, dict):
                        pre_params.append(_param(f"[{i}] {st.get('type', '?')}", _stage_summary(st)))
            else:
                pre_params = [
                    _param("Gain ×", float(settings.ingest_gain_multiplier), "ingest_gain_multiplier"),
                    _param("Highpass", float(overrides.get("hp_hz", settings.audio_highpass_hz)), "audio_highpass_hz", suffix=" Hz"),
                    _param("Lowpass", float(overrides.get("lp_hz", settings.audio_lowpass_hz)), "audio_lowpass_hz", suffix=" Hz"),
                ]
            add_node(
                id=pre_id,
                stage=PipelineStageKind.PREPROCESS,
                column="preprocess",
                lane=node_id,
                title="Preprocess",
                subtitle="per-node override" if overrides else "global",
                modality="audio",
                enabled=bool(settings.preprocess_enabled),
                node_type=node_type,
                params=pre_params,
                status=_node_audio_status(node_id, sidecar_dsp_status, fusion_available),
                link=lane_link,
            )
            add_edge(src_id, pre_id, kind="audio")

            # Gate
            gate_id = f"gate:{node_id}"
            omni_min_rms = routing.context(CONTEXT_OMNI_CONTINUOUS).min_rms
            gate_params = [_param("Trigger RMS", float(settings.trigger_rms), "trigger_rms")]
            if omni_min_rms is not None:
                gate_params.append(_param("Omni min RMS", float(omni_min_rms)))
            add_node(
                id=gate_id,
                stage=PipelineStageKind.GATE,
                column="gates",
                lane=node_id,
                title="RMS Gate",
                subtitle="trigger / omni",
                modality="audio",
                enabled=True,
                node_type=node_type,
                params=gate_params,
                status=PipelineStageStatus(health=unknown_health, summary=""),
                link=lane_link,
            )
            add_edge(pre_id, gate_id, kind="audio")
            gate_ids.append(gate_id)

            # Per-node DOA (only with array_localization capability)
            if NodeCapability.ARRAY_LOCALIZATION.value in caps:
                doa_id = f"doa:{node_id}"
                add_node(
                    id=doa_id,
                    stage=PipelineStageKind.LOCALIZATION,
                    column="localize",
                    lane=node_id,
                    title="Array DOA",
                    subtitle="per-node bearing",
                    modality="audio",
                    enabled=True,
                    node_type=node_type,
                    params=[_param("Algorithm", settings.localization_algorithm, "localization_algorithm")],
                    status=PipelineStageStatus(health=unknown_health),
                    link=lane_link,
                )
                add_edge(gate_id, doa_id, kind="audio", label="frames")
                doa_ids.append(doa_id)

        # Stub sources per capability.
        if NodeCapability.BLE_RSSI.value in caps:
            ble_present = True
            ble_id = f"src:{node_id}:ble"
            add_node(
                id=ble_id, stage=PipelineStageKind.SOURCE, column="sources", lane=node_id,
                title=f"{node_id} BLE", subtitle="RSSI beacons", modality="ble",
                enabled=True, node_type=node_type,
                params=[_param("Capability", "ble_rssi")],
                status=PipelineStageStatus(health="idle" if fusion_available else "unknown"),
                link=lane_link,
            )
            add_edge(ble_id, "ble:site:tracking", kind="metadata", label="RSSI")
        if NodeCapability.PTZ_CAMERA.value in caps:
            ptz_id = f"src:{node_id}:ptz"
            add_node(
                id=ptz_id, stage=PipelineStageKind.SOURCE, column="sources", lane=node_id,
                title=f"{node_id} PTZ", subtitle="video / effector", modality="video",
                enabled=True, node_type=node_type,
                params=[_param("Capability", "ptz_camera")],
                status=PipelineStageStatus(health="idle" if fusion_available else "unknown"),
                link=lane_link,
            )
        if NodeCapability.ENVIRONMENT.value in caps or NodeCapability.TEMPERATURE.value in caps:
            env_id = f"src:{node_id}:env"
            add_node(
                id=env_id, stage=PipelineStageKind.SOURCE, column="sources", lane=node_id,
                title=f"{node_id} Env", subtitle="temperature / humidity", modality="environment",
                enabled=True, node_type=node_type,
                params=[_param("Default temp", float(settings.default_temperature_c), "default_temperature_c", suffix=" °C")],
                status=PipelineStageStatus(health="idle" if fusion_available else "unknown"),
                link=lane_link,
            )
            add_edge(env_id, "loc:site:solve", kind="metadata", label="speed of sound")

    # ── Site: cross-node TDOA ───────────────────────────────────────────────
    tdoa_enabled = bool(settings.localization_cross_node_tdoa_enabled)
    if tdoa_enabled:
        tdoa_id = "loc:site:tdoa"
        add_node(
            id=tdoa_id,
            stage=PipelineStageKind.LOCALIZATION,
            column="localize",
            lane="site",
            title="Cross-node TDOA",
            subtitle="pairwise arrivals",
            modality="audio",
            enabled=True,
            params=[
                _param("Max τ", float(settings.localization_cross_node_max_tau_seconds), "localization_cross_node_max_tau_seconds", suffix=" s"),
                _param("Max baseline", float(settings.localization_cross_node_max_baseline_m), "localization_cross_node_max_baseline_m", suffix=" m"),
                _param("Min sync weight", float(settings.localization_cross_node_min_sync_weight), "localization_cross_node_min_sync_weight"),
            ],
            status=_tdoa_status(metrics, fusion_available),
            link="/settings/config#localization",
        )
        for gate_id in gate_ids:
            add_edge(gate_id, tdoa_id, kind="audio", label="TDOA pairs")

    # ── Site: solve ─────────────────────────────────────────────────────────
    solve_id = "loc:site:solve"
    add_node(
        id=solve_id,
        stage=PipelineStageKind.LOCALIZATION,
        column="localize",
        lane="site",
        title="Localization Solve",
        subtitle=f"{settings.localization_strategy} · {settings.localization_algorithm}",
        modality="audio",
        enabled=True,
        params=[
            _param("Algorithm", settings.localization_algorithm, "localization_algorithm"),
            _param("Strategy", settings.localization_strategy, "localization_strategy"),
            _param("Min sensors 3D", settings.min_sensors_for_3d, "min_sensors_for_3d"),
            _param("Min sensors 2D", settings.min_sensors_for_2d, "min_sensors_for_2d"),
            _param("Wavelength gating", bool(settings.wavelength_gating_enabled), "wavelength_gating_enabled"),
        ],
        status=_solve_status(metrics, fusion_available),
        link="/settings/config#localization",
    )
    for doa_id in doa_ids:
        add_edge(doa_id, solve_id, kind="audio", label="bearing")
    if tdoa_enabled:
        add_edge("loc:site:tdoa", solve_id, kind="audio", label="TDOA")

    # ── Site: beamform ──────────────────────────────────────────────────────
    beam_id = "beam:site"
    beam_active = settings.classification_audio_source == "beamformed"
    add_node(
        id=beam_id,
        stage=PipelineStageKind.BEAMFORM,
        column="beamform",
        lane="site",
        title="Beamformer",
        subtitle=settings.beamformer_type,
        modality="audio",
        enabled=beam_active,
        params=[
            _param("Type", settings.beamformer_type, "beamformer_type"),
            _param("MVDR loading", float(settings.mvdr_diagonal_loading), "mvdr_diagonal_loading"),
            _param("Cross-node beam", bool(settings.cross_node_beam_enabled), "cross_node_beam_enabled"),
        ],
        status=(
            _beam_status(metrics, fusion_available)
            if beam_active else PipelineStageStatus(health="off", summary="disabled by classification audio source")
        ),
        link="/settings/config#beamform",
    )
    add_edge(solve_id, beam_id, kind="audio", label="steer")

    # ── Classifiers ─────────────────────────────────────────────────────────
    _add_classifiers(
        routing=routing,
        settings=settings,
        gate_ids=gate_ids,
        beam_id=beam_id,
        add_node=add_node,
        add_edge=add_edge,
        fusion_available=fusion_available,
        metrics=metrics,
    )

    # ── Tracking ────────────────────────────────────────────────────────────
    tc: TrackingConfig = settings.tracking_config()
    track_id = "track:site"
    add_node(
        id=track_id,
        stage=PipelineStageKind.TRACKING,
        column="track",
        lane="site",
        title="Tracking",
        subtitle=tc.tracking_filter,
        modality="audio",
        enabled=True,
        params=[
            _param("Filter", tc.tracking_filter, "tracking_filter"),
            _param("Assoc dist", float(tc.association_distance_m), "association_distance_m", suffix=" m"),
            _param("Stale after", float(tc.track_stale_seconds), "track_stale_seconds", suffix=" s"),
        ],
        status=_track_status(metrics, fusion_available),
        link="/settings/config#tracking",
    )
    # Members feed tracking.
    for n in graph_nodes:
        if n.id.startswith("cls:member:"):
            add_edge(n.id, track_id, kind="metadata", label="detection")

    if ble_present:
        ble_track_id = "ble:site:tracking"
        add_node(
            id=ble_track_id,
            stage=PipelineStageKind.TRACKING,
            column="track",
            lane="site",
            title="BLE Tracking",
            subtitle="RSSI trilateration",
            modality="ble",
            enabled=True,
            params=[_param("Filter", settings.ble_tracking_config().tracking_filter)],
            status=PipelineStageStatus(health="idle" if fusion_available else "unknown"),
            link="/settings/config#tracking",
        )
        add_edge(ble_track_id, track_id, kind="metadata", label="ble track")

    # ── Rules / alerts ──────────────────────────────────────────────────────
    rules_id = "rules:site"
    rule_params: list[PipelineParam] = []
    action_kinds: set[str] = set()
    for rule in rules:
        rule_params.append(_param(rule.rule_id, _rule_condition_summary(rule)))
        for action in rule.actions:
            action_kinds.add(action.action_type)
    add_node(
        id=rules_id,
        stage=PipelineStageKind.RULES,
        column="alert",
        lane="site",
        title="Rules",
        subtitle=f"{len(rules)} rule{'s' if len(rules) != 1 else ''}",
        modality="audio",
        enabled=True,
        params=rule_params or [_param("Rules", 0)],
        status=PipelineStageStatus(health="ok" if fusion_available else "unknown"),
        link="/settings/rules",
    )
    add_edge(track_id, rules_id, kind="metadata", label="tracks")
    # Best-effort member → rules edges (fall back to context-level already covered
    # by member→track→rules). Direct member→rules edge for label-matched rules.
    member_ids = [n.id for n in graph_nodes if n.id.startswith("cls:member:")]
    for member_id in member_ids:
        add_edge(member_id, rules_id, kind="metadata", label="", active=True)

    for kind in sorted(action_kinds) or ["alert"]:
        alert_id = f"alert:{kind}"
        add_node(
            id=alert_id,
            stage=PipelineStageKind.ALERT,
            column="alert",
            lane="site",
            title=kind.replace("_", " ").title(),
            subtitle="action",
            modality="audio",
            enabled=True,
            params=[_param("Type", kind)],
            status=PipelineStageStatus(health="ok" if fusion_available else "unknown"),
            link="/settings/rules",
        )
        add_edge(rules_id, alert_id, kind="alert", label=kind)

    structure_hash = _structure_hash(columns, lanes, graph_nodes, graph_edges)
    return PipelineGraph(
        generated_ns=now_ns,
        active_pipeline=active_pipeline if active_pipeline in ("python", "rust") else "python",
        structure_hash=structure_hash,
        fusion_available=fusion_available,
        columns=columns,
        lanes=lanes,
        nodes=graph_nodes,
        edges=graph_edges,
    )


# ── Classifiers helper ──────────────────────────────────────────────────────
def _add_classifiers(*, routing, settings, gate_ids, beam_id, add_node, add_edge, fusion_available, metrics) -> None:
    health = "ok" if fusion_available else "unknown"
    source_label = settings.classification_audio_source

    ctx_order = [CONTEXT_DETECTION_TRIGGER, CONTEXT_LOCALIZED_RENDER, CONTEXT_OMNI_CONTINUOUS]

    # Members (deduped across contexts).
    seen_members: set[str] = set()
    for ctx in routing.contexts.values():
        for member_id in ctx.run:
            if member_id in seen_members:
                continue
            seen_members.add(member_id)
            spec = routing.classifiers.get(member_id)
            member_node_id = f"cls:member:{member_id}"
            params = []
            if spec is not None:
                params = [
                    _param("Backend", spec.backend),
                    _param("Min conf", float(spec.min_confidence)),
                ]
                if spec.model_path:
                    params.append(_param("Model", spec.model_path))
                if spec.preprocess_profile:
                    params.append(_param("Preprocess", spec.preprocess_profile))
            add_node(
                id=member_node_id,
                stage=PipelineStageKind.CLASSIFIER,
                column="classify",
                lane="site",
                title=member_id,
                subtitle=spec.backend if spec else "member",
                modality="audio",
                enabled=True,
                params=params,
                status=PipelineStageStatus(health=health),
                link="/settings/config#classification",
            )

    # Contexts describe routes rather than processing stages. Showing them as
    # named direct edges keeps a classifier's card singular while retaining
    # the important distinction between trigger, render, and periodic scans.
    for ctx_name in ctx_order:
        ctx = routing.contexts.get(ctx_name)
        if ctx is None:
            continue
        label = ctx_name.replace("_", " ").title()
        for member_id in ctx.run:
            target = f"cls:member:{member_id}"
            if ctx_name == CONTEXT_DETECTION_TRIGGER:
                for gate_id in gate_ids:
                    add_edge(
                        gate_id,
                        target,
                        kind="audio",
                        label=label,
                        edge_id=f"{gate_id}->{target}:detection_trigger",
                    )
            elif ctx_name == CONTEXT_LOCALIZED_RENDER:
                add_edge(
                    beam_id,
                    target,
                    kind="audio",
                    label=label,
                    active=source_label == "beamformed",
                    edge_id=f"{beam_id}->{target}:localized_render",
                )
            elif ctx_name == CONTEXT_OMNI_CONTINUOUS:
                for gate_id in gate_ids:
                    add_edge(
                        gate_id,
                        target,
                        kind="audio",
                        label=label,
                        edge_id=f"{gate_id}->{target}:omni_continuous",
                    )

    # Chain edges (embedding).
    for chain in routing.chains:
        parent = f"cls:member:{chain.after}"
        child = f"cls:member:{chain.chain_id}"
        # Ensure chained member node exists even if not in any context.
        if chain.chain_id not in seen_members:
            seen_members.add(chain.chain_id)
            spec = routing.classifiers.get(chain.chain_id)
            add_node(
                id=child,
                stage=PipelineStageKind.CLASSIFIER,
                column="classify",
                lane="site",
                title=chain.chain_id,
                subtitle=(spec.backend if spec else "chain"),
                modality="audio",
                enabled=True,
                params=[_param("Backend", spec.backend) if spec else _param("Chain", chain.chain_id)],
                status=PipelineStageStatus(health=health),
                link="/settings/config#classification",
            )
        add_edge(parent, child, kind="embedding", label=chain.input)

    # Trigger edges (side-effect hooks, e.g. speech → STT capture). The action's
    # target member is materialized if it isn't already present in a context/chain.
    _action_target_member = {"speech_capture": "stt"}
    for trig in routing.triggers:
        parent = f"cls:member:{trig.on}"
        target_member = _action_target_member.get(trig.action)
        if target_member is None:
            continue
        target = f"cls:member:{target_member}"
        if target_member not in seen_members:
            seen_members.add(target_member)
            spec = routing.classifiers.get(target_member)
            add_node(
                id=target,
                stage=PipelineStageKind.CLASSIFIER,
                column="classify",
                lane="site",
                title=target_member,
                subtitle=(spec.backend if spec else "trigger target"),
                modality="audio",
                enabled=True,
                params=[_param("Backend", spec.backend) if spec else _param("Member", target_member)],
                status=PipelineStageStatus(health=health),
                link="/settings/config#classification",
            )
        add_edge(parent, target, kind="trigger", label=trig.action.replace("_", " "))


# ── Status helpers ──────────────────────────────────────────────────────────
def _node_audio_status(node_id: str, sidecar_dsp_status: dict | None, fusion_available: bool) -> PipelineStageStatus:
    if not fusion_available:
        return PipelineStageStatus(health="unknown")
    return PipelineStageStatus(health="ok")


def _tdoa_status(metrics: dict, fusion_available: bool) -> PipelineStageStatus:
    if not fusion_available:
        return PipelineStageStatus(health="unknown")
    measured = int(metrics.get("localization_cross_node_pairs_measured_count") or 0)
    rejected = int(metrics.get("localization_cross_node_pairs_rejected_sync_count") or 0)
    health = "warn" if rejected > 0 and measured == 0 else "ok"
    return PipelineStageStatus(
        health=health,
        summary=(f"{measured} pairs measured" if measured else "configured; no pairs measured yet"),
        metrics=[
            _param("Pairs measured", measured),
            _param("Rejected (sync)", rejected),
            _param("Last pair count", int(metrics.get("last_cross_node_pair_count") or 0)),
        ],
    )


def _solve_status(metrics: dict, fusion_available: bool) -> PipelineStageStatus:
    if not fusion_available:
        return PipelineStageStatus(health="unknown")
    out = int(metrics.get("localization_stage_out") or 0)
    fails = int(metrics.get("localization_failures") or 0)
    fallback = int(metrics.get("localization_fallback_count") or 0)
    health = "ok" if out > 0 else ("warn" if fails > 0 else "ok")
    return PipelineStageStatus(
        health=health,
        summary=(
            f"last: {metrics.get('last_localization_algorithm', '?')}"
            if out else "configured; no solves yet"
        ),
        metrics=[
            _param("Solved", out),
            _param("Failures", fails),
            _param("Fallbacks", fallback),
            _param("Attempted", str(metrics.get("last_attempted_algorithm", "?"))),
        ],
    )


def _beam_status(metrics: dict, fusion_available: bool) -> PipelineStageStatus:
    if not fusion_available:
        return PipelineStageStatus(health="unknown")
    renders = int(metrics.get("beamform_renders") or 0)
    failures = int(metrics.get("beamform_failures") or 0)
    health = "ok" if renders > 0 else ("warn" if failures > 0 else "ok")
    return PipelineStageStatus(
        health=health,
        summary=f"{renders} renders" if renders else "configured; no renders yet",
        metrics=[_param("Renders", renders), _param("Failures", failures)],
    )


def _track_status(metrics: dict, fusion_available: bool) -> PipelineStageStatus:
    if not fusion_available:
        return PipelineStageStatus(health="unknown")
    associations = int(metrics.get("track_multi_node_association_count") or 0)
    active_tracks = int(metrics.get("tracks_multi_node_active") or 0)
    return PipelineStageStatus(
        health="ok",
        summary=(f"{active_tracks} active multi-node tracks" if active_tracks else "configured; no active tracks"),
        metrics=[
            _param("Multi-node associations", associations),
            _param("Active multi-node tracks", active_tracks),
        ],
    )


def _stage_summary(stage: dict) -> str:
    t = stage.get("type")
    if t == "gain":
        return _fmt(float(stage.get("db", 0.0)), suffix=" dB")
    if t in ("highpass", "lowpass"):
        return _fmt(float(stage.get("cutoff_hz", stage.get("hz", 0.0))), suffix=" Hz")
    return str(t)


def _rule_condition_summary(rule: RuleDef) -> str:
    c = rule.condition
    parts: list[str] = []
    if c.labels:
        parts.append("labels: " + ",".join(sorted(c.labels)))
    if c.label_categories:
        parts.append("cats: " + ",".join(sorted(c.label_categories)))
    if c.zone_ids:
        parts.append("zones: " + ",".join(sorted(c.zone_ids)))
    if c.min_confidence is not None:
        parts.append(f"conf≥{c.min_confidence:g}")
    summary = "; ".join(parts) if parts else "any"
    if not rule.enabled:
        summary = f"(disabled) {summary}"
    return summary


def _structure_hash(columns, lanes, nodes, edges) -> str:
    """sha1 of structural fields only (excludes live status), so the frontend
    can skip re-layout when only status changes across polls."""
    payload = {
        "columns": [(c.id, c.order) for c in columns],
        "lanes": [(l.id, l.order, l.node_type) for l in lanes],
        "nodes": [
            (n.id, n.stage.value, n.column, n.lane, n.enabled, n.node_type,
             [(p.label, p.value, p.config_key) for p in n.params], n.title, n.subtitle)
            for n in nodes
        ],
        "edges": [(e.id, e.source, e.target, e.kind, e.label, e.active) for e in edges],
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()
