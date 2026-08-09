"""Tests for the pipeline-flow DAG builder and GET /api/v1/pipeline/graph."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimappr.classifiers.routing import default_routing
from minimappr.config import Settings
from minimappr.core.config_groups import EXPOSED_CONFIG_KEYS
from minimappr.core.pipeline_graph import build_pipeline_graph
from minimappr.core.rules import default_rules
from minimappr.main import app
from minimappr.models import (
    NodeCapability,
    NodeSpec,
    NodeType,
    PipelineGraph,
)


def _tetra(node_id: str) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type=NodeType.SIRITH_TETRA,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1)],
        capabilities=[NodeCapability.AUDIO, NodeCapability.ARRAY_LOCALIZATION],
    )


def _point_ble(node_id: str) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type=NodeType.POINT,
        position_m=(5.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=[NodeCapability.AUDIO, NodeCapability.BLE_RSSI],
    )


def _build(settings=None, nodes=None, fusion_status=None, **kw) -> PipelineGraph:
    settings = settings or Settings()
    nodes = nodes if nodes is not None else [_tetra("t1"), _tetra("t2"), _point_ble("p1")]
    return build_pipeline_graph(
        settings=settings,
        nodes=nodes,
        routing=kw.pop("routing", None) or default_routing(),
        rules=kw.pop("rules", None) or default_rules(),
        fusion_status=fusion_status,
        sidecar_dsp_status=None,
        active_pipeline=kw.pop("active_pipeline", "python"),
        now_ns=time.time_ns(),
    )


class TestBuilderTopology:
    def test_lanes_columns_present(self):
        g = _build()
        lane_ids = {l.id for l in g.lanes}
        assert lane_ids == {"site", "t1", "t2", "p1"}
        assert [c.id for c in g.columns] == [
            "sources", "preprocess", "gates", "localize",
            "beamform", "classify", "track", "alert",
        ]

    def test_all_edge_endpoints_exist(self):
        g = _build(fusion_status={"metrics": {}})
        node_ids = {n.id for n in g.nodes}
        for e in g.edges:
            assert e.source in node_ids, f"missing edge source {e.source}"
            assert e.target in node_ids, f"missing edge target {e.target}"

    def test_ids_unique(self):
        g = _build()
        ids = [n.id for n in g.nodes]
        assert len(ids) == len(set(ids))
        edge_ids = [e.id for e in g.edges]
        assert len(edge_ids) == len(set(edge_ids))

    def test_tdoa_present_iff_enabled(self):
        g = _build()
        assert any(n.id == "loc:site:tdoa" for n in g.nodes)
        s = Settings()
        s.localization_cross_node_tdoa_enabled = False
        g2 = _build(settings=s)
        assert not any(n.id == "loc:site:tdoa" for n in g2.nodes)

    def test_doa_only_for_array_nodes(self):
        g = _build()
        assert any(n.id == "doa:t1" for n in g.nodes)
        assert not any(n.id == "doa:p1" for n in g.nodes)  # point node has no array cap

    def test_ble_stub_source_and_tracking(self):
        g = _build()
        assert any(n.id == "src:p1:ble" for n in g.nodes)
        assert any(n.id == "ble:site:tracking" for n in g.nodes)

    def test_kill_switch_removes_member(self):
        s = Settings()
        s.birdnet_enabled = False
        g = _build(settings=s, routing=None)  # routing rebuilt from default within _build
        # default routing kill-switches are applied by load_routing, not default_routing;
        # emulate by passing a routing without birdnet.
        from minimappr.classifiers.routing import load_routing
        g = _build(settings=s, routing=load_routing(s))
        assert not any(n.id == "cls:member:birdnet" for n in g.nodes)
        # no edges reference the removed member
        for e in g.edges:
            assert "birdnet" not in e.source and "birdnet" not in e.target

    def test_classifier_contexts_are_named_direct_edges(self):
        g = _build(fusion_status={"metrics": {}})
        assert not any(n.id.startswith("cls:ctx:") for n in g.nodes)
        birdnet = "cls:member:birdnet"
        birdnet_labels = {e.label for e in g.edges if e.target == birdnet}
        assert {"Localized Render", "Omni Continuous"} <= birdnet_labels
        assert sum(n.id == birdnet for n in g.nodes) == 1

    def test_context_routes_with_shared_endpoints_have_unique_ids(self):
        g = _build(fusion_status={"metrics": {}})
        edge_ids = [e.id for e in g.edges]
        assert len(edge_ids) == len(set(edge_ids))
        birdnet_from_t1 = [
            e for e in g.edges
            if e.source == "gate:t1" and e.target == "cls:member:birdnet"
        ]
        assert {e.label for e in birdnet_from_t1} == {"Omni Continuous"}


class TestBuilderStatus:
    def test_structure_hash_stable_across_status_change(self):
        g1 = _build(fusion_status={"metrics": {"localization_stage_out": 1}})
        g2 = _build(fusion_status={"metrics": {"localization_stage_out": 999, "beamform_renders": 5}})
        assert g1.structure_hash == g2.structure_hash

    def test_structure_hash_changes_on_config_change(self):
        g1 = _build()
        s = Settings()
        s.beamformer_type = "delay_and_sum"
        g2 = _build(settings=s)
        assert g1.structure_hash != g2.structure_hash

    def test_fusion_none_marks_unavailable(self):
        g = _build(fusion_status=None)
        assert g.fusion_available is False
        solve = next(n for n in g.nodes if n.id == "loc:site:solve")
        assert solve.status.health == "unknown"

    def test_fusion_present_available(self):
        g = _build(fusion_status={"metrics": {"localization_stage_out": 3}})
        assert g.fusion_available is True

    def test_enabled_zero_work_site_stages_are_ok(self):
        g = _build(fusion_status={"metrics": {}})
        health_by_id = {n.id: n.status.health for n in g.nodes}
        assert health_by_id["loc:site:tdoa"] == "ok"
        assert health_by_id["loc:site:solve"] == "ok"
        assert health_by_id["beam:site"] == "ok"
        assert health_by_id["track:site"] == "ok"

    def test_disabled_beamformer_is_off(self):
        settings = Settings()
        settings.classification_audio_source = "omni"
        g = _build(settings=settings, fusion_status={"metrics": {}})
        beam = next(n for n in g.nodes if n.id == "beam:site")
        assert beam.status.health == "off"

    def test_tracking_uses_tracking_telemetry(self):
        g = _build(fusion_status={"metrics": {
            "track_multi_node_association_count": 7,
            "tracks_multi_node_active": 2,
            "track_dormant_reacquired_count": 3,
            "classification_stage_out": 999,
        }})
        track = next(n for n in g.nodes if n.id == "track:site")
        metrics = {p.label: p.value for p in track.status.metrics}
        assert metrics == {
            "Multi-node associations": "7",
            "Active multi-node tracks": "2",
            "Dormant reacquisitions": "3",
        }


class TestHassBridgeStage:
    """AGENTS §2.5: a new pipeline stage must register in the DAG builder."""

    def test_absent_when_hass_is_disabled(self):
        graph = _build(settings=Settings())
        assert not any(node.id == "alert:hass_bridge" for node in graph.nodes)

    def test_present_with_an_edge_from_rules_when_enabled(self):
        graph = _build(settings=Settings(hass_enabled=True, hass_mqtt_host="mqtt.local"))
        node = next(node for node in graph.nodes if node.id == "alert:hass_bridge")
        assert node.title == "Home Assistant"
        assert node.link == "/settings/integrations"
        assert any(
            edge.source == "rules:site" and edge.target == "alert:hass_bridge"
            for edge in graph.edges
        )

    def test_params_surface_the_broker_and_base_topic(self):
        graph = _build(
            settings=Settings(
                hass_enabled=True, hass_mqtt_host="mqtt.local", hass_base_topic="site_a"
            )
        )
        node = next(node for node in graph.nodes if node.id == "alert:hass_bridge")
        values = {param.label: param.value for param in node.params}
        assert values["Broker"] == "mqtt.local"
        assert values["Base topic"] == "site_a"


class TestConfigKeyCoverage:
    def test_every_config_key_is_exposed(self):
        g = _build(fusion_status={"metrics": {}})
        for n in g.nodes:
            for p in list(n.params) + list(n.status.metrics):
                if p.config_key is not None:
                    assert p.config_key in EXPOSED_CONFIG_KEYS


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

def _configure_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")


class TestGraphEndpoint:
    def test_returns_200_and_parses(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.get("/api/v1/pipeline/graph")
            assert resp.status_code == 200
            graph = PipelineGraph.model_validate(resp.json())
            assert graph.active_pipeline == "python"
            assert any(c.id == "sources" for c in graph.columns)


class TestClassifyLaneStatus:
    def test_localized_render_runs_yamnet_and_birdnet_unconditionally(self):
        """The DAG must reflect the real routing: BirdNET is a run member on
        every triggered (localized_render) inference, not a gated chain."""
        graph = _build()
        localized_edges = [
            e for e in graph.edges if (e.label or "").lower() == "localized render"
        ]
        targets = {e.target for e in localized_edges}
        assert "cls:member:yamnet" in targets
        assert "cls:member:birdnet" in targets

    def test_classifier_members_surface_lane_timing_metrics(self):
        fusion_status = {
            "metrics": {
                "classification_stage_in": 10,
                "classification_stage_out": 4,
                "classification_stage_total_time_ms": 500.0,
                "classification_stage_max_time_ms": 120.0,
            }
        }
        graph = _build(fusion_status=fusion_status)
        member = next(n for n in graph.nodes if n.id == "cls:member:yamnet")
        by_label = {p.label: p.value for p in (member.status.metrics or [])}
        assert {"Lane in", "Lane out", "Avg ms", "Max ms"} <= set(by_label)
        assert by_label["Lane in"] == "10"
        # _fmt renders whole-valued floats without the trailing ".0"
        assert by_label["Avg ms"] == "50"
        assert by_label["Max ms"] == "120"
