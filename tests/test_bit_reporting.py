"""Tests for Built-In Test (BIT) reporting: models, evaluator, storage, and API."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from minimappr.core.bit_report import BITReportEvaluator, _derive_overall_status
from minimappr.models import (
    BITReport,
    BITReportIn,
    BITStatus,
    BITTestResult,
    BITType,
    NodeHealthStatus,
)


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


class TestBITModels:
    def test_bit_test_result_pass(self):
        result = BITTestResult(test_name="mic_ch0_noise_floor", status=BITStatus.PASS)
        assert result.failure_code is None
        assert result.subsystem is None

    def test_bit_test_result_fail_with_code(self):
        result = BITTestResult(
            test_name="mic_ch3_clip",
            status=BITStatus.FAIL,
            failure_code="CBIT_FAIL: MIC_CH3_CLIP",
            detail="Channel 3 ADC clipping detected at 98% duty cycle",
            measured_value=0.98,
            threshold=0.05,
            subsystem="audio",
        )
        assert result.failure_code == "CBIT_FAIL: MIC_CH3_CLIP"
        assert result.subsystem == "audio"

    def test_bit_report_in_minimal(self):
        report = BITReportIn(
            report_type=BITType.CBIT,
            results=[BITTestResult(test_name="heartbeat", status=BITStatus.PASS)],
        )
        assert report.report_type == BITType.CBIT
        assert report.firmware_version is None
        assert report.timestamp_ns is None

    def test_bit_report_in_rejects_empty_results(self):
        with pytest.raises(Exception):
            BITReportIn(report_type=BITType.PBIT, results=[])

    def test_bit_report_full(self):
        report = BITReport(
            id="abc123",
            node_id="node-1",
            report_type=BITType.PBIT,
            overall_status=BITStatus.PASS,
            timestamp_ns=1_000_000,
            received_ns=1_000_100,
            results=[BITTestResult(test_name="boot_check", status=BITStatus.PASS)],
            failure_codes=[],
            firmware_version="1.2.3",
            uptime_seconds=0.5,
        )
        assert report.node_id == "node-1"
        assert report.failure_codes == []

    def test_node_health_status_enum_values(self):
        assert NodeHealthStatus.BIT_FAIL.value == "bit_fail"
        assert NodeHealthStatus.ONLINE.value == "online"


# ---------------------------------------------------------------------------
# _derive_overall_status helper
# ---------------------------------------------------------------------------


class TestDeriveOverallStatus:
    def test_all_pass(self):
        results = [
            BITTestResult(test_name="t1", status=BITStatus.PASS),
            BITTestResult(test_name="t2", status=BITStatus.PASS),
        ]
        assert _derive_overall_status(results) == BITStatus.PASS

    def test_any_fail_overrides(self):
        results = [
            BITTestResult(test_name="t1", status=BITStatus.PASS),
            BITTestResult(test_name="t2", status=BITStatus.FAIL, failure_code="X"),
            BITTestResult(test_name="t3", status=BITStatus.DEGRADED),
        ]
        assert _derive_overall_status(results) == BITStatus.FAIL

    def test_degraded_without_fail(self):
        results = [
            BITTestResult(test_name="t1", status=BITStatus.PASS),
            BITTestResult(test_name="t2", status=BITStatus.DEGRADED),
        ]
        assert _derive_overall_status(results) == BITStatus.DEGRADED

    def test_single_fail(self):
        results = [BITTestResult(test_name="t1", status=BITStatus.FAIL, failure_code="ERR")]
        assert _derive_overall_status(results) == BITStatus.FAIL


# ---------------------------------------------------------------------------
# BITReportEvaluator — in-memory behavior
# ---------------------------------------------------------------------------


class TestBITReportEvaluator:
    @pytest.fixture
    def evaluator(self):
        return BITReportEvaluator()

    async def test_submit_and_retrieve(self, evaluator: BITReportEvaluator):
        report_in = BITReportIn(
            report_type=BITType.CBIT,
            timestamp_ns=100_000,
            results=[
                BITTestResult(test_name="mic_noise", status=BITStatus.PASS),
                BITTestResult(
                    test_name="lora_link",
                    status=BITStatus.FAIL,
                    failure_code="CBIT_FAIL: LORA_CORE_TIMEOUT",
                ),
            ],
        )
        report = await evaluator.submit_report("node-A", report_in)
        assert report.overall_status == BITStatus.FAIL
        assert "CBIT_FAIL: LORA_CORE_TIMEOUT" in report.failure_codes
        assert report.node_id == "node-A"
        assert report.report_type == BITType.CBIT

        # Retrieve
        reports = await evaluator.latest_reports_for_node("node-A")
        assert len(reports) == 1
        assert reports[0].id == report.id

    async def test_latest_per_type_stored_separately(self, evaluator: BITReportEvaluator):
        pbit = BITReportIn(
            report_type=BITType.PBIT,
            results=[BITTestResult(test_name="boot", status=BITStatus.PASS)],
        )
        cbit = BITReportIn(
            report_type=BITType.CBIT,
            results=[BITTestResult(test_name="heartbeat", status=BITStatus.PASS)],
        )
        await evaluator.submit_report("node-1", pbit, received_ns=1000)
        await evaluator.submit_report("node-1", cbit, received_ns=2000)

        reports = await evaluator.latest_reports_for_node("node-1")
        types = {r.report_type for r in reports}
        assert types == {BITType.PBIT, BITType.CBIT}

    async def test_newer_report_replaces_older(self, evaluator: BITReportEvaluator):
        old = BITReportIn(
            report_type=BITType.CBIT,
            timestamp_ns=1000,
            results=[BITTestResult(test_name="x", status=BITStatus.FAIL, failure_code="OLD")],
        )
        new = BITReportIn(
            report_type=BITType.CBIT,
            timestamp_ns=2000,
            results=[BITTestResult(test_name="x", status=BITStatus.PASS)],
        )
        await evaluator.submit_report("node-1", old)
        await evaluator.submit_report("node-1", new)

        report = await evaluator.latest_report_of_type("node-1", BITType.CBIT)
        assert report is not None
        assert report.overall_status == BITStatus.PASS

    async def test_older_report_does_not_replace_newer(self, evaluator: BITReportEvaluator):
        new = BITReportIn(
            report_type=BITType.CBIT,
            timestamp_ns=2000,
            results=[BITTestResult(test_name="x", status=BITStatus.PASS)],
        )
        old = BITReportIn(
            report_type=BITType.CBIT,
            timestamp_ns=1000,
            results=[BITTestResult(test_name="x", status=BITStatus.FAIL, failure_code="OLD")],
        )
        await evaluator.submit_report("node-1", new)
        await evaluator.submit_report("node-1", old)

        report = await evaluator.latest_report_of_type("node-1", BITType.CBIT)
        assert report is not None
        assert report.overall_status == BITStatus.PASS

    async def test_all_nodes_with_bit_failures(self, evaluator: BITReportEvaluator):
        await evaluator.submit_report(
            "failing-node",
            BITReportIn(
                report_type=BITType.CBIT,
                results=[
                    BITTestResult(
                        test_name="mic",
                        status=BITStatus.FAIL,
                        failure_code="CBIT_FAIL: MIC_CH0_DEAD",
                    ),
                ],
            ),
        )
        await evaluator.submit_report(
            "healthy-node",
            BITReportIn(
                report_type=BITType.CBIT,
                results=[BITTestResult(test_name="mic", status=BITStatus.PASS)],
            ),
        )
        failures = await evaluator.all_nodes_with_bit_failures()
        assert "failing-node" in failures
        assert "CBIT_FAIL: MIC_CH0_DEAD" in failures["failing-node"]
        assert "healthy-node" not in failures

    async def test_empty_node_returns_no_reports(self, evaluator: BITReportEvaluator):
        reports = await evaluator.latest_reports_for_node("nonexistent")
        assert reports == []


# ---------------------------------------------------------------------------
# Health derivation
# ---------------------------------------------------------------------------


class TestHealthDerivation:
    @pytest.fixture
    def evaluator(self):
        return BITReportEvaluator(bit_fail_override_heartbeat=True)

    async def test_no_bit_data_uses_heartbeat(self, evaluator: BITReportEvaluator):
        status = await evaluator.derive_health_status("node-x", "online")
        assert status == "online"

        status = await evaluator.derive_health_status("node-x", "degraded")
        assert status == "degraded"

        status = await evaluator.derive_health_status("node-x", "offline")
        assert status == "offline"

    async def test_bit_fail_overrides_online(self, evaluator: BITReportEvaluator):
        await evaluator.submit_report(
            "node-1",
            BITReportIn(
                report_type=BITType.CBIT,
                results=[
                    BITTestResult(
                        test_name="gps", status=BITStatus.FAIL, failure_code="CBIT_FAIL: GPS_NO_FIX"
                    ),
                ],
            ),
        )
        status = await evaluator.derive_health_status("node-1", "online")
        assert status == NodeHealthStatus.BIT_FAIL.value

    async def test_bit_fail_overrides_offline_when_configured(self, evaluator: BITReportEvaluator):
        """With bit_fail_override_heartbeat=True (default), BIT fail wins over offline."""
        await evaluator.submit_report(
            "node-1",
            BITReportIn(
                report_type=BITType.PBIT,
                results=[
                    BITTestResult(
                        test_name="boot", status=BITStatus.FAIL, failure_code="PBIT_FAIL: FLASH_CRC"
                    ),
                ],
            ),
        )
        status = await evaluator.derive_health_status("node-1", "offline")
        assert status == NodeHealthStatus.BIT_FAIL.value

    async def test_bit_fail_does_not_override_offline_when_disabled(self):
        evaluator = BITReportEvaluator(bit_fail_override_heartbeat=False)
        await evaluator.submit_report(
            "node-1",
            BITReportIn(
                report_type=BITType.CBIT,
                results=[
                    BITTestResult(
                        test_name="x", status=BITStatus.FAIL, failure_code="CBIT_FAIL: X"
                    ),
                ],
            ),
        )
        # With override=False, offline heartbeat is kept when offline
        status = await evaluator.derive_health_status("node-1", "offline")
        assert status == NodeHealthStatus.OFFLINE.value
        # But online heartbeat still becomes bit_fail
        status = await evaluator.derive_health_status("node-1", "online")
        assert status == NodeHealthStatus.BIT_FAIL.value

    async def test_bit_degraded_demotes_online(self, evaluator: BITReportEvaluator):
        await evaluator.submit_report(
            "node-1",
            BITReportIn(
                report_type=BITType.CBIT,
                results=[BITTestResult(test_name="snr", status=BITStatus.DEGRADED)],
            ),
        )
        status = await evaluator.derive_health_status("node-1", "online")
        assert status == NodeHealthStatus.DEGRADED.value

    async def test_bit_pass_preserves_heartbeat(self, evaluator: BITReportEvaluator):
        await evaluator.submit_report(
            "node-1",
            BITReportIn(
                report_type=BITType.CBIT,
                results=[BITTestResult(test_name="all_ok", status=BITStatus.PASS)],
            ),
        )
        assert await evaluator.derive_health_status("node-1", "online") == "online"
        assert await evaluator.derive_health_status("node-1", "degraded") == "degraded"

    async def test_stale_cbit_causes_degradation(self):
        evaluator = BITReportEvaluator(stale_cbit_seconds=5.0)
        old_ns = time.time_ns() - 10_000_000_000  # 10 seconds ago
        await evaluator.submit_report(
            "node-1",
            BITReportIn(
                report_type=BITType.CBIT,
                timestamp_ns=old_ns,
                results=[BITTestResult(test_name="ok", status=BITStatus.PASS)],
            ),
        )
        status = await evaluator.derive_health_status("node-1", "online")
        assert status == NodeHealthStatus.DEGRADED.value


# ---------------------------------------------------------------------------
# Storage round-trip
# ---------------------------------------------------------------------------


class TestBITStorage:
    @pytest.fixture
    async def storage(self, tmp_path: Path):
        from minimappr.storage.db import Storage

        db_path = tmp_path / "bit_test.db"
        store = Storage(db_path)
        await store.initialize()

        # Ensure a node exists for FK
        from minimappr.models import NodeSpec, NodeType

        spec = NodeSpec(
            id="node-1",
            node_type=NodeType.POINT,
            position_m=(1.0, 2.0, 0.0),
            sensor_offsets_m=[(0.0, 0.0, 0.0)],
        )
        await store.upsert_node(spec, last_seen_ns=time.time_ns())
        yield store
        await store.close()

    async def test_insert_and_list(self, storage):
        await storage.insert_bit_report(
            report_id="rpt-1",
            node_id="node-1",
            report_type="cbit",
            overall_status="fail",
            timestamp_ns=100_000,
            received_ns=100_100,
            results_json=json.dumps([{"test_name": "mic", "status": "fail", "failure_code": "X"}]),
            failure_codes_json=json.dumps(["X"]),
            firmware_version="2.0.0",
            uptime_seconds=600.0,
            metadata_json="{}",
        )
        rows = await storage.list_bit_reports(node_id="node-1")
        assert len(rows) == 1
        assert rows[0]["id"] == "rpt-1"
        assert rows[0]["overall_status"] == "fail"
        assert rows[0]["failure_codes"] == ["X"]
        assert rows[0]["firmware_version"] == "2.0.0"

    async def test_list_filters_by_type(self, storage):
        for i, rt in enumerate(["pbit", "cbit", "ibit"]):
            await storage.insert_bit_report(
                report_id=f"rpt-{i}",
                node_id="node-1",
                report_type=rt,
                overall_status="pass",
                timestamp_ns=100_000 + i,
                received_ns=100_100 + i,
                results_json="[]",
                failure_codes_json="[]",
                firmware_version=None,
                uptime_seconds=None,
                metadata_json="{}",
            )
        cbit_rows = await storage.list_bit_reports(node_id="node-1", report_type="cbit")
        assert len(cbit_rows) == 1
        assert cbit_rows[0]["report_type"] == "cbit"

    async def test_latest_per_type(self, storage):
        # Insert two CBITs — only the newest should appear
        await storage.insert_bit_report(
            report_id="old",
            node_id="node-1",
            report_type="cbit",
            overall_status="fail",
            timestamp_ns=1000,
            received_ns=1000,
            results_json="[]",
            failure_codes_json='["OLD"]',
            firmware_version=None,
            uptime_seconds=None,
            metadata_json="{}",
        )
        await storage.insert_bit_report(
            report_id="new",
            node_id="node-1",
            report_type="cbit",
            overall_status="pass",
            timestamp_ns=2000,
            received_ns=2000,
            results_json="[]",
            failure_codes_json="[]",
            firmware_version=None,
            uptime_seconds=None,
            metadata_json="{}",
        )
        await storage.insert_bit_report(
            report_id="pbit-1",
            node_id="node-1",
            report_type="pbit",
            overall_status="pass",
            timestamp_ns=500,
            received_ns=500,
            results_json="[]",
            failure_codes_json="[]",
            firmware_version=None,
            uptime_seconds=None,
            metadata_json="{}",
        )
        latest = await storage.latest_bit_report_per_type("node-1")
        types = {r["report_type"] for r in latest}
        assert types == {"cbit", "pbit"}
        cbit = next(r for r in latest if r["report_type"] == "cbit")
        assert cbit["id"] == "new"


# ---------------------------------------------------------------------------
# HTTP API integration
# ---------------------------------------------------------------------------


class TestBITHttpAPI:
    @pytest.fixture
    def configured_env(self, monkeypatch, tmp_path: Path):
        db_path = tmp_path / "bit_api.db"
        monkeypatch.setenv("MINIMAPPR_DB_PATH", str(db_path))
        monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
        monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
        monkeypatch.setenv("MINIMAPPR_FUSION_WORKER_COUNT", "1")
        return db_path

    def _register_node(self, client) -> None:
        """Register a node via ingest so it exists in storage."""
        import numpy as np
        from minimappr.utils.audio import encode_pcm16le_b64

        samples = np.zeros((1, 512), dtype=np.float32)
        payload = {
            "node": {
                "id": "bit-test-node",
                "node_type": "point",
                "position_m": [1.0, 2.0, 0.0],
                "sensor_offsets_m": [[0.0, 0.0, 0.0]],
                "capabilities": ["audio"],
                "metadata": {},
                "properties": {},
            },
            "frame": {
                "start_time_ns": time.time_ns(),
                "sample_rate_hz": 16000,
                "channels": 1,
                "encoding": "pcm16le",
                "samples_b64": encode_pcm16le_b64(samples),
                "sequence": 1,
                "source_type": "raw_sensor",
            },
        }
        resp = client.post("/api/v1/ingest/frame", json=payload)
        assert resp.status_code == 200

    def test_submit_and_get_bit_report(self, configured_env):
        from fastapi.testclient import TestClient
        from minimappr.main import app

        with TestClient(app) as client:
            self._register_node(client)

            bit_payload = {
                "report_type": "cbit",
                "results": [
                    {"test_name": "mic_ch0_noise", "status": "pass"},
                    {
                        "test_name": "lora_link",
                        "status": "fail",
                        "failure_code": "CBIT_FAIL: LORA_CORE_TIMEOUT",
                        "subsystem": "lora",
                    },
                ],
                "firmware_version": "1.0.0",
                "uptime_seconds": 3600.0,
            }
            resp = client.post("/api/v1/nodes/bit-test-node/bit", json=bit_payload)
            assert resp.status_code == 200
            body = resp.json()
            assert body["overall_status"] == "fail"
            assert "CBIT_FAIL: LORA_CORE_TIMEOUT" in body["failure_codes"]
            assert body["node_id"] == "bit-test-node"
            assert body["report_type"] == "cbit"

            # GET reports for node
            resp = client.get("/api/v1/nodes/bit-test-node/bit")
            assert resp.status_code == 200
            reports = resp.json()
            assert len(reports) >= 1

            # GET latest per type
            resp = client.get("/api/v1/nodes/bit-test-node/bit/latest")
            assert resp.status_code == 200

            # GET failures summary
            resp = client.get("/api/v1/bit/failures")
            assert resp.status_code == 200
            failures = resp.json()
            assert "bit-test-node" in failures

    def test_bit_affects_node_health_status(self, configured_env):
        from fastapi.testclient import TestClient
        from minimappr.main import app

        with TestClient(app) as client:
            self._register_node(client)

            # Before BIT report, node should be online (recently seen)
            resp = client.get("/api/v1/nodes")
            assert resp.status_code == 200
            nodes = resp.json()
            node = next((n for n in nodes if n["id"] == "bit-test-node"), None)
            assert node is not None
            assert node["health_status"] == "online"

            # Submit failing BIT report
            client.post(
                "/api/v1/nodes/bit-test-node/bit",
                json={
                    "report_type": "cbit",
                    "results": [
                        {
                            "test_name": "power_supply",
                            "status": "fail",
                            "failure_code": "CBIT_FAIL: PSU_UNDERVOLT",
                        }
                    ],
                },
            )

            # Now health should be bit_fail
            resp = client.get("/api/v1/nodes")
            nodes = resp.json()
            node = next((n for n in nodes if n["id"] == "bit-test-node"), None)
            assert node is not None
            assert node["health_status"] == "bit_fail"
            assert "CBIT_FAIL: PSU_UNDERVOLT" in node.get("bit_failure_codes", [])
