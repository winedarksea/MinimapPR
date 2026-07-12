from __future__ import annotations

import asyncio
import functools
import json
import os
import shutil
import socket
import struct
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import correlate

pytest.importorskip("birdnet")

try:
    from birdnet import model_loader as _birdnet_model_loader  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - environment-dependent optional dependency
    pytest.skip(
        f"BirdNET integration tests require compatible birdnet package (missing model_loader): {exc}",
        allow_module_level=True,
    )

from tests.hybrid_settings import hybrid_production_settings
from minimappr.classifiers.birdnet import BirdNETClassifier
from minimappr.classifiers.factory import create_classifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.beamforming import create_beamformer
from minimappr.core.environment import LiveEnvironmentProvider
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization_dispatch import build_localizer_from_settings
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.preprocessing import create_localization_preprocessor
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import EnvironmentSampleIn, GeoPoint, IngestFrameRequest, NodeSpec, NodeType
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64, read_wav_mono, rms
from tests.helpers import (
    SIRITH_TETRA_SENSOR_OFFSETS_M,
    geometric_array_propagation_delays_s,
    load_sensor_wav_files,
    load_wav_fixture_mono,
    prepend_noise_padding,
    resample_signal,
    split_channels_into_frames,
    synthesize_delayed_array_channels,
    write_delayed_array_wav_files,
)


HOUSE_FINCH_FIXTURE_PATH = Path(__file__).with_name("house_finch.wav")
HOUSE_FINCH_SOURCE_POSITION_M = (0.35, 0.18, 0.07)
HOUSE_FINCH_LABEL = "house finch"
DEFAULT_SITE_ORIGIN = GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0)
TIGHT_SRP_GRID_RESOLUTION_M = 0.05
TIGHT_SRP_SEARCH_PADDING_M = 0.3
# The SIRITH tetrahedral array has a ~0.0525 m aperture.  At the house-finch
# source range (~0.4 m) the spherical-wavefront curvature across the array
# (~2.5 us) is below the inter-sample timing resolution (~5.2 us at 48 kHz), so
# range is physically unobservable -- the array is a direction-of-arrival (DOA)
# sensor, not a range sensor, and the solver correctly reports RANGE_BEARING_PROJECTED
# (bearing is well-determined; range is uncertain and explicitly marked so via an
# elongated radial covariance).  These tests therefore assert *bearing* accuracy,
# the quantity the geometry can actually recover.  Observed angular error is ~2-3 deg
# at 48 kHz and ~6-9 deg at 16 kHz; 12 deg leaves margin for both rates.
TIGHT_BEARING_MAX_ERROR_DEG = 12.0
PRODUCTION_BIRDNET_CHUNK_OVERLAP_SECONDS = 2.0
BIRDNET_TEST_TARGET_CONTEXT_SECONDS = 3.0
MINIMAPPR_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_BINARY_PATH = MINIMAPPR_ROOT / "minimappr-ingest-sidecar" / "target" / "debug" / "minimappr-ingest-sidecar"


@pytest.fixture(scope="module")
def house_finch_fixture_48khz() -> tuple[np.ndarray, int]:
    return load_wav_fixture_mono(HOUSE_FINCH_FIXTURE_PATH)


@pytest.fixture(scope="module")
def birdnet_classifier() -> BirdNETClassifier:
    try:
        return BirdNETClassifier(min_confidence=0.05)
    except RuntimeError as exc:
        pytest.skip(f"BirdNET backend unavailable in this environment: {exc}")


def _node_spec() -> NodeSpec:
    return NodeSpec(
        id="sirith-house-finch",
        node_type=NodeType.SIRITH_TETRA,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=list(SIRITH_TETRA_SENSOR_OFFSETS_M),
        capabilities=["audio", "array_localization"],
        metadata={},
    )


def _sensor_positions() -> dict[str, np.ndarray]:
    return {
        f"sirith-house-finch:ch{index}": np.asarray(offset, dtype=np.float64)
        for index, offset in enumerate(SIRITH_TETRA_SENSOR_OFFSETS_M)
    }


def _resampled_fixture(
    fixture: tuple[np.ndarray, int],
    sample_rate_hz: int,
) -> np.ndarray:
    samples, fixture_sample_rate_hz = fixture
    return resample_signal(samples, fixture_sample_rate_hz, sample_rate_hz)


def _speed_of_sound_mps(temperature_c: float, humidity_fraction: float) -> float:
    humidity_percent = max(0.0, min(1.0, humidity_fraction)) * 100.0
    return 331.3 + (0.606 * temperature_c) + (0.0124 * humidity_percent)


def _synthesized_sensor_wav_paths(
    tmp_path: Path,
    fixture: tuple[np.ndarray, int],
    *,
    sample_rate_hz: int,
    classification_window_seconds: float,
    leading_padding_seconds: float,
    sound_speed_mps: float = 343.2,
) -> dict[str, Path]:
    bird_signal = _resampled_fixture(fixture, sample_rate_hz)
    del classification_window_seconds
    padded_signal = prepend_noise_padding(
        bird_signal,
        pad_samples=max(1, int(round(leading_padding_seconds * sample_rate_hz))),
        noise_rms=0.002,
        seed=sample_rate_hz,
    )
    return write_delayed_array_wav_files(
        tmp_path / f"house_finch_array_{sample_rate_hz}",
        mono_signal=padded_signal,
        sample_rate_hz=sample_rate_hz,
        source_position_m=HOUSE_FINCH_SOURCE_POSITION_M,
        sound_speed_mps=sound_speed_mps,
        filename_prefix="sirith-house-finch_ch",
    )


def _synthesized_fixture_channels(
    fixture: tuple[np.ndarray, int],
    *,
    sample_rate_hz: int,
    sound_speed_mps: float,
    leading_padding_seconds: float = 0.25,
) -> np.ndarray:
    active_signal = _dominant_active_region(_resampled_fixture(fixture, sample_rate_hz), sample_rate_hz)
    padded_signal = prepend_noise_padding(
        active_signal,
        pad_samples=max(1, int(round(leading_padding_seconds * sample_rate_hz))),
        noise_rms=0.002,
        seed=int(round(sound_speed_mps * 10.0)),
    )
    return synthesize_delayed_array_channels(
        padded_signal,
        sample_rate_hz,
        source_position_m=HOUSE_FINCH_SOURCE_POSITION_M,
        sound_speed_mps=sound_speed_mps,
    )


def _build_profile_settings(
    tmp_path: Path,
    *,
    sample_rate_hz: int,
    snippet_retention_seconds: int = 0,
    coordinate_mode: str = "flat",
    trigger_rms: float = 0.006,
    trigger_cooldown_seconds: float = 1.0,
    localization_srp_grid_resolution_m: float | None = None,
    localization_search_padding_m: float | None = None,
) -> Settings:
    settings_kwargs = dict(
        db_path=tmp_path / f"birdnet_hybrid_{sample_rate_hz}.db",
        snippet_dir=tmp_path / f"snippets_{sample_rate_hz}",
        snippet_retention_seconds=snippet_retention_seconds,
        trigger_rms=trigger_rms,
        trigger_cooldown_seconds=trigger_cooldown_seconds,
        fusion_worker_count=1,
        fusion_localization_queue_size=16,
        fusion_classification_queue_size=16,
        fusion_rules_queue_size=16,
        site_origin_lat=DEFAULT_SITE_ORIGIN.lat,
        site_origin_lon=DEFAULT_SITE_ORIGIN.lon,
        site_origin_alt_m=DEFAULT_SITE_ORIGIN.alt_m,
        coordinate_mode=coordinate_mode,
        model_chain_config_path=tmp_path / "missing_model_chain.json",
        birdnet_trigger_min_confidence=0.05,
        birdnet_chunk_overlap_seconds=PRODUCTION_BIRDNET_CHUNK_OVERLAP_SECONDS,
    )
    if localization_srp_grid_resolution_m is not None:
        settings_kwargs["localization_srp_grid_resolution_m"] = localization_srp_grid_resolution_m
    if localization_search_padding_m is not None:
        settings_kwargs["localization_search_padding_m"] = localization_search_padding_m
    settings = hybrid_production_settings(**settings_kwargs)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)
    return settings


async def _start_fusion_with_profile(
    tmp_path: Path,
    *,
    sample_rate_hz: int,
    snippet_retention_seconds: int = 0,
    coordinate_mode: str = "flat",
    trigger_rms: float = 0.006,
    trigger_cooldown_seconds: float = 1.0,
    localization_srp_grid_resolution_m: float | None = None,
    localization_search_padding_m: float | None = None,
) -> tuple[FusionNode, Storage, Settings]:
    settings = _build_profile_settings(
        tmp_path,
        sample_rate_hz=sample_rate_hz,
        snippet_retention_seconds=snippet_retention_seconds,
        coordinate_mode=coordinate_mode,
        trigger_rms=trigger_rms,
        trigger_cooldown_seconds=trigger_cooldown_seconds,
        localization_srp_grid_resolution_m=localization_srp_grid_resolution_m,
        localization_search_padding_m=localization_search_padding_m,
    )
    storage = Storage(settings.db_path)
    await storage.initialize()
    coordinate_frame = LocalCoordinateFrame(origin=DEFAULT_SITE_ORIGIN, mode=settings.coordinate_mode)
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=build_localizer_from_settings(settings),
        classifier=create_classifier(settings),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=coordinate_frame,
        zone_matcher=ZoneMatcher(storage=storage),
        environment_provider=LiveEnvironmentProvider(
            fallback_temperature_c=settings.default_temperature_c,
            fallback_humidity_fraction=settings.default_humidity,
        ),
    )
    await fusion.start()
    return fusion, storage, settings


async def _stream_fixture_into_fusion(
    fusion: FusionNode,
    *,
    sample_rate_hz: int,
    frame_samples: int,
    fixture: tuple[np.ndarray, int],
    classification_window_seconds: float,
    start_time_ns: int,
    environment: EnvironmentSampleIn | None = None,
    sound_speed_mps: float = 343.2,
) -> int:
    birdnet_chunk_stride_seconds = max(
        fusion.settings.localization_window_seconds,
        fusion.settings.classification_window_seconds - fusion.settings.birdnet_chunk_overlap_seconds,
    )
    leading_padding_seconds = max(
        fusion.settings.localization_window_seconds,
        birdnet_chunk_stride_seconds - BIRDNET_TEST_TARGET_CONTEXT_SECONDS,
    )
    sensor_paths = _synthesized_sensor_wav_paths(
        Path(fusion.settings.snippet_dir).parent,
        fixture,
        sample_rate_hz=sample_rate_hz,
        classification_window_seconds=classification_window_seconds,
        leading_padding_seconds=leading_padding_seconds,
        sound_speed_mps=sound_speed_mps,
    )
    windows, loaded_sample_rate_hz = load_sensor_wav_files(sensor_paths)
    assert loaded_sample_rate_hz == sample_rate_hz
    channels = np.stack(
        [windows[f"sirith-house-finch_ch{index}"] for index in range(len(SIRITH_TETRA_SENSOR_OFFSETS_M))],
        axis=0,
    )
    return await _stream_channels_into_fusion(
        fusion,
        sample_rate_hz=sample_rate_hz,
        frame_samples=frame_samples,
        channels=channels,
        start_time_ns=start_time_ns,
        environment=environment,
    )


async def _stream_channels_into_fusion(
    fusion: FusionNode,
    *,
    sample_rate_hz: int,
    frame_samples: int,
    channels: np.ndarray,
    start_time_ns: int,
    environment: EnvironmentSampleIn | None = None,
) -> int:
    triggered_frames = 0
    for sequence, (frame_start_ns, frame) in enumerate(
        split_channels_into_frames(
            channels,
            sample_rate_hz=sample_rate_hz,
            start_time_ns=start_time_ns,
            frame_samples=frame_samples,
        ),
        start=1,
    ):
        response = await fusion.ingest(
            IngestFrameRequest(
                node=_node_spec(),
                frame={
                    "start_time_ns": frame_start_ns,
                    "sample_rate_hz": sample_rate_hz,
                    "channels": int(frame.shape[0]),
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(frame),
                    "sequence": sequence,
                },
                environment=environment,
            )
        )
        assert response.accepted is True
        triggered_frames += int(response.triggered)
    return triggered_frames


def _binary_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    assert len(encoded) <= 255
    return struct.pack("<B", len(encoded)) + encoded


def _binary_node(*, node_id: str) -> bytes:
    payload = bytearray()
    payload += _binary_string(node_id)
    payload += struct.pack("<B", 0)
    payload += struct.pack("<B", 1)  # has_geo_position
    payload += struct.pack("<fff", 44.987, -93.258, 0.0)  # fallback geo
    payload += struct.pack("<B", len(SIRITH_TETRA_SENSOR_OFFSETS_M))
    for offset_x, offset_y, offset_z in SIRITH_TETRA_SENSOR_OFFSETS_M:
        payload += struct.pack("<fff", offset_x, offset_y, offset_z)
    capabilities = ["audio", "array_localization"]
    payload += struct.pack("<B", len(capabilities))
    for capability in capabilities:
        payload += _binary_string(capability)
    payload += _binary_string("sirith-test-hardware")
    payload += _binary_string("sirith-test-firmware")
    payload += _binary_string("gps_locked")
    payload += _binary_string("test")
    payload += struct.pack("<I", 1)
    return bytes(payload)


def _binary_environment(
    *,
    temperature_c: float | None = None,
    humidity_fraction: float | None = None,
    source: str | None = None,
) -> bytes:
    flags = 0
    payload = bytearray()
    if temperature_c is not None:
        flags |= 0x01
        payload += struct.pack("<f", temperature_c)
    if humidity_fraction is not None:
        flags |= 0x02
        payload += struct.pack("<f", humidity_fraction)
    if source is not None:
        flags |= 0x04
        payload += _binary_string(source)
    return struct.pack("<B", flags) + payload


def _binary_frame(
    samples: np.ndarray,
    *,
    start_time_ns: int,
    sequence: int,
    start_sample_index: int,
    sample_rate_hz: int,
    temperature_c: float | None = None,
    humidity_fraction: float | None = None,
    source: str | None = None,
) -> bytes:
    channels, samples_per_channel = samples.shape
    end_sample_index = start_sample_index + samples_per_channel
    end_time_ns = start_time_ns + int(round(samples_per_channel / sample_rate_hz * 1_000_000_000))
    pcm = np.clip(samples.T, -1.0, 0.9999695)
    pcm16 = (pcm * 32768.0).astype("<i2").tobytes()
    payload = bytearray()
    payload += struct.pack(
        "<QQQQIBQQQB",
        start_time_ns,
        end_time_ns,
        start_sample_index,
        end_sample_index,
        sample_rate_hz,
        channels,
        sequence,
        start_time_ns,
        start_time_ns + 1_000_000,
        0,
    )
    payload += struct.pack("<B", 0)
    payload += _binary_environment(
        temperature_c=temperature_c,
        humidity_fraction=humidity_fraction,
        source=source,
    )
    payload += struct.pack("<I", samples_per_channel)
    payload += pcm16
    return bytes(payload)


def _binary_ingest_payload(*, node_id: str, frames: list[bytes]) -> bytes:
    payload = bytearray()
    payload += b"MMB2"
    payload += struct.pack("<BBH", 2, 0, len(frames))
    payload += _binary_node(node_id=node_id)
    for frame in frames:
        payload += frame
    return bytes(payload)


def _pick_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(
    port: int,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> dict | list:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        method=method,
        headers=headers or {},
    )
    with urllib.request.urlopen(request, timeout=5.0) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload) if payload else {}


def _sidecar_output(process: subprocess.Popen[str]) -> str:
    if process.stdout is None:
        return ""
    try:
        return process.stdout.read()
    except Exception:
        return ""


@functools.lru_cache(maxsize=1)
def _ensure_sidecar_binary() -> Path:
    if shutil.which("cargo") is None:
        pytest.skip("cargo toolchain not available; cannot build the Rust sidecar")
    sidecar_dir = MINIMAPPR_ROOT / "minimappr-ingest-sidecar"
    subprocess.run(
        ["cargo", "build", "--quiet", "--bin", "minimappr-ingest-sidecar"],
        cwd=str(sidecar_dir),
        check=True,
    )
    return SIDECAR_BINARY_PATH


def _wait_for_sidecar_ready(process: subprocess.Popen[str], *, port: int, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                "Rust sidecar exited before becoming ready.\n"
                f"Captured output:\n{_sidecar_output(process)}"
            )
        try:
            payload = _http_json(port, "/healthz")
            if payload.get("status") == "ok":
                return
        except Exception as exc:  # pragma: no cover - transient startup race
            last_error = exc
        time.sleep(0.05)
    raise AssertionError(
        "Timed out waiting for the Rust sidecar health endpoint.\n"
        f"Last error: {last_error}\n"
        f"Captured output:\n{_sidecar_output(process)}"
    )


@contextmanager
def _running_rust_sidecar(
    tmp_path: Path,
    *,
    default_temperature_c: float = 20.0,
    default_humidity_fraction: float = 0.5,
):
    binary_path = _ensure_sidecar_binary()
    if not binary_path.exists():
        pytest.skip(f"Rust sidecar binary not found at {binary_path}")

    port = _pick_free_tcp_port()
    spool_dir = tmp_path / "sidecar_spool"
    env = {
        **os.environ,
        "MINIMAPPR_SIDECAR_PORT": str(port),
        "MINIMAPPR_INGEST_SPOOL_DIR": str(spool_dir),
        "MINIMAPPR_SIDECAR_STORAGE_MODE": "journal",
        "MINIMAPPR_SIDECAR_ALLOW_NON_TMPFS_JOURNAL": "true",
        "MINIMAPPR_RUNTIME_PROFILE": "birdnet_hybrid_production",
        "MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M": str(TIGHT_SRP_GRID_RESOLUTION_M),
        "MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M": str(TIGHT_SRP_SEARCH_PADDING_M),
        "MINIMAPPR_DEFAULT_TEMPERATURE_C": str(default_temperature_c),
        "MINIMAPPR_DEFAULT_HUMIDITY": str(default_humidity_fraction),
        "MINIMAPPR_SIDECAR_TOTAL_JOURNAL_BUDGET_BYTES": str(32 * 1024 * 1024),
        "MINIMAPPR_SIDECAR_ADMISSION_RESERVE_BYTES": str(4 * 1024 * 1024),
        "MINIMAPPR_SIDECAR_DERIVED_CACHE_BUDGET_BYTES": str(16 * 1024 * 1024),
        "MINIMAPPR_SIDECAR_DERIVED_CACHE_ADMISSION_RESERVE_BYTES": str(2 * 1024 * 1024),
    }
    process = subprocess.Popen(
        [str(binary_path)],
        cwd=str(MINIMAPPR_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_sidecar_ready(process, port=port)
        yield port
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)


def _wait_for_rust_localization_manifest(port: int, *, timeout_s: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout_s
    last_results: list[dict] = []
    latest_localization_result: dict | None = None
    latest_classifier_render: dict | None = None
    while time.monotonic() < deadline:
        results = _http_json(port, "/api/v1/dsp/results?limit=20")
        assert isinstance(results, list)
        last_results = results
        for result in results:
            classifier_render = result.get("classifier_render")
            if result.get("manifest_type") == "classifier_render" and isinstance(classifier_render, dict):
                latest_classifier_render = classifier_render
            localization = result.get("localization")
            if result.get("manifest_type") == "localization_result" and isinstance(localization, dict):
                if isinstance(classifier_render, dict):
                    return result
                latest_localization_result = result
        if latest_localization_result is not None and latest_classifier_render is not None:
            merged_result = dict(latest_localization_result)
            merged_result["classifier_render"] = latest_classifier_render
            return merged_result
        time.sleep(0.05)
    raise AssertionError(
        "Timed out waiting for a Rust localization manifest. "
        f"Last manifest types: {[row.get('manifest_type') for row in last_results]}"
    )


async def _wait_for_localized_house_finch(storage: Storage, *, timeout_s: float = 45.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_detections: list[dict] = []
    while asyncio.get_running_loop().time() < deadline:
        last_detections = await storage.list_detections(limit=128)
        for detection in last_detections:
            if detection["label"] != HOUSE_FINCH_LABEL:
                continue
            if detection["reporting_modality"] != "localized":
                continue
            return detection
        await asyncio.sleep(0.25)
    raise AssertionError(
        "Timed out waiting for a localized house finch detection. "
        f"Last labels: {[row['label'] for row in last_detections[:10]]}"
    )


def _localization_error_m(position_m: tuple[float, float, float] | list[float]) -> float:
    return float(
        np.linalg.norm(
            np.asarray(position_m, dtype=np.float64)
            - np.asarray(HOUSE_FINCH_SOURCE_POSITION_M, dtype=np.float64)
        )
    )


def _array_centroid_m() -> np.ndarray:
    return np.mean(np.vstack(list(_sensor_positions().values())), axis=0)


def _bearing_error_deg(position_m: tuple[float, float, float] | list[float]) -> float:
    """Angle (degrees) between the estimated and true source bearing as seen from
    the array centroid.  Range is intentionally ignored -- see
    TIGHT_BEARING_MAX_ERROR_DEG for why the tight array can only recover bearing."""
    centroid_m = _array_centroid_m()
    estimated = np.asarray(position_m, dtype=np.float64) - centroid_m
    truth = np.asarray(HOUSE_FINCH_SOURCE_POSITION_M, dtype=np.float64) - centroid_m
    estimated_norm = float(np.linalg.norm(estimated))
    truth_norm = float(np.linalg.norm(truth))
    if estimated_norm == 0.0 or truth_norm == 0.0:
        return 180.0
    cos_angle = float(np.clip(estimated @ truth / (estimated_norm * truth_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _estimate_integer_sample_lag(reference: np.ndarray, candidate: np.ndarray) -> int:
    correlation = correlate(candidate, reference, mode="full", method="fft")
    return int(np.argmax(correlation) - (reference.size - 1))


def _align_start_time_to_birdnet_chunk(start_time_ns: int, settings: Settings) -> int:
    stride_ns = max(
        1,
        int(
            max(
                settings.localization_window_seconds,
                settings.classification_window_seconds - settings.birdnet_chunk_overlap_seconds,
            )
            * 1_000_000_000
        ),
    )
    return int(start_time_ns) - (int(start_time_ns) % stride_ns)


def _best_contiguous_subsegment_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float, int, bool]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.size == 0 or candidate.size == 0:
        raise AssertionError("Signals must be non-empty")

    if reference.size <= candidate.size:
        shorter = reference
        longer = candidate
        reference_is_shorter = True
    else:
        shorter = candidate
        longer = reference
        reference_is_shorter = False

    if shorter.size == longer.size:
        offset = 0
    else:
        correlation = correlate(longer, shorter, mode="valid", method="fft")
        offset = int(np.argmax(correlation))

    aligned = longer[offset : offset + shorter.size]
    centered_shorter = shorter - np.mean(shorter, dtype=np.float32)
    centered_aligned = aligned - np.mean(aligned, dtype=np.float32)
    denominator = float(
        np.linalg.norm(centered_shorter.astype(np.float64))
        * np.linalg.norm(centered_aligned.astype(np.float64))
    )
    normalized_correlation = 0.0
    if denominator > 0.0:
        normalized_correlation = float(
            np.dot(centered_shorter.astype(np.float64), centered_aligned.astype(np.float64))
            / denominator
        )
    normalized_rms_error = float(
        rms(aligned - shorter) / max(rms(shorter), 1e-9)
    )
    return normalized_correlation, normalized_rms_error, offset, reference_is_shorter


def _dominant_active_region(
    samples: np.ndarray,
    sample_rate_hz: int,
    *,
    analysis_window_ms: float = 25.0,
    edge_padding_ms: float = 75.0,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise AssertionError("Expected non-empty samples")

    window_samples = max(8, int(round((analysis_window_ms / 1000.0) * sample_rate_hz)))
    power = np.square(samples.astype(np.float64))
    window = np.ones(window_samples, dtype=np.float64) / float(window_samples)
    rolling_rms = np.sqrt(np.convolve(power, window, mode="same"))

    peak_rms = float(np.max(rolling_rms))
    noise_floor_rms = float(np.percentile(rolling_rms, 60))
    threshold_rms = max(noise_floor_rms * 3.0, peak_rms * 0.2, 1e-5)
    active_indices = np.flatnonzero(rolling_rms >= threshold_rms)
    if active_indices.size == 0:
        return samples.copy()

    gap_tolerance_samples = max(1, int(round(0.15 * sample_rate_hz)))
    segment_start = int(active_indices[0])
    segment_end = int(active_indices[0])
    best_start = segment_start
    best_end = segment_end
    best_energy = float(np.sum(power[segment_start : segment_end + 1]))

    for index in active_indices[1:]:
        current_index = int(index)
        if current_index - segment_end <= gap_tolerance_samples:
            segment_end = current_index
            continue

        current_energy = float(np.sum(power[segment_start : segment_end + 1]))
        if current_energy > best_energy:
            best_start = segment_start
            best_end = segment_end
            best_energy = current_energy
        segment_start = current_index
        segment_end = current_index

    current_energy = float(np.sum(power[segment_start : segment_end + 1]))
    if current_energy > best_energy:
        best_start = segment_start
        best_end = segment_end

    edge_padding_samples = max(1, int(round((edge_padding_ms / 1000.0) * sample_rate_hz)))
    trimmed_start = max(0, best_start - edge_padding_samples)
    trimmed_end = min(samples.size, best_end + edge_padding_samples)
    return samples[trimmed_start:trimmed_end].copy()


def _longest_internal_quiet_gap_seconds(
    samples: np.ndarray,
    sample_rate_hz: int,
    *,
    analysis_window_ms: float = 25.0,
) -> float:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return 0.0

    window_samples = max(8, int(round((analysis_window_ms / 1000.0) * sample_rate_hz)))
    power = np.square(samples.astype(np.float64))
    window = np.ones(window_samples, dtype=np.float64) / float(window_samples)
    rolling_rms = np.sqrt(np.convolve(power, window, mode="same"))

    peak_rms = float(np.max(rolling_rms))
    if peak_rms <= 1e-8:
        return float(samples.size) / float(sample_rate_hz)

    quiet_threshold_rms = max(peak_rms * 0.08, 1e-6)
    quiet_mask = rolling_rms < quiet_threshold_rms
    if quiet_mask.size <= 2:
        return 0.0

    # Ignore leading/trailing quiet padding and only measure holes inside the
    # already-isolated active region.
    active_indices = np.flatnonzero(~quiet_mask)
    if active_indices.size == 0:
        return float(samples.size) / float(sample_rate_hz)
    interior_quiet = quiet_mask[active_indices[0] : active_indices[-1] + 1]

    longest_quiet_run = 0
    current_quiet_run = 0
    for is_quiet in interior_quiet:
        if bool(is_quiet):
            current_quiet_run += 1
            longest_quiet_run = max(longest_quiet_run, current_quiet_run)
        else:
            current_quiet_run = 0
    return float(longest_quiet_run) / float(sample_rate_hz)


async def _reconstruct_expected_classification_signal(
    fusion: FusionNode,
    detection: dict,
    *,
    sample_rate_hz: int,
    classification_window_seconds: float,
) -> np.ndarray:
    selected_sensor_ids = [str(sensor_id) for sensor_id in detection["source_sensors"]]
    classification_windows = await fusion.buffer.get_synchronized_window_ending_at(
        sensor_ids=selected_sensor_ids,
        end_time_ns=int(detection["timestamp_ns"]),
        window_seconds=classification_window_seconds,
        sample_rate_hz=sample_rate_hz,
    )
    assert classification_windows

    reference_sensor = str(detection["reference_sensor"])
    classification_path = str(detection["feature_summary"]["classification_path"])
    if classification_path == "omni":
        expected_signal = classification_windows[reference_sensor]
    elif classification_path.startswith("beamformed:"):
        selected_positions = {
            sensor_id: _sensor_positions()[sensor_id]
            for sensor_id in selected_sensor_ids
        }
        localization_position_m = tuple(float(value) for value in detection["position_m"])
        expected_signal = fusion.beamformer.beamform(
            selected_positions,
            classification_windows,
            sample_rate_hz,
            localization_position_m,
            fusion.environment_provider.get_speed_of_sound(localization_position_m),
        )
    else:
        raise AssertionError(f"Unsupported classification path: {classification_path}")

    if fusion.classification_preprocessor is not None:
        expected_signal = fusion.classification_preprocessor.process(
            expected_signal,
            sample_rate_hz,
        )
    return np.asarray(expected_signal, dtype=np.float32)


def test_house_finch_fixture_classifies_with_birdnet(
    birdnet_classifier: BirdNETClassifier,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
) -> None:
    samples, sample_rate_hz = house_finch_fixture_48khz
    result = birdnet_classifier.classify(samples, sample_rate_hz)

    assert result.label == HOUSE_FINCH_LABEL
    assert result.confidence >= 0.05
    assert result.features["model"] == "birdnet_v2m4"


@pytest.mark.parametrize("sample_rate_hz", [16_000, 48_000])
def test_synthesized_house_finch_localizes_with_tight_srp(
    house_finch_fixture_48khz: tuple[np.ndarray, int],
    sample_rate_hz: int,
) -> None:
    signal = _resampled_fixture(house_finch_fixture_48khz, sample_rate_hz)
    windows = {
        sensor_id: channel
        for sensor_id, channel in zip(
            _sensor_positions().keys(),
            synthesize_delayed_array_channels(
                signal,
                sample_rate_hz,
                source_position_m=HOUSE_FINCH_SOURCE_POSITION_M,
            ),
            strict=True,
        )
    }

    settings = hybrid_production_settings(
        localization_srp_grid_resolution_m=TIGHT_SRP_GRID_RESOLUTION_M,
        localization_search_padding_m=TIGHT_SRP_SEARCH_PADDING_M,
        model_chain_config_path=Path("missing_model_chain.json"),
    )
    localizer = build_localizer_from_settings(settings)
    result = localizer.localize(
        sensor_positions=_sensor_positions(),
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )

    # The tight array can only recover bearing, not range (see
    # TIGHT_BEARING_MAX_ERROR_DEG), so assert direction-of-arrival accuracy.
    assert _bearing_error_deg(result.position_m) < TIGHT_BEARING_MAX_ERROR_DEG
    assert result.attempted_algorithm == "srp_phat"
    assert result.resolved_algorithm == "srp_phat"


@pytest.mark.parametrize("sample_rate_hz", [16_000, 48_000])
def test_synthesized_house_finch_beamforming_recovers_source_waveform(
    house_finch_fixture_48khz: tuple[np.ndarray, int],
    sample_rate_hz: int,
) -> None:
    source_signal = _resampled_fixture(house_finch_fixture_48khz, sample_rate_hz)
    sensor_positions = _sensor_positions()
    sensor_windows = {
        sensor_id: channel
        for sensor_id, channel in zip(
            sensor_positions.keys(),
            synthesize_delayed_array_channels(
                source_signal,
                sample_rate_hz,
                source_position_m=HOUSE_FINCH_SOURCE_POSITION_M,
            ),
            strict=True,
        )
    }

    # Use FFT delay-and-sum to avoid interpolation artifacts in this fidelity test.
    beamformer = create_beamformer("freq_domain_das")
    beamformed_signal = beamformer.beamform(
        sensor_positions=sensor_positions,
        sensor_windows=sensor_windows,
        sample_rate_hz=sample_rate_hz,
        steer_position_m=HOUSE_FINCH_SOURCE_POSITION_M,
        sound_speed_mps=343.2,
    )

    source_active_region = _dominant_active_region(source_signal, sample_rate_hz)
    beamformed_active_region = _dominant_active_region(beamformed_signal, sample_rate_hz)
    correlation, normalized_error, _, _ = _best_contiguous_subsegment_metrics(
        source_active_region,
        beamformed_active_region,
    )

    assert _longest_internal_quiet_gap_seconds(beamformed_active_region, sample_rate_hz) <= 0.15
    assert correlation >= 0.80
    assert normalized_error <= 0.55


@pytest.mark.parametrize("sample_rate_hz", [16_000, 48_000])
def test_house_finch_fixture_expands_to_four_sensor_wavs_with_expected_relative_delays(
    tmp_path: Path,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
    sample_rate_hz: int,
) -> None:
    sensor_paths = _synthesized_sensor_wav_paths(
        tmp_path,
        house_finch_fixture_48khz,
        sample_rate_hz=sample_rate_hz,
        classification_window_seconds=30.0,
        leading_padding_seconds=24.0,
    )

    assert sorted(sensor_paths.keys()) == [
        "sirith-house-finch_ch0",
        "sirith-house-finch_ch1",
        "sirith-house-finch_ch2",
        "sirith-house-finch_ch3",
    ]
    assert all(path.exists() for path in sensor_paths.values())

    windows, loaded_sample_rate_hz = load_sensor_wav_files(sensor_paths)
    assert loaded_sample_rate_hz == sample_rate_hz

    stacked_windows = np.stack(
        [windows[f"sirith-house-finch_ch{index}"] for index in range(len(SIRITH_TETRA_SENSOR_OFFSETS_M))],
        axis=0,
    )
    assert stacked_windows.shape[0] == 4
    assert not np.allclose(stacked_windows[0], stacked_windows[1])

    expected_delays_s = geometric_array_propagation_delays_s(
        source_position_m=HOUSE_FINCH_SOURCE_POSITION_M,
        sensor_offsets_m=SIRITH_TETRA_SENSOR_OFFSETS_M,
    )
    expected_relative_samples = np.round(
        (expected_delays_s - expected_delays_s.min()) * sample_rate_hz
    ).astype(np.int64)
    earliest_sensor_index = int(np.argmin(expected_delays_s))
    observed_relative_samples = np.asarray(
        [
            _estimate_integer_sample_lag(
                stacked_windows[earliest_sensor_index],
                stacked_windows[index],
            )
            for index in range(stacked_windows.shape[0])
        ],
        dtype=np.int64,
    )
    assert np.max(np.abs(observed_relative_samples - expected_relative_samples)) <= 1

    result = build_localizer_from_settings(
        hybrid_production_settings(
            localization_srp_grid_resolution_m=TIGHT_SRP_GRID_RESOLUTION_M,
            localization_search_padding_m=TIGHT_SRP_SEARCH_PADDING_M,
            model_chain_config_path=Path("missing_model_chain.json"),
        )
    ).localize(
        sensor_positions=_sensor_positions(),
        sensor_windows={
            sensor_id.replace("_", ":"): windows[sensor_id]
            for sensor_id in sensor_paths
        },
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )
    # The per-sensor delays above are recovered exactly; the tight array can then
    # only resolve bearing from them, not range (see TIGHT_BEARING_MAX_ERROR_DEG).
    assert _bearing_error_deg(result.position_m) < TIGHT_BEARING_MAX_ERROR_DEG
    assert result.resolved_algorithm == "srp_phat"


@pytest.mark.asyncio
async def test_birdnet_hybrid_production_detects_localized_house_finch(
    tmp_path: Path,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
) -> None:
    fusion, storage, settings = await _start_fusion_with_profile(tmp_path, sample_rate_hz=16_000)
    try:
        triggered_frames = await _stream_fixture_into_fusion(
            fusion,
            sample_rate_hz=16_000,
            frame_samples=1024,
            fixture=house_finch_fixture_48khz,
            classification_window_seconds=settings.classification_window_seconds,
            start_time_ns=_align_start_time_to_birdnet_chunk(1_739_910_000_000_000_000, settings),
        )
        assert triggered_frames >= 1

        detection = await _wait_for_localized_house_finch(storage)
        assert detection["label"] == HOUSE_FINCH_LABEL
        assert detection["reporting_modality"] == "localized"
        assert detection["feature_summary"]["capability_tier"] == "full_3d"
        assert detection["feature_summary"]["localization_method"] == "srp_phat"
        assert detection["label_confidence"] >= settings.birdnet_trigger_min_confidence

        # Requirement: detections carry lat/lon derived from the localized position.
        assert detection["position_geo"] is not None
        coord_frame = LocalCoordinateFrame(origin=DEFAULT_SITE_ORIGIN, mode=settings.coordinate_mode)
        expected_geo = coord_frame.local_to_geo(HOUSE_FINCH_SOURCE_POSITION_M)
        assert detection["position_geo"]["lat"] == pytest.approx(expected_geo.lat, abs=1e-3)
        assert detection["position_geo"]["lon"] == pytest.approx(expected_geo.lon, abs=1e-3)

        # Requirement: no audio written to disk when snippet_retention_seconds=0.
        assert detection["snippet_path"] is None
        assert list(Path(settings.snippet_dir).rglob("*.wav")) == []
    finally:
        await fusion.stop()
        await storage.close()


@pytest.mark.asyncio
async def test_birdnet_hybrid_production_detects_house_finch_at_native_48k(
    tmp_path: Path,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
) -> None:
    fusion, storage, settings = await _start_fusion_with_profile(tmp_path, sample_rate_hz=48_000)
    try:
        triggered_frames = await _stream_fixture_into_fusion(
            fusion,
            sample_rate_hz=48_000,
            frame_samples=8192,
            fixture=house_finch_fixture_48khz,
            classification_window_seconds=settings.classification_window_seconds,
            start_time_ns=_align_start_time_to_birdnet_chunk(1_739_920_000_000_000_000, settings),
        )
        assert triggered_frames >= 1

        detection = await _wait_for_localized_house_finch(storage)
        assert detection["label"] == HOUSE_FINCH_LABEL
        assert detection["reporting_modality"] == "localized"
        assert detection["feature_summary"]["capability_tier"] == "full_3d"
        assert detection["feature_summary"]["localization_method"] == "srp_phat"
        assert detection["label_confidence"] >= settings.birdnet_trigger_min_confidence

        # Requirement: detections carry lat/lon derived from the localized position.
        assert detection["position_geo"] is not None
        coord_frame = LocalCoordinateFrame(origin=DEFAULT_SITE_ORIGIN, mode=settings.coordinate_mode)
        expected_geo = coord_frame.local_to_geo(HOUSE_FINCH_SOURCE_POSITION_M)
        assert detection["position_geo"]["lat"] == pytest.approx(expected_geo.lat, abs=1e-3)
        assert detection["position_geo"]["lon"] == pytest.approx(expected_geo.lon, abs=1e-3)

        # Requirement: no audio written to disk when snippet_retention_seconds=0.
        assert detection["snippet_path"] is None
        assert list(Path(settings.snippet_dir).rglob("*.wav")) == []
    finally:
        await fusion.stop()
        await storage.close()


@pytest.mark.asyncio
async def test_birdnet_hybrid_production_python_uses_reported_environment(
    tmp_path: Path,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
) -> None:
    sample_rate_hz = 16_000
    reported_temperature_c = 2.0
    reported_humidity_fraction = 0.2
    expected_sound_speed_mps = _speed_of_sound_mps(
        reported_temperature_c,
        reported_humidity_fraction,
    )
    environment = EnvironmentSampleIn(
        temperature_c=reported_temperature_c,
        humidity_fraction=reported_humidity_fraction,
        source="node-telemetry",
    )
    fusion, storage, settings = await _start_fusion_with_profile(
        tmp_path,
        sample_rate_hz=sample_rate_hz,
        localization_srp_grid_resolution_m=TIGHT_SRP_GRID_RESOLUTION_M,
        localization_search_padding_m=TIGHT_SRP_SEARCH_PADDING_M,
    )
    try:
        triggered_frames = await _stream_fixture_into_fusion(
            fusion,
            sample_rate_hz=sample_rate_hz,
            frame_samples=1024,
            fixture=house_finch_fixture_48khz,
            classification_window_seconds=settings.classification_window_seconds,
            start_time_ns=_align_start_time_to_birdnet_chunk(time.time_ns(), settings),
            environment=environment,
            sound_speed_mps=expected_sound_speed_mps,
        )
        assert triggered_frames >= 1

        detection = await _wait_for_localized_house_finch(storage)
        environment_summary = detection["feature_summary"]["environment"]

        assert detection["label"] == HOUSE_FINCH_LABEL
        assert detection["reporting_modality"] == "localized"
        assert environment_summary["temperature_c"] == pytest.approx(reported_temperature_c)
        assert environment_summary["humidity_fraction"] == pytest.approx(reported_humidity_fraction)
        assert environment_summary["metadata"]["source"] == "live"
        assert environment_summary["metadata"]["temperature_available"] is True
        assert environment_summary["metadata"]["humidity_available"] is True
        assert fusion.environment_provider.get_speed_of_sound(HOUSE_FINCH_SOURCE_POSITION_M) == pytest.approx(
            expected_sound_speed_mps,
            rel=1e-6,
        )
    finally:
        await fusion.stop()
        await storage.close()


def test_birdnet_hybrid_production_rust_sidecar_uses_reported_environment(
    tmp_path: Path,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
) -> None:
    sample_rate_hz = 16_000
    reported_temperature_c = 2.0
    reported_humidity_fraction = 0.2
    expected_sound_speed_mps = _speed_of_sound_mps(
        reported_temperature_c,
        reported_humidity_fraction,
    )
    channels = _synthesized_fixture_channels(
        house_finch_fixture_48khz,
        sample_rate_hz=sample_rate_hz,
        sound_speed_mps=expected_sound_speed_mps,
    )
    payload = _binary_ingest_payload(
        node_id="sirith-house-finch",
        frames=[
            _binary_frame(
                channels,
                start_time_ns=1_739_940_000_000_000_000,
                sequence=1,
                start_sample_index=0,
                sample_rate_hz=sample_rate_hz,
                temperature_c=reported_temperature_c,
                humidity_fraction=reported_humidity_fraction,
                source="node-telemetry",
            )
        ],
    )

    with _running_rust_sidecar(tmp_path) as port:
        response = _http_json(
            port,
            "/api/v1/ingest/binary",
            method="POST",
            body=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response["accepted"] is True

        localization_manifest = _wait_for_rust_localization_manifest(port)
        localization = localization_manifest["localization"]
        assert localization is not None
        assert localization["resolved_algorithm"] == "srp_phat"
        assert localization["sound_speed_mps"] == pytest.approx(expected_sound_speed_mps, rel=1e-4)
        assert localization["position_m"] is not None
        assert localization["range_projection_mode"] == "range_bearing_projected"
        assert _bearing_error_deg(localization["position_m"]) < TIGHT_BEARING_MAX_ERROR_DEG
        assert localization_manifest["classifier_render"] is not None
        assert localization_manifest["classifier_render"]["sample_count"] > 0

        status = _http_json(port, "/api/v1/dsp/status")
        assert status["total_localization_results"] >= 1
        assert status["total_classifier_renders"] >= 1


@pytest.mark.asyncio
async def test_birdnet_hybrid_production_persists_contiguous_classification_audio_snippet(
    tmp_path: Path,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
) -> None:
    sample_rate_hz = 16_000
    fusion, storage, settings = await _start_fusion_with_profile(
        tmp_path,
        sample_rate_hz=sample_rate_hz,
        snippet_retention_seconds=3600,
    )
    try:
        triggered_frames = await _stream_fixture_into_fusion(
            fusion,
            sample_rate_hz=sample_rate_hz,
            frame_samples=1024,
            fixture=house_finch_fixture_48khz,
            classification_window_seconds=settings.classification_window_seconds,
            start_time_ns=_align_start_time_to_birdnet_chunk(1_739_930_000_000_000_000, settings),
        )
        assert triggered_frames >= 1

        detection = await _wait_for_localized_house_finch(storage)
        assert detection["snippet_path"] is not None

        snippet_path = Path(detection["snippet_path"])
        assert snippet_path.exists()

        snippet_signal, snippet_sample_rate_hz = read_wav_mono(snippet_path)
        assert snippet_sample_rate_hz == sample_rate_hz
        assert snippet_signal.size > 0

        expected_signal = await _reconstruct_expected_classification_signal(
            fusion,
            detection,
            sample_rate_hz=sample_rate_hz,
            classification_window_seconds=settings.classification_window_seconds,
        )
        expected_correlation, expected_error, _, _ = _best_contiguous_subsegment_metrics(
            expected_signal,
            snippet_signal,
        )
        assert expected_correlation >= 0.995
        assert expected_error <= 0.02

        # The attached snippet may include front padding and classifier-oriented
        # conditioning, so compare the dominant active region rather than the
        # entire stored clip. This still catches obvious gaps or severe damage.
        fixture_signal = _dominant_active_region(
            _resampled_fixture(house_finch_fixture_48khz, sample_rate_hz),
            sample_rate_hz,
        )
        snippet_active_region = _dominant_active_region(
            snippet_signal,
            sample_rate_hz,
        )
        fixture_correlation, _, _, _ = _best_contiguous_subsegment_metrics(
            fixture_signal,
            snippet_active_region,
        )
        assert _longest_internal_quiet_gap_seconds(snippet_active_region, sample_rate_hz) <= 0.15
        # This integration test validates contiguous snippet persistence and signal identity against
        # the exact classification input reconstruction. Beamformer fidelity against the true source
        # waveform is covered by test_synthesized_house_finch_beamforming_recovers_source_waveform.
        # The dominant-region comparison against the dry fixture is a
        # coarse sanity check. Keep it permissive because classification
        # input conditioning and event anchoring can shift this value even
        # when the exact reconstruction checks above already prove snippet
        # continuity and identity.
        assert fixture_correlation >= 0.15
    finally:
        await fusion.stop()
        await storage.close()


@pytest.mark.parametrize("sample_rate_hz", [16_000, 48_000])
def test_synthesized_house_finch_localizes_tightly_in_geodetic_space(
    house_finch_fixture_48khz: tuple[np.ndarray, int],
    sample_rate_hz: int,
) -> None:
    signal = _resampled_fixture(house_finch_fixture_48khz, sample_rate_hz)
    windows = {
        sensor_id: channel
        for sensor_id, channel in zip(
            _sensor_positions().keys(),
            synthesize_delayed_array_channels(
                signal,
                sample_rate_hz,
                source_position_m=HOUSE_FINCH_SOURCE_POSITION_M,
            ),
            strict=True,
        )
    }
    settings = hybrid_production_settings(
        localization_srp_grid_resolution_m=TIGHT_SRP_GRID_RESOLUTION_M,
        localization_search_padding_m=TIGHT_SRP_SEARCH_PADDING_M,
        site_origin_lat=DEFAULT_SITE_ORIGIN.lat,
        site_origin_lon=DEFAULT_SITE_ORIGIN.lon,
        site_origin_alt_m=DEFAULT_SITE_ORIGIN.alt_m,
        coordinate_mode="geodetic",
        model_chain_config_path=Path("missing_model_chain.json"),
    )
    localizer = build_localizer_from_settings(settings)
    coordinate_frame = LocalCoordinateFrame(origin=DEFAULT_SITE_ORIGIN, mode="geodetic")
    result = localizer.localize(
        sensor_positions=_sensor_positions(),
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )

    estimated_geo = coordinate_frame.local_to_geo(result.position_m)
    # The tight array recovers bearing but not range (see TIGHT_BEARING_MAX_ERROR_DEG);
    # assert direction-of-arrival accuracy rather than absolute position.
    assert _bearing_error_deg(result.position_m) < TIGHT_BEARING_MAX_ERROR_DEG
    # The geodetic projection must round-trip the local estimate exactly so that
    # downstream consumers can move between frames without drift.
    np.testing.assert_allclose(
        coordinate_frame.geo_to_local(estimated_geo),
        np.asarray(result.position_m, dtype=np.float64),
        rtol=1e-6,
        atol=1e-3,
    )


@pytest.mark.asyncio
async def test_birdnet_hybrid_production_detection_timestamp_matches_packet_time(
    tmp_path: Path,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
) -> None:
    """Detection.timestamp_ns must be anchored to the firmware packet time,
    not the server wall clock.  The packet epoch used here is ~Feb 2025,
    over a year in the past; if server time leaked in, the assert would fail."""
    PACKET_EPOCH_NS = 1_739_910_000_000_000_000  # 2025-02-19 ~01:00 UTC
    TOLERANCE_NS = 120 * 10 ** 9  # ±2 min around packet window

    fusion, storage, settings = await _start_fusion_with_profile(tmp_path, sample_rate_hz=16_000)
    try:
        triggered = await _stream_fixture_into_fusion(
            fusion,
            sample_rate_hz=16_000,
            frame_samples=1024,
            fixture=house_finch_fixture_48khz,
            classification_window_seconds=settings.classification_window_seconds,
            start_time_ns=_align_start_time_to_birdnet_chunk(PACKET_EPOCH_NS, settings),
        )
        assert triggered >= 1
        detection = await _wait_for_localized_house_finch(storage)

        # Timestamp must be near the packet epoch, not near the server clock.
        assert abs(detection["timestamp_ns"] - PACKET_EPOCH_NS) < TOLERANCE_NS
        # Server time is >1 year ahead of the packet epoch; gap must be large.
        assert abs(detection["timestamp_ns"] - time.time_ns()) > 300 * 24 * 3600 * 10 ** 9

        # TDOA dict must be populated — confirms cross-correlation ran on the
        # packet-time-anchored audio windows, not a degenerate stub.
        assert isinstance(detection["tdoa_s"], dict)
        assert len(detection["tdoa_s"]) >= 1
    finally:
        await fusion.stop()
        await storage.close()


@pytest.mark.asyncio
async def test_birdnet_hybrid_production_no_audio_written_to_disk_without_retention(
    tmp_path: Path,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
) -> None:
    """Streaming audio must remain in-memory (MultiSensorBuffer) until a
    detection triggers a snippet write.  With snippet_retention_seconds=0
    (default) no .wav files should ever appear in the snippet directory."""
    fusion, storage, settings = await _start_fusion_with_profile(
        tmp_path, sample_rate_hz=16_000, snippet_retention_seconds=0
    )
    try:
        triggered = await _stream_fixture_into_fusion(
            fusion,
            sample_rate_hz=16_000,
            frame_samples=1024,
            fixture=house_finch_fixture_48khz,
            classification_window_seconds=settings.classification_window_seconds,
            start_time_ns=_align_start_time_to_birdnet_chunk(1_739_915_000_000_000_000, settings),
        )
        assert triggered >= 1
        detection = await _wait_for_localized_house_finch(storage)

        assert detection["snippet_path"] is None
        # Synthesis fixture wavs are written outside snippet_dir; the snippet
        # directory itself must be empty.
        assert list(Path(settings.snippet_dir).rglob("*.wav")) == []
    finally:
        await fusion.stop()
        await storage.close()


def test_birdnet_hybrid_production_localization_bandpass_configured_for_bird_frequencies(
    tmp_path: Path,
) -> None:
    """The birdnet_hybrid_production profile must apply a bandpass to the
    localization path so that only frequencies resolvable at the sensor
    spacing contribute to TDOA estimation (bird-call range: 300–3500 Hz).
    Frequencies above ~1715 Hz alias given the ~0.1 m sensor spacing at
    343 m/s, but lower bird harmonics still anchor the cross-correlation."""
    settings = _build_profile_settings(tmp_path, sample_rate_hz=16_000)

    assert settings.localization_band_min_hz == pytest.approx(300.0)
    assert settings.localization_band_max_hz == pytest.approx(3500.0)

    loc_config = settings.localization_config()
    preprocessor = create_localization_preprocessor(loc_config)
    assert preprocessor is not None

    # Verify the filter actually attenuates out-of-band content.
    sample_rate_hz = 16_000
    t = np.linspace(0.0, 0.5, int(0.5 * sample_rate_hz), endpoint=False, dtype=np.float32)
    out_of_band = np.sin(2 * np.pi * 6000.0 * t)  # 6 kHz — above 3500 Hz cutoff
    in_band = np.sin(2 * np.pi * 1000.0 * t)       # 1 kHz — well within band

    filtered_oob = np.asarray(preprocessor.process(out_of_band, sample_rate_hz), dtype=np.float32)
    filtered_inb = np.asarray(preprocessor.process(in_band, sample_rate_hz), dtype=np.float32)

    # Out-of-band energy must be substantially suppressed (< 10 % of original).
    assert rms(filtered_oob) < rms(out_of_band) * 0.10
    # In-band energy must be largely preserved (> 70 % of original).
    assert rms(filtered_inb) > rms(in_band) * 0.70


@pytest.mark.asyncio
async def test_birdnet_hybrid_production_ingest_not_blocked_by_slow_live_callback(
    tmp_path: Path,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
) -> None:
    """FusionNode.ingest() must return quickly even when the live_callback
    (the WebSocket/UI delivery path) introduces multi-second latency.
    The pipeline is queue-based; ingest() only enqueues to the localization
    queue and must never await downstream workers or the callback."""
    callback_latency_s = 1.5
    call_count = 0

    async def slow_callback(payload: object) -> None:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(callback_latency_s)

    settings = _build_profile_settings(tmp_path, sample_rate_hz=16_000)
    storage = Storage(settings.db_path)
    await storage.initialize()
    coordinate_frame = LocalCoordinateFrame(origin=DEFAULT_SITE_ORIGIN, mode=settings.coordinate_mode)
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=build_localizer_from_settings(settings),
        classifier=create_classifier(settings),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=slow_callback,
        coordinate_frame=coordinate_frame,
        zone_matcher=ZoneMatcher(storage=storage),
        environment_provider=LiveEnvironmentProvider(
            fallback_temperature_c=settings.default_temperature_c,
            fallback_humidity_fraction=settings.default_humidity,
        ),
    )
    await fusion.start()
    try:
        sample_rate_hz = 16_000
        frame_samples = 1024
        start_time_ns = _align_start_time_to_birdnet_chunk(1_739_925_000_000_000_000, settings)
        leading_padding_seconds = max(
            settings.localization_window_seconds,
            max(
                settings.localization_window_seconds,
                settings.classification_window_seconds - settings.birdnet_chunk_overlap_seconds,
            ) - BIRDNET_TEST_TARGET_CONTEXT_SECONDS,
        )
        sensor_paths = _synthesized_sensor_wav_paths(
            tmp_path,
            house_finch_fixture_48khz,
            sample_rate_hz=sample_rate_hz,
            classification_window_seconds=settings.classification_window_seconds,
            leading_padding_seconds=leading_padding_seconds,
        )
        windows, _ = load_sensor_wav_files(sensor_paths)
        channels = np.stack(
            [windows[f"sirith-house-finch_ch{i}"] for i in range(len(SIRITH_TETRA_SENSOR_OFFSETS_M))],
            axis=0,
        )

        max_ingest_latency_s = 0.0
        for seq, (frame_start_ns, frame) in enumerate(
            split_channels_into_frames(
                channels,
                sample_rate_hz=sample_rate_hz,
                start_time_ns=start_time_ns,
                frame_samples=frame_samples,
            ),
            start=1,
        ):
            t0 = asyncio.get_event_loop().time()
            response = await fusion.ingest(
                IngestFrameRequest(
                    node=_node_spec(),
                    frame={
                        "start_time_ns": frame_start_ns,
                        "sample_rate_hz": sample_rate_hz,
                        "channels": int(frame.shape[0]),
                        "encoding": "pcm16le",
                        "samples_b64": encode_pcm16le_b64(frame),
                        "sequence": seq,
                    },
                )
            )
            assert response.accepted is True
            elapsed = asyncio.get_event_loop().time() - t0
            max_ingest_latency_s = max(max_ingest_latency_s, elapsed)

        # Each ingest() call must complete well under the callback latency.
        assert max_ingest_latency_s < 0.200, (
            f"ingest() took {max_ingest_latency_s:.3f}s — likely blocking on live_callback"
        )
    finally:
        await fusion.stop()
        await storage.close()
