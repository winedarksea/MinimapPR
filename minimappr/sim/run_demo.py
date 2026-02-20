"""Realtime two-node demo stream: point node + Sirith tetra node."""

from __future__ import annotations

import argparse
import asyncio
import math
import time
from dataclasses import dataclass

import httpx
import numpy as np

from minimappr.models import NodeSpec, NodeType
from minimappr.sim.geometry import regular_tetrahedron_offsets_m
from minimappr.utils.audio import encode_pcm16le_b64


@dataclass(slots=True)
class DemoConfig:
    server: str
    sample_rate_hz: int
    frame_size: int
    duration_seconds: float


def parse_vec3(raw: str) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected x,y,z")
    return parts[0], parts[1], parts[2]


def source_position(times_s: np.ndarray) -> np.ndarray:
    x = 2.8 + 1.9 * np.sin(2.0 * np.pi * times_s / 8.5)
    y = 3.2 + 1.6 * np.cos(2.0 * np.pi * times_s / 11.0)
    z = 1.5 + 0.4 * np.sin(2.0 * np.pi * times_s / 6.0)
    return np.stack([x, y, z], axis=1)


def source_waveform(times_s: np.ndarray) -> np.ndarray:
    phase = np.mod(times_s, 4.0)
    out = np.zeros_like(times_s)

    # 0-1s: high-frequency chirped burst (bird-like)
    seg = (phase >= 0.0) & (phase < 1.0)
    t = phase[seg]
    env = np.sin(np.pi * t) ** 2
    freq = 1800.0 + (1200.0 * t)
    out[seg] = 0.5 * env * np.sin(2.0 * np.pi * freq * times_s[seg])

    # 1-2s: low machine hum
    seg = (phase >= 1.0) & (phase < 2.0)
    t = phase[seg] - 1.0
    out[seg] = 0.35 * np.sin(2.0 * np.pi * 220.0 * times_s[seg]) * (0.8 + 0.2 * np.sin(2.0 * np.pi * 2.0 * t))

    # 2-3s: short impulses
    seg = (phase >= 2.0) & (phase < 3.0)
    t = phase[seg]
    burst = np.zeros_like(t)
    for center in (2.08, 2.32, 2.56, 2.83):
        burst += np.exp(-((t - center) ** 2) / (2 * (0.006 ** 2)))
    out[seg] = 0.9 * np.clip(burst, 0.0, 1.0)

    # 3-4s: speech-like harmonic stack
    seg = (phase >= 3.0) & (phase < 4.0)
    t = phase[seg] - 3.0
    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)
    speech = (
        0.45 * np.sin(2.0 * np.pi * 180.0 * times_s[seg])
        + 0.22 * np.sin(2.0 * np.pi * 360.0 * times_s[seg])
        + 0.15 * np.sin(2.0 * np.pi * 540.0 * times_s[seg])
    )
    out[seg] = envelope * speech

    return out


def render_sensor_frame(
    sensor_position_m: np.ndarray,
    frame_start_s: float,
    frame_size: int,
    sample_rate_hz: int,
    sound_speed_mps: float,
    noise_std: float,
) -> np.ndarray:
    t = frame_start_s + (np.arange(frame_size, dtype=np.float64) / float(sample_rate_hz))
    src_pos = source_position(t)
    distances = np.linalg.norm(src_pos - sensor_position_m[None, :], axis=1)
    delayed_t = t - (distances / sound_speed_mps)

    signal = source_waveform(delayed_t)
    attenuation = 1.0 / (1.0 + 0.4 * distances)
    audio = attenuation * signal
    audio += np.random.normal(0.0, noise_std, size=audio.shape)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


async def run_demo(
    cfg: DemoConfig,
    point_position_m: tuple[float, float, float],
    tetra_position_m: tuple[float, float, float],
    tetra_edge_m: float,
) -> None:
    point_node = NodeSpec(
        id="point-node-01",
        node_type=NodeType.POINT,
        position_m=point_position_m,
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio", "gps_pps"],
        metadata={"hw": "simulated_esp32_point"},
    )

    tetra_offsets = regular_tetrahedron_offsets_m(edge_m=tetra_edge_m)
    tetra_node = NodeSpec(
        id="sirith-tetra-01",
        node_type=NodeType.SIRITH_TETRA,
        position_m=tetra_position_m,
        sensor_offsets_m=tetra_offsets,
        capabilities=["audio", "array_localization"],
        metadata={"hw": "simulated_sirith_tetra"},
    )

    point_sensor = np.asarray(point_node.position_m, dtype=np.float64)
    tetra_sensors = [
        np.asarray(tetra_node.position_m, dtype=np.float64) + np.asarray(offset, dtype=np.float64)
        for offset in tetra_node.sensor_offsets_m
    ]

    frame_ns = int((cfg.frame_size / cfg.sample_rate_hz) * 1_000_000_000)
    wall_start = time.perf_counter()
    stream_time_ns = time.time_ns()

    async with httpx.AsyncClient(timeout=3.0) as client:
        frame_idx = 0
        while True:
            if cfg.duration_seconds > 0:
                elapsed = frame_idx * (cfg.frame_size / cfg.sample_rate_hz)
                if elapsed > cfg.duration_seconds:
                    break

            frame_start_s = frame_idx * (cfg.frame_size / cfg.sample_rate_hz)

            point_audio = render_sensor_frame(
                sensor_position_m=point_sensor,
                frame_start_s=frame_start_s,
                frame_size=cfg.frame_size,
                sample_rate_hz=cfg.sample_rate_hz,
                sound_speed_mps=343.2,
                noise_std=0.01,
            )[None, :]

            tetra_audio = np.stack(
                [
                    render_sensor_frame(
                        sensor_position_m=sensor,
                        frame_start_s=frame_start_s,
                        frame_size=cfg.frame_size,
                        sample_rate_hz=cfg.sample_rate_hz,
                        sound_speed_mps=343.2,
                        noise_std=0.01,
                    )
                    for sensor in tetra_sensors
                ],
                axis=0,
            )

            frame_start_ns = stream_time_ns + frame_idx * frame_ns

            point_payload = {
                "node": point_node.model_dump(),
                "frame": {
                    "start_time_ns": frame_start_ns,
                    "sample_rate_hz": cfg.sample_rate_hz,
                    "channels": 1,
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(point_audio),
                    "sequence": frame_idx,
                },
            }
            tetra_payload = {
                "node": tetra_node.model_dump(),
                "frame": {
                    "start_time_ns": frame_start_ns,
                    "sample_rate_hz": cfg.sample_rate_hz,
                    "channels": 4,
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(tetra_audio),
                    "sequence": frame_idx,
                },
            }

            url = f"{cfg.server.rstrip('/')}/api/v1/ingest/frame"
            results = await asyncio.gather(
                client.post(url, json=point_payload),
                client.post(url, json=tetra_payload),
                return_exceptions=True,
            )

            if frame_idx % max(1, int(math.ceil(cfg.sample_rate_hz / cfg.frame_size))) == 0:
                point_status = results[0].status_code if isinstance(results[0], httpx.Response) else "ERR"
                tetra_status = results[1].status_code if isinstance(results[1], httpx.Response) else "ERR"
                print(
                    f"frame={frame_idx:05d} point={point_status} tetra={tetra_status} "
                    f"t={frame_start_s:7.2f}s"
                )

            frame_idx += 1
            target = wall_start + frame_idx * (cfg.frame_size / cfg.sample_rate_hz)
            await asyncio.sleep(max(0.0, target - time.perf_counter()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MinimapPR two-node localization demo stream")
    parser.add_argument("--server", default="http://127.0.0.1:8080", help="Backend server base URL")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--frame-size", type=int, default=1024, help="Samples per frame")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Duration in seconds (0 means run forever)",
    )
    parser.add_argument(
        "--point-position",
        default="0,0,2",
        help="Point node XYZ in meters",
    )
    parser.add_argument(
        "--tetra-position",
        default="6,0,2",
        help="Sirith tetra node XYZ in meters",
    )
    parser.add_argument(
        "--tetra-edge",
        type=float,
        default=0.05,
        help="Tetrahedron edge length in meters",
    )

    args = parser.parse_args()

    cfg = DemoConfig(
        server=args.server,
        sample_rate_hz=args.sample_rate,
        frame_size=args.frame_size,
        duration_seconds=args.duration,
    )

    asyncio.run(
        run_demo(
            cfg=cfg,
            point_position_m=parse_vec3(args.point_position),
            tetra_position_m=parse_vec3(args.tetra_position),
            tetra_edge_m=args.tetra_edge,
        )
    )


if __name__ == "__main__":
    main()
