# MinimapPR
Realtime environmental awareness: distributed sound localization + classification + common operating picture.

This repository now includes a complete Phase 1 core build focused on the base case:
- functional two-node ingestion model (point node + Sirith tetrahedral array node)
- TDOA localization backend (GCC-PHAT + least-squares solve)
- canonical event envelope (`event_id`, `source_type`, TOA/TOR, `time_quality`, provenance refs)
- core schema persistence for nodes, observations, detections, tracks, labels, alerts, and track updates
- geographic COP frontend with node health, GDOP overlay, symbology, uncertainty ellipses, velocity vectors, track table, and detection feed

## First Delivery Scope (Implemented)
- `FastAPI` backend for node ingestion, localization, classification, tracking, and APIs
- queue-driven fusion node runtime (ingest stage decoupled from localization/classification workers)
- `SQLite` persistence for Phase 1 core tables (`nodes`, `observations`, `detections`, `tracks`, `labels`, `alerts`, `track_updates`) plus active `environment` ingestion/persistence
- automatic snippet retention cleanup (self-cleaning raw audio extracts)
- snippet serving endpoint (`/api/v1/detections/{id}/audio`)
- node audio debug endpoint for recent per-node listen checks (`/api/v1/nodes/{node_id}/audio/recent`)
- live websocket feed with server-side subscription filtering (zone/category/confidence/track status)
- geographic Leaflet COP dashboard
- deterministic simulation stream with both required node types

## Project Layout
- `minimappr/main.py`: API server + lifecycle
- `minimappr/core/`: buffering, localization, tracking, fusion-node orchestration
- `minimappr/classifiers/`: classifier interface, heuristic backend, optional YAMNet backend
- `minimappr/storage/db.py`: SQLite schema + persistence
- `minimappr/frontend/`: basic web UI
- `minimappr/sim/run_demo.py`: realtime two-node simulator (point + Sirith tetra)
- `firmware/`: shared embedded node runtime + Sirith/point firmware targets
- `tests/`: localization and classifier tests

## Quick Start
```bash
pip install minimappr[full]   # installs birdnet + tensorflow + tensorflow-hub
minimappr                      # starts server at :8080
```

1. Create environment and install dependencies.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start server.
```bash
uvicorn minimappr.main:app --host 0.0.0.0 --port 8080 --reload
```

3. In a second terminal, run two-node live demo stream.
```bash
source .venv/bin/activate
python -m minimappr.sim.run_demo --server http://127.0.0.1:8080
```

4. Open UI.
- `http://127.0.0.1:8080`

You should see nodes appear, detections populate, and tracks move on the map.

## Node Types in MVP
### 1) Point node (ESP32-style stream node)
- Single-channel microphone stream
- Node type: `point`
- Intended for GPS/PPS-timestamped network localization

### 2) Sirith tetrahedral node
- Four-channel array stream
- Node type: `sirith_tetra`
- Default geometry: regular tetrahedron with 50 mm edge (`schematics` kept for hardware reference)
- In this MVP it contributes 4 independent channels to the backend solver

## Firmware (ESP32 + Pico)
Firmware projects are in `firmware/`:
- `firmware/lib/minimap_node_core`: shared node runtime/protocol/transport
- `firmware/lib/minimap_audio_esp32`: ESP32 I2S audio sources
- `firmware/lib/minimap_audio_pico`: RP2040/RP2350 Pico TDM audio sources
- `firmware/nodes/sirith_tetra`: Sirith tetrahedral node firmware (dual-I2S -> 4 channels)
- `firmware/nodes/sirith_tetra_pico`: Sirith tetrahedral Pico W / Pico 2 W firmware (TDM-4)
- `firmware/nodes/point_single_mic`: reference point node firmware

Full firmware setup and build instructions are in `firmware/README.md`.

## Ingestion Protocol (Current)
Endpoint: `POST /api/v1/ingest/frame`

Payload:
```json
{
  "node": {
    "id": "point-node-01",
    "node_type": "point",
    "position_m": [0.0, 0.0, 2.0],
    "sensor_offsets_m": [[0.0, 0.0, 0.0]],
    "capabilities": ["audio", "gps_pps"],
    "metadata": {}
  },
  "frame": {
    "start_time_ns": 1739810000000000000,
    "sample_rate_hz": 16000,
    "channels": 1,
    "encoding": "pcm16le",
    "samples_b64": "...",
    "sequence": 42
  },
  "environment": {
    "temperature_c": 21.4,
    "humidity_fraction": 0.52,
    "pressure_pa": 101325.0,
    "source": "onboard_sensor"
  }
}
```

Notes:
- audio payload is interleaved `pcm16le`, base64 encoded
- `frame.channels` must match `len(node.sensor_offsets_m)`
- timestamps are per-frame start timestamps in `ns`
- optional per-frame timing quality metadata supported: `time_quality`, `toa_ns`, `tor_ns`
- optional environmental payload supported: `environment.temperature_c` (minimum), humidity/pressure/wind/lux optional
- firmware-compatible fallback: if `node.metadata.temperature_c` is provided, it is ingested into `environment` even without an explicit `environment` object
- response `triggered=true` means an event candidate was queued for fusion workers; detection emission is asynchronous

## Processing Pipeline
1. Ingest timestamped audio frames.
2. Append channel streams to rolling per-sensor buffers.
3. Trigger candidate events from frame RMS threshold.
4. Enqueue trigger candidates to fusion workers.
5. Build synchronized multi-sensor windows.
6. Run GCC-PHAT TDOA measurement and nonlinear 3D solve.
7. Classify event audio (heuristic baseline, pluggable backend).
8. Associate/update track.
9. Persist detection + track and emit live websocket event.
10. Save mono snippet for retention window; periodic cleanup removes expired snippets.

## Classifier System
Implemented classifier architecture is pluggable:
- default: `heuristic` (fast baseline labels: `bird_like`, `speech_like`, `impulse`, `machine_hum`, `ambient`, `unknown`)
- optional: `yamnet` (if TensorFlow dependencies are installed)
- optional: `birdnet` (direct bird species classifier, better suited to bird-focused testing with longer clips)

Set backend with:
```bash
export MINIMAPPR_CLASSIFIER=heuristic
# or
export MINIMAPPR_CLASSIFIER=yamnet
# or
export MINIMAPPR_CLASSIFIER=birdnet
```

For bird-focused testing, there is also a bundled runtime preset:

```bash
export MINIMAPPR_RUNTIME_PROFILE=birdnet_omni_testing
```

That preset switches to direct BirdNET, disables beamformed classification, skips localization before classification, and expands the classification clip to a 30 s trailing omni window while keeping the short localization trigger window intact.

For production wildlife deployment, use the hybrid preset:

```bash
export MINIMAPPR_RUNTIME_PROFILE=birdnet_hybrid_production
```

That preset uses direct BirdNET, keeps omnidirectional classification on a 30 s reporting window, runs SRP-PHAT localization on a low-frequency band for the sirith tetrahedral array, and emits one canonical detection per label/reporting window with localized detections preferred over omni-only detections.

## API Endpoints
- `GET /health`
- `GET /api/v1/config`
- `GET /api/v1/fusion/status`
- `GET /api/v1/federation/status`
- `POST /api/v1/ingest/frame`
- `GET /api/v1/nodes`
- `GET /api/v1/detections?limit=100`
- `GET /api/v1/tracks?limit=200&include_standby=false`
- `GET /api/v1/cop/status`
- `GET /api/v1/alerts?limit=100`
- `GET /api/v1/environment?limit=500&node_id=...`
- `GET /api/v1/environment/current?x=...&y=...&z=...`
- `GET /api/v1/detections/{detection_id}/audio`
- `GET /api/v1/nodes/{node_id}/audio/recent?seconds=10`
- `POST /api/v1/federation/heartbeat` (peer-to-peer)
- `POST /api/v1/federation/snapshot` (peer-to-peer)
- `WS /ws/live`

### Audio Path Validation (Developer)
Use node-level debug audio playback when detections are absent and you need to verify that audio ingest is healthy.

1. Open the COP dashboard and use the `Node Audio Debug` panel.
2. Click `Listen` on a node to request the most recent buffered clip.
3. Confirm it sounds reasonable; if not, inspect node health and ingest transport before classifier tuning.

Equivalent API call:

```bash
curl "http://127.0.0.1:8080/api/v1/nodes/http-node-1/audio/recent?seconds=10" --output node_recent.wav
```

## Runtime Configuration
Key env vars:
- `MINIMAPPR_HOST` (default `0.0.0.0`)
- `MINIMAPPR_PORT` (default `8080`)
- `MINIMAPPR_DB_PATH` (default `data/minimappr.db`)
- `MINIMAPPR_SNIPPET_DIR` (default `data/snippets`)
- `MINIMAPPR_SNIPPET_RETENTION_SECONDS` (default `3600`)
- `MINIMAPPR_TRIGGER_RMS` (default `0.015`)
- `MINIMAPPR_TRIGGER_COOLDOWN_SECONDS` (default `0.8`)
- `MINIMAPPR_LOCALIZATION_WINDOW_SECONDS` (default `0.08`)
- `MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS` (default `0`, inherits `MINIMAPPR_LOCALIZATION_WINDOW_SECONDS`)
- `MINIMAPPR_DEFAULT_TEMPERATURE_C` (default `20.0`)
- `MINIMAPPR_DEFAULT_HUMIDITY` (default `0.5`)
- `MINIMAPPR_ENVIRONMENT_READING_MAX_AGE_SECONDS` (default `300.0`, `0` disables staleness cutoff)
- `MINIMAPPR_SITE_ORIGIN_LAT` (default `37.7749`)
- `MINIMAPPR_SITE_ORIGIN_LON` (default `-122.4194`)
- `MINIMAPPR_SITE_ORIGIN_ALT_M` (default `0.0`)
- `MINIMAPPR_COORDINATE_MODE` (`flat` or `geodetic`; default `flat`)
- `MINIMAPPR_RUNTIME_PROFILE` (`default`, `birdnet_omni_testing`, or `birdnet_hybrid_production`)
- `MINIMAPPR_CLASSIFIER` (`heuristic`, `yamnet`, or `birdnet`)
- `MINIMAPPR_DETECTION_MIN_CONFIDENCE` (default `0.05`; hides lower-confidence detections from detection APIs/UI and soundscape rendering)
- `MINIMAPPR_SKIP_LOCALIZATION_FOR_CLASSIFICATION` (`false` default)
- `MINIMAPPR_LOCALIZATION_BAND_MIN_HZ` / `MINIMAPPR_LOCALIZATION_BAND_MAX_HZ` (optional localization-only bandpass)
- `MINIMAPPR_REPORTING_WINDOW_SECONDS` (canonical detection dedupe window; default `30`)
- `MINIMAPPR_TRACKING_FILTER` (`linear` default, or `kalman`)
- `MINIMAPPR_KALMAN_PROCESS_NOISE` (default `2.0`)
- `MINIMAPPR_KALMAN_MEASUREMENT_NOISE` (default `1.5`)
- `MINIMAPPR_KALMAN_INITIAL_POSITION_VARIANCE` (default `4.0`)
- `MINIMAPPR_KALMAN_INITIAL_VELOCITY_VARIANCE` (default `16.0`)
- `MINIMAPPR_FUSION_WORKER_COUNT` (default `1`)
- `MINIMAPPR_FUSION_EVENT_QUEUE_SIZE` (default `256`)
- `MINIMAPPR_NODE_DEGRADED_AFTER_SECONDS` (default `15.0`)
- `MINIMAPPR_NODE_OFFLINE_AFTER_SECONDS` (default `45.0`)
- `MINIMAPPR_EVENT_STALE_SECONDS` (default `30.0`)
- `MINIMAPPR_RETENTION_TRACK_UPDATES_SECONDS` (default `604800`, set `-1` to disable cleanup)
- `MINIMAPPR_RETENTION_ALERTS_SECONDS` (default `2592000`, set `-1` to disable cleanup)
- `MINIMAPPR_RETENTION_ENVIRONMENT_SECONDS` (default `604800`, set `-1` to disable cleanup)
- `MINIMAPPR_RETENTION_DROPPED_TRACKS_SECONDS` (default `604800`, set `-1` to disable cleanup)
- `MINIMAPPR_FEDERATION_ENABLED` (`false` default)
- `MINIMAPPR_FEDERATION_SERVER_ID` (`srv-local` default)
- `MINIMAPPR_FEDERATION_PEERS_CONFIG_PATH` (`data/federation_peers.json` default)
- `MINIMAPPR_FEDERATION_PEERS_JSON` (optional inline JSON peer config override)
- `MINIMAPPR_FEDERATION_AUTH_TOKEN` (optional shared token/fallback peer auth token)
- `MINIMAPPR_FEDERATION_PUBLISH_INTERVAL_SECONDS` (default `1.0`)
- `MINIMAPPR_FEDERATION_HEARTBEAT_INTERVAL_SECONDS` (default `2.0`)
- `MINIMAPPR_FEDERATION_LINK_TIMEOUT_SECONDS` (default `8.0`)
- `MINIMAPPR_FEDERATION_REQUEST_TIMEOUT_SECONDS` (default `2.5`)
- `MINIMAPPR_FEDERATION_TRACK_TTL_SECONDS` (default `20.0`)
- `MINIMAPPR_FEDERATION_DECONFLICT_MAHALANOBIS_GATE` (default `4.5`)
- `MINIMAPPR_FEDERATION_TQI_HYSTERESIS` (default `0.05`)

## Testing
```bash
source .venv/bin/activate
pytest -q
```

5-minute soak harness:
```bash
source .venv/bin/activate
python scripts/run_soak.py --duration 300
```

## Frontend Development (Leptos/WASM)

The operator UI lives in `minimappr-frontend/` (Rust → WASM via Leptos 0.8 + Trunk).
**End users** who `pip install minimappr` get the pre-built WASM bundle — no Rust toolchain needed.
**Contributors** editing the UI need `cargo`, `trunk`, and the `wasm32-unknown-unknown` target.

### One-time contributor setup
```bash
cargo install trunk
rustup target add wasm32-unknown-unknown
```

### Dev loop (live-reload)
```bash
# Terminal 1: FastAPI backend
.venv/bin/python -m minimappr

# Terminal 2: Trunk dev server (proxies /api and /ws to :8000)
cd minimappr-frontend && trunk serve
# Open http://localhost:8080 in a browser
```

### Release build (required before `python -m build`)
```bash
scripts/build_frontend.sh
# Outputs: minimappr/frontend/{index.html,*.js,*.wasm,*.css}
```

### Pre-publish check
```bash
ls minimappr/frontend/*.wasm   # must exist before packaging
python -m build
unzip -l dist/minimappr-*.whl | grep frontend   # confirm wasm is included
```

## Roadmap Foundation Included
This MVP lays groundwork for the broader goals in your notes/proposals:
- additional sensor modalities
- richer model chaining (speech/STT/Home Assistant integration)
- federated fusion server topology
- richer COP layers (GDOP overlays, zones, alerting policies)
- advanced tracking and multi-hypothesis association
