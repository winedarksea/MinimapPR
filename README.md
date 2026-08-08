# MinimapPR

**Realtime environmental awareness: distributed sound localization + classification + common operating picture.**

Point a microphone array — or several, PPS/GPS-synced — at the world, and MinimapPR gives you a live map of
what is making noise, *where* it is, and whether you should care about it. Sounds are localized in 3D,
classified (bird species, speech, drones, gunshots, machinery), associated
into tracks over time, and pushed to a browser-based common operating picture with alerting rules you write.

Everything runs **locally on your own hardware**. One command starts the server, one browser tab is the
whole UI. No cloud service, no account, no telemetry.

```
mic array(s) ──▶ ingest ──▶ localize (TDOA/SRP-PHAT) ──▶ classify ──▶ track ──▶ rules ──▶ COP / alerts / integrations
```

> **Status:** beta (`0.2.1`). The core pipeline, UI, and integrations are implemented and tested;
> interfaces still move between releases. See [TODO.md](TODO.md) for the full implementation status.

---

## Why you might want this

### Highlights

- **Live common operating picture.** Full-bleed geographic map with entity symbology (MIL-STD-2525-inspired),
  uncertainty ellipses, velocity leaders, GDOP coverage overlay, zone polygons, a sortable track table, and a
  filtered live detection feed over WebSocket.
- **One node or a whole array.** A single Sirith tetrahedral array on a Pi does useful 3D localization on its
  own. Add more nodes and you get networked TDOA, cross-node SRP-PHAT beamforming, and multi-server
  federation — same software, same UI.
- **Species-level bird ID, placed on the map.** BirdNET names the species, YAMNet covers the other 521 classes,
  and the result is a *located, tracked entity* rather than a bare label — with an eBird-shaped export when
  you're done reviewing.
- **Speech-to-text with keyword alerting.** Transcribe speech in range and fire alerts on matched keywords,
  with the triggering audio snippet attached as evidence.
- **Alerting you actually control.** A config-driven rules engine evaluates detections and tracks against your
  conditions (class, confidence, zone, time, environment) and emits alerts, Home Assistant state, or effector
  commands. Coyote, gunshot, drone, perimeter-intrusion, and a "help me" speech-keyword rule all ship enabled
  in `data/rules.json`; anything else follows from classifier labels plus your own conditions. Rules
  hot-reload — no restart to tune.
- **Zones and exclusions.** Draw polygons and suppress expected sounds inside them, or alert only on sounds
  from them. Per-zone occupancy and sound level are first-class outputs.
- **Spatial audio recording.** An offline studio render pipeline exports ambisonic/IAMF MP4 so you can go back
  and *listen* to a scene spatially after the fact.

### Also included

- **PTZ camera slew-to-track** — ONVIF cameras point themselves at a localized sound event.
- **BLE device localization** — RSSI multilateration puts Bluetooth devices on the same picture as audio tracks.
- **Environmental sensing** — per-node temperature/humidity feeds live speed-of-sound correction for tighter
  localization.
- **Home Assistant integration** — outbound MQTT with auto-discovery: zone occupancy, per-zone SPL, detection
  impulse sensors, node diagnostics. Nothing to configure on the HA side beyond a shared broker. (Discovery
  payloads are locked down by golden fixtures and a spec lint, but have not yet been smoke-tested against a
  live HA instance — see [the doc](docs/home_assistant_integration.md).)
- **Analysis views** — daily activity rollups, detection heatmaps, and a label browser.
- **Review + training loop** — confirm/reject detections in the UI or API, promote confirmed clips to training
  data, capture ground-truth bundles, and replay them through the pipeline as a regression gate.
- **Built for the field** — store-and-forward buffering across network drops, graceful degradation as nodes
  fall off (3D → 2D → classify-only → alert-only), BIT health reporting, retention tiers with automatic
  cleanup, and a Rust fast-path ingest sidecar for high-rate deployments.

Self-hosted, GPL-3.0, SQLite on disk. Your audio never leaves your network.

---

## Quick start

### 1. Install

The one-liner installs [`uv`](https://docs.astral.sh/uv/) and then `uv tool install`s MinimapPR —
cross-platform, no manual Python setup:

```bash
# macOS / Linux
curl -LsSf https://minimappr.com/install.sh | sh

# Windows (PowerShell)
irm https://minimappr.com/install.ps1 | iex
```

To skip the BirdNET/ONVIF/MQTT extras, install the base package instead:
`curl -LsSf https://minimappr.com/install.sh | sh -s -- --base`

Prefer pip? Python 3.11–3.13, and note TensorFlow is a ~500 MB download:

```bash
pip install "minimappr[full]"     # BirdNET + ONVIF effectors + HA MQTT
# or just:  pip install minimappr
```

### 2. Run

```bash
minimappr
```

Open **<http://127.0.0.1:8080>**. That's the entire UI — map, detections, tracks, alerts, settings.

Uninstall any time with `uv tool uninstall minimappr`.

### 3. See it working without hardware

In a second terminal, run the built-in two-node simulator (one point node, one tetrahedral array):

```bash
minimappr-demo --server http://127.0.0.1:8080
```

Nodes should appear on the map, detections should populate the feed, and tracks should move.

### 4. Point real nodes at it

Set your site origin so the map lands in the right place, then flash nodes to POST at your server:

```bash
export MINIMAPPR_SITE_ORIGIN_LAT=37.7749
export MINIMAPPR_SITE_ORIGIN_LON=-122.4194
export MINIMAPPR_SITE_ORIGIN_ALT_M=0.0
minimappr
```

Firmware lives in [firmware/](firmware/) — see [firmware/README.md](firmware/README.md) for build and flash
instructions, and [BETA_SETUP.md](BETA_SETUP.md) for the tested end-to-end deployment path.

---

## Node types

### Sirith tetrahedral node
Four-channel array (`node_type: sirith_tetra`); default geometry is a regular tetrahedron with 50 mm edge
(see [schematics/](schematics/)). Localizes on its own from a single node — this is the baseline standalone
deployment, and works on a Pi 5 running the full fusion server.

### Sirith planar node
Planar microphone array (`node_type: sirith_planar`) — same firmware family, different geometry.

### Point node
Single-channel stream node (`node_type: point`), ESP32-style. Intended for GPS/PPS-timestamped network
localization across several physically separated nodes.

Firmware targets:

| Path | Target |
|---|---|
| [firmware/lib/minimap_node_core](firmware/lib/minimap_node_core) | shared node runtime / protocol / transport |
| [firmware/lib/minimap_node_runtime](firmware/lib/minimap_node_runtime) | generic node runner (audio source + publisher loop) |
| [firmware/lib/minimap_audio_esp32](firmware/lib/minimap_audio_esp32) | ESP32 I2S audio sources |
| [firmware/lib/minimap_audio_pico](firmware/lib/minimap_audio_pico) | RP2040/RP2350 Pico TDM audio sources |
| [firmware/lib/minimap_transport_cyw43](firmware/lib/minimap_transport_cyw43) | Pico W CYW43 WiFi transport |
| [firmware/lib/minimap_transport_espc5](firmware/lib/minimap_transport_espc5) | ESP32-C5 transport |
| [firmware/nodes/sirith_tetrahedral](firmware/nodes/sirith_tetrahedral) | Sirith tetrahedral node (Pico SDK / CMake) |
| [firmware/nodes/sirith_planar](firmware/nodes/sirith_planar) | Sirith planar array node (Pico SDK / CMake) |
| [firmware/nodes/point_single_mic](firmware/nodes/point_single_mic) | reference point node (PlatformIO) |

---

## Configuration

Most settings are editable live in the UI under **Settings**, and persist as a sparse YAML overlay at
`data/config.yml`. Environment variables override the file and are the right choice for deployment scripts.
The **Settings → Pipeline** view renders the live processing DAG with each stage's config attached, which is
usually the fastest way to find the knob you want.

### The handful you'll actually set

| Variable | Default | Purpose |
|---|---|---|
| `MINIMAPPR_HOST` / `MINIMAPPR_PORT` | `0.0.0.0` / `8080` | bind address |
| `MINIMAPPR_DB_PATH` | `data/minimappr.db` | SQLite database |
| `MINIMAPPR_SITE_ORIGIN_SOURCE` | `auto` | derive site origin from node GPS midpoint, else the fallback coords below |
| `MINIMAPPR_SITE_ORIGIN_LAT` / `_LON` / `_ALT_M` | `44.987` / `-93.258` / `0.0` | fallback reference point for local ↔ geographic conversion |
| `MINIMAPPR_COORDINATE_MODE` | `flat` | `flat` (local XY meters) or `geodetic` |
| `MINIMAPPR_CLASSIFIER_ROUTING_CONFIG_PATH` | `data/classifier_routing.json` | which models run on which audio context |
| `MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE` | `beamformed` | `beamformed` (localized render) or `omni` |
| `MINIMAPPR_TRIGGER_RMS` | `0.001` | event trigger threshold |
| `MINIMAPPR_SNIPPET_RETENTION_SECONDS` | `3600` | how long detection audio is kept |

### Classifier routing

Classification is **always on and per-context**, driven by `data/classifier_routing.json`
(override the path with `MINIMAPPR_CLASSIFIER_ROUTING_CONFIG_PATH`). The file names classifier
instances, says which of them run for each audio context, and wires chained stages:

```json
{
  "version": 1,
  "classifiers": { "<member_id>": { "backend": "yamnet|birdnet|drone_head|stt" } },
  "contexts":    { "<context>": { "run": ["<member_id>", ...] } },
  "chains":      [ { "id": "...", "after": "<member_id>", "input": "audio|embedding" } ],
  "triggers":    [ { "id": "...", "on": "<member_id>", "action": "speech_capture", "labels": [...] } ]
}
```

The three contexts are `detection_trigger` (the cheap RMS/cooldown admission gate — empty by
default, not an inference context), `localized_render` (the beamformed track render), and
`omni_continuous` (a normalized sum of each node's synchronized mic windows). Shipped default:
YAMNet and BirdNET on `localized_render`, BirdNET and `t3t4_alarm` on `omni_continuous`, the drone
head chained off YAMNet's embeddings, and STT triggered on speech labels.

Available backends:

- **`yamnet`** — general 521-class audio classifier, shipped as a validated local TensorFlow
  SavedModel under `minimappr/assets/yamnet`. Nothing is fetched from TF Hub at startup.
  Apache-2.0, with provenance and checksums recorded alongside the asset.
- **`birdnet`** — bird species classifier, with a site-specific allow-list built from your
  site origin coordinates.
- **`drone_head`** — a chained head that rides YAMNet's per-frame embeddings (`input: "embedding"`),
  so it costs nothing extra wherever YAMNet already runs.
- **`stt`** — Moonshine speech-to-text, feeding transcripts and keyword alerting.
- **`heuristic`** — dependency-free baseline labels (`bird_like`, `speech_like`, `impulse`, `machine_hum`,
  `ambient`). Also the automatic fallback if routing resolves zero members for a context.
- **`t3t4_alarm`** — temporal alarm-pattern detector for repeating alert tones.

Per-backend kill switches, if you'd rather not edit the routing file:

```bash
export MINIMAPPR_BIRDNET_ENABLED=false
export MINIMAPPR_DRONE_HEAD_ENABLED=false
export MINIMAPPR_STT_ENABLED=false
export MINIMAPPR_OMNI_SCAN_ENABLED=false
```

See [docs/classifier_routing.md](docs/classifier_routing.md) for the full schema.

> **Migrating from ≤0.1.x:** `MINIMAPPR_CLASSIFIER`, `MINIMAPPR_MODEL_CHAIN_CONFIG_PATH`, and
> `MINIMAPPR_RUNTIME_PROFILE` were removed, and startup *fails loudly* if any of them is still
> set — silently ignoring them would change which models run on live audio. The error message
> lists the exact replacement variables.

<details>
<summary><strong>Full environment variable reference</strong></summary>

Environment variables override `data/config.yml`, which in turn overrides the built-in defaults.

**Server & storage**
- `MINIMAPPR_HOST` (default `0.0.0.0`)
- `MINIMAPPR_PORT` (default `8080`)
- `MINIMAPPR_CONFIG_PATH` (default `data/config.yml`)
- `MINIMAPPR_DB_PATH` (default `data/minimappr.db`)
- `MINIMAPPR_SNIPPET_DIR` (default `data/snippets`)
- `MINIMAPPR_SNIPPET_RETENTION_SECONDS` (default `3600`)
- `MINIMAPPR_RETENTION_TRACK_UPDATES_SECONDS` (default `604800`, `-1` disables cleanup)
- `MINIMAPPR_RETENTION_ALERTS_SECONDS` (default `2592000`, `-1` disables cleanup)
- `MINIMAPPR_RETENTION_ENVIRONMENT_SECONDS` (default `604800`, `-1` disables cleanup)
- `MINIMAPPR_RETENTION_DROPPED_TRACKS_SECONDS` (default `604800`, `-1` disables cleanup)

**Site geometry**
- `MINIMAPPR_SITE_ORIGIN_SOURCE` (`auto` default; uses the midpoint of active nodes with GPS `position_geo` when available, otherwise the configured fallback coordinates)
- `MINIMAPPR_SITE_ORIGIN_LAT` (default `44.98698840878797`)
- `MINIMAPPR_SITE_ORIGIN_LON` (default `-93.2579197515542`)
- `MINIMAPPR_SITE_ORIGIN_ALT_M` (default `0.0`)
- `MINIMAPPR_COORDINATE_MODE` (`flat` or `geodetic`; default `flat`)

**Triggering & windows**
- `MINIMAPPR_TRIGGER_RMS` (default `0.001`)
- `MINIMAPPR_TRIGGER_COOLDOWN_SECONDS` (default `0.8`)
- `MINIMAPPR_LOCALIZATION_WINDOW_SECONDS` (default `0.08`)
- `MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS` (default `30.0`)
- `MINIMAPPR_REPORTING_WINDOW_SECONDS` (canonical detection dedupe window; default `30`)
- `MINIMAPPR_EVENT_STALE_SECONDS` (default `30.0`)

**Localization**
- `MINIMAPPR_LOCALIZATION_ALGORITHM` (`gcc_phat` default; also `srp_phat`, `music`, `esprit`)
- `MINIMAPPR_LOCALIZATION_STRATEGY` (`geometry_aware` default, or `fixed`)
- `MINIMAPPR_LOCALIZATION_BAND_MIN_HZ` / `MINIMAPPR_LOCALIZATION_BAND_MAX_HZ` (optional localization-only bandpass; `0` disables)
- `MINIMAPPR_LOCALIZATION_SINGLE_NODE_SOLVER` (`python_cartesian` default — re-homes the single-node tetrahedral position solve onto Python's Cartesian TDOA solver using the Rust sidecar's pairwise TDOAs + bearing, falling back to the sidecar's own estimate if TDOAs are missing; set `rust` to trust the sidecar's own SRP-PHAT position/confidence directly, the legacy behavior)
- `MINIMAPPR_SKIP_LOCALIZATION_FOR_CLASSIFICATION` (`false` default)

**Classification**
- `MINIMAPPR_CLASSIFIER_ROUTING_CONFIG_PATH` (default `data/classifier_routing.json`)
- `MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE` (`beamformed` default, or `omni`)
- `MINIMAPPR_BIRDNET_ENABLED` / `MINIMAPPR_DRONE_HEAD_ENABLED` / `MINIMAPPR_STT_ENABLED` / `MINIMAPPR_OMNI_SCAN_ENABLED` (all `true` by default)
- `MINIMAPPR_BIRDNET_TRIGGER_MIN_CONFIDENCE` (default `0.40`; minimum confidence before BirdNET detections are emitted or BirdNET chain stages fire)
- `MINIMAPPR_BIRDNET_GEO_MIN_CONFIDENCE` (default `0.01`; BirdNET geo-model occurrence threshold used to build the site-specific species allow-list from `MINIMAPPR_SITE_ORIGIN_LAT/LON`)
- `MINIMAPPR_DETECTION_MIN_CONFIDENCE` (default `0.4`; hides lower-confidence detections from detection APIs/UI and soundscape rendering)
- `MINIMAPPR_OMNI_SCAN_INTERVAL_SECONDS` (default `30.0`)
- `MINIMAPPR_STT_MODEL_ID` (default `onnx-community/moonshine-base-ONNX`)

**Environment / speed of sound**
- `MINIMAPPR_DEFAULT_TEMPERATURE_C` (default `20.0`)
- `MINIMAPPR_DEFAULT_HUMIDITY` (default `0.5`)
- `MINIMAPPR_ENVIRONMENT_READING_MAX_AGE_SECONDS` (default `300.0`, `0` disables the staleness cutoff)

**Tracking**
- `MINIMAPPR_TRACKING_FILTER` (`kalman` default, or `linear`)
- `MINIMAPPR_KALMAN_PROCESS_NOISE` (default `2.0`)
- `MINIMAPPR_KALMAN_MEASUREMENT_NOISE` (default `1.5`)
- `MINIMAPPR_KALMAN_INITIAL_POSITION_VARIANCE` (default `4.0`)
- `MINIMAPPR_KALMAN_INITIAL_VELOCITY_VARIANCE` (default `16.0`)

**Pipeline & node health**
- `MINIMAPPR_FUSION_WORKER_COUNT` (default `2`)
- `MINIMAPPR_FUSION_EVENT_QUEUE_SIZE` (default `512`)
- `MINIMAPPR_NODE_DEGRADED_AFTER_SECONDS` (default `15.0`)
- `MINIMAPPR_NODE_OFFLINE_AFTER_SECONDS` (default `45.0`)

**Rules & ingest**
- `MINIMAPPR_RULES_CONFIG_PATH` (default `data/rules.json`)
- `MINIMAPPR_DIRECT_INGEST_ENABLED` (`true` default; set `false` to force firmware batch ingest through the Rust sidecar)
- `MINIMAPPR_INGEST_SPOOL_DIR` (default `data/spool`)
- `MINIMAPPR_INGEST_SPOOL_READY_TTL_SECONDS` (default `60`)

**Federation**
- `MINIMAPPR_FEDERATION_ENABLED` (`false` default)
- `MINIMAPPR_FEDERATION_SERVER_ID` (`srv-local` default)
- `MINIMAPPR_FEDERATION_PEERS_CONFIG_PATH` (default `data/federation_peers.json`)
- `MINIMAPPR_FEDERATION_PEERS_JSON` (optional inline JSON peer config override)
- `MINIMAPPR_FEDERATION_AUTH_TOKEN` (optional shared token / fallback peer auth token)
- `MINIMAPPR_FEDERATION_PUBLISH_INTERVAL_SECONDS` (default `1.0`)
- `MINIMAPPR_FEDERATION_HEARTBEAT_INTERVAL_SECONDS` (default `2.0`)
- `MINIMAPPR_FEDERATION_LINK_TIMEOUT_SECONDS` (default `8.0`)
- `MINIMAPPR_FEDERATION_REQUEST_TIMEOUT_SECONDS` (default `2.5`)
- `MINIMAPPR_FEDERATION_TRACK_TTL_SECONDS` (default `20.0`)
- `MINIMAPPR_FEDERATION_DECONFLICT_MAHALANOBIS_GATE` (default `4.5`)
- `MINIMAPPR_FEDERATION_TQI_HYSTERESIS` (default `0.05`)

**Removed (startup fails if set):** `MINIMAPPR_CLASSIFIER`, `MINIMAPPR_MODEL_CHAIN_CONFIG_PATH`,
`MINIMAPPR_RUNTIME_PROFILE`. See the migration note above.

</details>

---

## Deployment modes

Either way it's a single `minimappr` command; the difference is whether firmware posts directly to the Python
API or to the Rust ingest sidecar.

### Mode 1 — direct Python ingest (default)

```bash
minimappr
```

Firmware posts to `POST /api/v1/ingest/frame` or `/api/v1/ingest/binary` on `:8080`. No Rust process runs.

### Mode 2 — managed Rust ingest sidecar

```bash
export MINIMAPPR_DIRECT_INGEST_ENABLED=false
minimappr
```

Python launches and supervises the sidecar; firmware posts high-rate batch ingest to it on `:8081`. Requires
the binary at `dist/minimappr-ingest-sidecar` (build with `scripts/build_rust.sh --all`). The sidecar's
SRP-PHAT pairwise TDOAs feed Python's Cartesian solver for the position estimate
(`MINIMAPPR_LOCALIZATION_SINGLE_NODE_SOLVER=python_cartesian`, the default).

### Mode 3 — fully split two-process

Run the sidecar as its own independent process with its own lifecycle (e.g. on a separate host):

```bash
# Terminal 1: Python UI/API and spool consumer
minimappr

# Terminal 2: Rust fast-path proxy
MINIMAPPR_INGEST_SPOOL_DIR=data/spool ./dist/minimappr-ingest-sidecar
```

The sidecar accepts `POST /api/v1/ingest/binary` and `/api/v1/ingest/store-forward`, streams bodies to
`data/spool/tmp/`, atomically publishes complete items to `data/spool/ready/`, then returns `202 Accepted`.
Python drains `ready/`, drops items older than `MINIMAPPR_INGEST_SPOOL_READY_TTL_SECONDS`, and moves
parse/delivery failures to `data/spool/failed/`. Point firmware at port `8081`.

### Tuning for continuous wildlife monitoring

If you're running BirdNET over long omni windows rather than short localized impulses, this is the
settings group that matters (these were the old `birdnet_hybrid_production` profile):

```bash
export MINIMAPPR_BIRDNET_ENABLED=true
export MINIMAPPR_LOCALIZATION_ALGORITHM=srp_phat
export MINIMAPPR_LOCALIZATION_STRATEGY=fixed
export MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE=omni
export MINIMAPPR_BIRDNET_CHUNKED_DISPATCH_ENABLED=true
export MINIMAPPR_BIRDNET_CHUNK_OVERLAP_SECONDS=2.0
export MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS=30.0
export MINIMAPPR_MAX_SENSOR_BUFFER_SECONDS=32.0
export MINIMAPPR_LOCALIZATION_BAND_MIN_HZ=300.0
export MINIMAPPR_LOCALIZATION_BAND_MAX_HZ=3500.0
export MINIMAPPR_REPORTING_WINDOW_SECONDS=30.0
```

---

## Development

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[full,train]"
```

Run the backend with live reload:

```bash
uvicorn minimappr.main:app --host 0.0.0.0 --port 8080 --reload
```

### Tests

```bash
source .venv/bin/activate
pytest -q
```

5-minute soak harness:

```bash
python scripts/run_soak.py --duration 300
```

### Frontend (Leptos / WASM)

The operator UI lives in [minimappr-frontend/](minimappr-frontend/) (Rust → WASM via Leptos 0.8 + Trunk).
**End users don't need a Rust toolchain** — the pre-built WASM bundle ships in the wheel. Only contributors
editing the UI do.

```bash
# one-time
cargo install trunk
rustup target add wasm32-unknown-unknown

# dev loop, terminal 1: backend on :8000 (Trunk.toml proxies /api and /ws there)
MINIMAPPR_PORT=8000 .venv/bin/python -m minimappr

# dev loop, terminal 2: Trunk dev server with live reload
cd minimappr-frontend && trunk serve
# open http://localhost:8080
```

Release build (required before `python -m build`):

```bash
scripts/build_frontend.sh                # → minimappr/frontend/{index.html,*.js,*.wasm,*.css}
scripts/build_rust.sh --all              # also builds dist/minimappr-ingest-sidecar
```

Pre-publish check:

```bash
ls minimappr/frontend/*.wasm             # must exist before packaging
python -m build
unzip -l dist/minimappr-*.whl | grep frontend
```

### Project layout

| Path | Contents |
|---|---|
| [minimappr/main.py](minimappr/main.py) | FastAPI app, HTTP endpoints, `/ws/live` |
| [minimappr/core/](minimappr/core/) | buffering, localization, beamforming, tracking, zones, rules, federation, fusion orchestration |
| [minimappr/classifiers/](minimappr/classifiers/) | classifier interface, heuristic/YAMNet/BirdNET/STT backends, chaining and routing |
| [minimappr/storage/db.py](minimappr/storage/db.py) | SQLite schema + persistence |
| [minimappr/models.py](minimappr/models.py) | Pydantic v2 models for the whole system |
| [minimappr/sim/run_demo.py](minimappr/sim/run_demo.py) | realtime two-node simulator |
| [minimappr-frontend/](minimappr-frontend/) | Leptos/WASM operator UI |
| [minimappr-ingest-sidecar/](minimappr-ingest-sidecar/) | Rust firmware-facing ingest + DSP fast path |
| [firmware/](firmware/) | shared embedded node runtime + Sirith/point firmware targets |
| [tests/](tests/) | pipeline, localization, classifier, and integration tests |

Contributor conventions live in [AGENTS.md](AGENTS.md) — notably §2.5: any new pipeline stage or config key
must be registered in both `core/pipeline_graph.py` and `core/config_groups.py`.

### Further reading

- [docs/ui_architecture.md](docs/ui_architecture.md) — frontend structure
- [docs/classifier_routing.md](docs/classifier_routing.md) — how audio reaches which classifier
- [docs/home_assistant_integration.md](docs/home_assistant_integration.md) — MQTT entity contract
- [docs/distributed_localization_verification.md](docs/distributed_localization_verification.md) — multi-node accuracy
- [BETA_SETUP.md](BETA_SETUP.md) — tested end-to-end deployment path
- [TODO.md](TODO.md) — implementation status and roadmap

---

## Reference

### Processing pipeline

1. Ingest timestamped audio frames.
2. Append channel streams to rolling per-sensor buffers.
3. Trigger candidate events from frame RMS threshold.
4. Enqueue trigger candidates to fusion workers.
5. Build synchronized multi-sensor windows.
6. Run TDOA measurement (GCC-PHAT / SRP-PHAT / MUSIC / ESPRIT via dispatch) and nonlinear 3D solve.
7. Classify event audio (beamformed or omni, with optional model chaining).
8. Associate/update track.
9. Persist detection + track, evaluate rules, emit live WebSocket event.
10. Save mono snippet for the retention window; periodic cleanup removes expired snippets.

### Ingestion protocol

`POST /api/v1/ingest/frame`

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

- audio payload is interleaved `pcm16le`, base64 encoded
- `frame.channels` must match `len(node.sensor_offsets_m)` (enforced by a model validator)
- a node may report `node.position_geo` (lat/lon/alt) instead of `node.position_m`; geographic positions
  are converted to local offsets against the site origin on registration
- timestamps are per-frame start timestamps in `ns`
- optional per-frame timing quality metadata: `time_quality`, `toa_ns`, `tor_ns`
- optional environmental payload: `environment.temperature_c` (minimum), humidity/pressure/wind/lux optional
- firmware-compatible fallback: `node.metadata.temperature_c` is ingested into `environment` even without an
  explicit `environment` object
- response `triggered=true` means an event candidate was queued for fusion workers; detection emission is
  asynchronous

### API endpoints

- `GET /health`
- `GET /api/v1/config`
- `GET /api/v1/config/structured`
- `GET /api/v1/pipeline/graph`
- `GET /api/v1/fusion/status`
- `GET /api/v1/federation/status`
- `GET /api/v1/context/current`
- `POST /api/v1/ingest/frame`
- `GET /api/v1/nodes`
- `GET /api/v1/nodes/{node_id}/audio/recent?seconds=10`
- `GET /api/v1/detections?limit=100`
- `GET /api/v1/detections/{detection_id}`
- `GET /api/v1/detections/{detection_id}/audio`
- `PATCH /api/v1/detections/{detection_id}/review`
- `GET /api/v1/tracks?limit=200&include_standby=false`
- `GET /api/v1/zones/occupancy`
- `GET /api/v1/cop/status`
- `GET /api/v1/alerts?limit=100`
- `GET /api/v1/environment?limit=500&node_id=...`
- `GET /api/v1/environment/current?x=...&y=...&z=...`
- `GET /api/v1/exports/ebird?format=json|csv&limit=500&since_hours=24`
- `GET /api/v1/transcripts` and `/api/v1/transcripts/{transcript_id}/audio`
- `GET /api/v1/analytics/daily`, `/api/v1/analytics/heatmap`, `/api/v1/analytics/labels`, `/api/v1/analytics/classifiers`
- `GET|POST /api/v1/zones`, `GET|POST /api/v1/rules`, `GET|POST /api/v1/overlays`
- `GET|POST /api/v1/recordings` and `/api/v1/recordings/{session_id}/download`
- `POST /api/v1/capture/start` / `/api/v1/capture/{session_id}/stop` (calibration bundles)
- `GET /api/v1/classifier-routing`
- `GET /api/v1/ble/devices`, `POST /api/v1/ingest/ble`, `POST /api/v1/ingest/env`
- `GET /api/v1/nodes/{node_id}/bit` (built-in test reports)
- `POST /api/v1/nodes/{node_id}/effector/aim|arm|disarm` and `GET .../effector/snapshot.jpg`
- `GET /api/v1/integrations/hass/status` and `POST .../republish-discovery`
- `GET /api/v1/system/diagnostics`, `/api/v1/system/logs`, `/api/v1/debug/selftest`
- `POST /api/v1/federation/heartbeat` (peer-to-peer)
- `POST /api/v1/federation/snapshot` (peer-to-peer)
- `WS /ws/live`

The full surface is browsable at `/docs` (FastAPI's generated OpenAPI UI) on a running server.

### Detection review and export workflow

The v1 bird workflow is review-driven rather than classifier-final:

1. Inspect detections with `GET /api/v1/detections` (or the UI).
2. Review with `PATCH /api/v1/detections/{detection_id}/review`.
3. Export confirmed detections with `GET /api/v1/exports/ebird`.

The review mutation accepts `review_state` (`unreviewed` / `confirmed` / `rejected`), `review_label`,
`review_label_category`, `review_notes`, and `promote_to_training` (confirmed reviews only).

```bash
curl -X PATCH "http://127.0.0.1:8080/api/v1/detections/det-123/review" \
  -H "Content-Type: application/json" \
  -d '{
    "review_state": "confirmed",
    "review_label": "song_sparrow",
    "review_label_category": "bird",
    "review_notes": "confirmed by operator",
    "promote_to_training": true
  }'

curl "http://127.0.0.1:8080/api/v1/exports/ebird?format=json&since_hours=24"
curl "http://127.0.0.1:8080/api/v1/exports/ebird?format=csv&since_hours=24" --output ebird_export.csv
```

### Audio path validation (troubleshooting)

When detections are absent, verify audio ingest is healthy before touching classifier settings:

1. Open the COP dashboard and use the **Node Audio Debug** panel.
2. Click **Listen** on a node to request the most recent buffered clip.
3. If it doesn't sound reasonable, inspect node health and ingest transport first.

```bash
curl "http://127.0.0.1:8080/api/v1/nodes/http-node-1/audio/recent?seconds=10" --output node_recent.wav
```

---

## Roadmap

Groundwork is in place for:

- additional sensor modalities and multi-modal fusion
- richer model chaining (speech/STT → Home Assistant automation)
- federated fusion-server topologies at larger scale
- richer COP layers (advanced zones, alerting policies, coverage planning)
- multi-hypothesis tracking and JPDA association

See [TODO.md](TODO.md) for per-item status.

## License

GPL-3.0. See [LICENSE](LICENSE).
