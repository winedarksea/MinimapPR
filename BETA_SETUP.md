# MinimapPR Beta Setup — Sirith Tetrahedral

Tested path for beta deployment with YAMNet classification, BirdNET species chaining,
and coyote-howl alerting on a Sirith tetrahedral node.

---

## Requirements

| Item | Requirement |
|------|-------------|
| Python | **3.11 or 3.12** (BirdNET 0.2.x requires <3.14; TensorFlow requires >=3.11) |
| Hardware | Sirith tetrahedral node (4-mic array), USB audio or I²S via ADAU7112 |
| OS | Linux (Raspberry Pi OS 64-bit recommended) or macOS for dev |

---

## Install

```bash
# 1. Create a Python 3.11+ virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install MinimapPR with the full beta stack (YAMNet + BirdNET)
pip install -e ".[full]"
# Equivalent to: pip install -e ".[yamnet,birdnet]"
# Installs: tensorflow>=2.19, tensorflow-hub>=0.16, birdnet>=0.2.12
```

> **Note:** TensorFlow downloads are large (~500 MB). On a Pi, use a machine with
> good bandwidth and copy the `.venv/` to the Pi, or build a wheel cache.

---

## Environment Variables

Create a `.env` file or export these before starting:

```bash
# --- Site geometry (required — replace with your actual coordinates) ---
export MINIMAPPR_SITE_ORIGIN_LAT=37.7749
export MINIMAPPR_SITE_ORIGIN_LON=-122.4194
export MINIMAPPR_SITE_ORIGIN_ALT_M=0.0
export MINIMAPPR_COORDINATE_MODE=flat        # flat = local XY (meters from origin)

# --- Classifier (defaults to yamnet since beta; shown here for clarity) ---
export MINIMAPPR_CLASSIFIER=yamnet
export MINIMAPPR_YAMNET_MIN_CONFIDENCE=0.25

# --- Localization (geometry_aware selects tight-array algo for Sirith) ---
export MINIMAPPR_LOCALIZATION_STRATEGY=geometry_aware
export MINIMAPPR_LOCALIZATION_TIGHT_ARRAY_APERTURE_M=0.35  # Sirith aperture

# --- Config files (defaults ship in data/) ---
export MINIMAPPR_RULES_CONFIG_PATH=data/rules.json
export MINIMAPPR_MODEL_CHAIN_CONFIG_PATH=data/model_chain.json

# --- Optional: federation token if running multiple nodes ---
# export MINIMAPPR_FEDERATION_TOKEN=your-secret-token
```

### Bird-Focused Test Mode

When you want to test bird detections without the YAMNet gate and without beamformed classification, use the built-in BirdNET omni profile instead:

```bash
export MINIMAPPR_RUNTIME_PROFILE=birdnet_omni_testing
export MINIMAPPR_BIRDNET_TRIGGER_MIN_CONFIDENCE=0.05
```

That profile:
- switches the primary classifier to `birdnet`
- disables beamformed classification
- skips localization before classification
- uses a 30 s trailing omni classification clip instead of the short localization window

---

## Startup

```bash
source .venv/bin/activate
# From the repo root (so that data/ relative paths resolve correctly):
minimappr
# Or explicitly:
.venv/bin/python -m minimappr
```

Open `http://localhost:8000` in a browser. The COP UI shows:
- Live track table and map
- Detection feed
- **Alert feed** — coyote howl and other rule alerts appear here and as toasts

---

## Beta Config Files

Two checked-in configs in `data/` drive the beta behaviour:

| File | Purpose |
|------|---------|
| `data/model_chain.json` | YAMNet base → BirdNET species ID for bird detections |
| `data/rules.json` | coyote_howl_alert (priority: high, 30 s cooldown), security_high_confidence, human_perimeter |

Both files are hot-reloaded on change — no restart needed to tune rules or the chain.

---

## Firmware Target

Node firmware target: **Sirith tetrahedral** (`firmware/nodes/sirith_tetrahedral`).

Flash and configure each node so that it POSTs audio frames to:
```
http://<server-ip>:8000/api/v1/ingest/frame
```
with the current `IngestFrameRequest` schema:
- `node.id`, `node.node_type`, `node.position_m`, and `node.sensor_offsets_m`
- `frame.start_time_ns`, `frame.sample_rate_hz`, `frame.channels`, `frame.encoding`, `frame.samples_b64`
- optional `frame.sequence`, `frame.toa_ns`, `frame.tor_ns`, and `frame.time_quality`
- optional `environment` payload for temperature and other environmental measurements

The checked-in firmware already targets this endpoint and schema.

---

## Running Tests

```bash
.venv/bin/python -m pytest tests/ --ignore=tests/test_soak_harness.py
```

177 core tests should pass. The soak harness (`test_soak_harness.py`) requires a live
backend and is excluded from CI.
