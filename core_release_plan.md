# MinimapPR Core Release — Gap-Closure Plan

## Context

MinimapPR is a real-time acoustic localization + COP system. Server-side code spans the Python FastAPI app (`minimappr/`) and the Rust ingest sidecar (`minimappr-ingest-sidecar/`). The nine core release requirements (audio continuity, TDOA precision, pipeline isolation, geolocation, off-disk streaming, BirdNET production, wavelength-aware localization, IAMF + synced-video export) are mostly already in place. This plan focuses on **server-side code improvements with clear release value**.

Full hardware-in-the-loop validation (multi-node sessions with physical sources, induced drops, etc.) is deferred to a **Phase 2** verification pass and not in scope here. Each work item below ships with **synthetic / unit-level tests** that confirm correctness without requiring a deployed array.

### Open gaps (only items still needing work)

| # | Gap | Where |
|---|---|---|
| A | SQLite retention — missing VACUUM, missing tier indexes, untracked `ingested_frames`/`bit_reports`/`pings` rows | `cleanup_service.py`, `storage/db.py` |
| B | Audio gap detection is passive — `audio_buffer.py:44-88` reports but zero-pads silently; no sequence-gap detection | `audio_buffer.py`, `ingest.py`, `node_registry.py` |
| C | UI / hot-path isolation has no SLO enforcement on workers; needs concurrent-load test to verify isolation | `fusion_node.py`, tests |
| D | BirdNET → labelled-track → IAMF object propagation needs an end-to-end regression test | `classification.py`, `tracking.py`, `assembly.py` |
| E | Wavelength-aware TDOA confidence — alias-cutoff math exists (`ambi_atob.py:232-237`) but localization never consults it | `localization_dispatch.py`, `localization.py` |
| F | IAMF Parameter_Block_OBU emits empty subblocks in Python — Rust writer already encodes object positions; bring Python to parity | `core/iamf_writer.py` |
| G | Rust ↔ Python ingest design drift: IAMF object positions, metrics surface, classification dual-path, shared config keys | both ingest paths |
| H | `config.py` is a 160-field flat mega-class with scattered defaults, naming inconsistencies, and risky/silent defaults | `config.py`, `core/classification.py`, `cleanup_policy.json` |

### Decisions (resolved)

- Wavelength gate behavior → **try TDOA anyway, attach a `wavelength_factor` confidence multiplier**.
- Alert policy → **metrics-only**, surfaced through `/api/v1/status`. No `alerts`-table rows.
- Gap recovery → **server-side detection + metrics only**. No firmware protocol change.
- IAMF fidelity → **encode per-temporal-unit positions in Parameter_Block_OBU**. Bring Python to parity with the Rust writer.

---

## Implementation order

### 1. SQLite cleanup hardening (Gap A) — effort: **S**

Independent, low-risk, unblocks soak validation for the rest.

**Modify:**
- [`minimappr/storage/db.py`](minimappr/storage/db.py) — at DB open, set `PRAGMA auto_vacuum=INCREMENTAL`; add covering indexes `ix_detections_retention_tier_created`, `ix_large_artifacts_retention_tier_expires`, `ix_track_updates_retention_tier_created` (if applicable).
- [`minimappr/cleanup_service.py:84-118`](minimappr/cleanup_service.py#L84-L118) — after `cleanup_retention()`, run `PRAGMA incremental_vacuum;` and once per cycle `PRAGMA optimize;`.
- Extend `StorageBackend.cleanup_retention()` to accept TTLs for `ingested_frames`, `bit_reports`, `pings` and prune those tables.

**Config (will move into the new `RetentionConfig` block from item 10):** `retention_ingested_frames_seconds`, `retention_bit_reports_seconds`, `retention_pings_seconds`.

**Test:** extend `tests/test_storage_retention.py` — assert ingested_frames expire, indexes exist via `sqlite_master`, freelist grows on bulk delete and shrinks after `incremental_vacuum`.

---

### 2. Pipeline metrics + visibility (Gaps B, C) — effort: **S**

Instrument first so subsequent changes are measurable, and align with the Rust sidecar's exposed counters (see item 9).

**Modify:**
- [`minimappr/core/fusion_node.py`](minimappr/core/fusion_node.py) — extend `FusionMetrics` with:
  - `frames_zero_padded_degraded: int = 0`
  - `frame_sequence_gaps: int = 0`
  - `localization_band_aliased_count: int = 0`
  - `stage_timeout_count: int = 0`
- Increment `frames_zero_padded_degraded` in the consumer of `AudioCoverageStats` when `degraded=True`; log a structured warning.
- Confirm `triggers_dropped_queue_full` is exposed in `/api/v1/status` and add it to the metrics summary if missing.

**Test:** extend `tests/test_fusion_node.py` — drive a deliberate queue overflow + degraded coverage stats and assert counters increment.

---

### 3. Server-side audio sequence-gap detection (Gap B) — effort: **S/M**

Server-only per the decision above.

**Modify:**
- [`minimappr/core/node_registry.py`](minimappr/core/node_registry.py) — runtime entries gain `last_sequence_seen: int | None` and `boot_session: str | None`.
- [`minimappr/core/ingest.py:170-181`](minimappr/core/ingest.py#L170-L181) — after `has_ingested_frame()` returns False, compare `frame.sequence` against the cached last-seen for `(node_id, boot_session)`. On gap (received > expected+1), increment `frame_sequence_gaps` by `received - (expected+1)`, log warning with `(expected, received, gap_size)`. Reset on `boot_session` change.

**Test:** new `tests/test_ingest_sequence_gap.py` — inject sequences `[3,4,7,8]` (gap of 2), assert counter increments by 2 and no crash on `boot_session` rollover.

---

### 4. Pipeline SLO / stage timeout (Gap C) — effort: **M**

Prevents a slow BirdNET inference from starving the localization stage.

**Modify:**
- [`minimappr/core/fusion_node.py`](minimappr/core/fusion_node.py) — wrap classifier and localization calls inside `_classification_worker_loop` / `_localization_worker_loop` with `asyncio.wait_for(...)`. On timeout: increment `stage_timeout_count`, log warning, drop the candidate.
- Remove the module constant `_CLASSIFICATION_TIMEOUT_S = 30.0` from [`minimappr/core/classification.py:33`](minimappr/core/classification.py#L33) and route it through the new config block (item 10).

**Test:** extend `tests/test_classification_timeout_recovery.py` — inject a 30s sleep into a classifier mock, assert timeout fires, counter increments, next candidate is processed.

---

### 5. UI / hot-path isolation verification (Gap C) — effort: **S**

**Audit** API handlers in `main.py` to confirm they only touch read paths and don't await the ingest queues. Add a synthetic load test:

- New `tests/test_ui_concurrency_isolation.py` — simulate ingest at 8 synthetic nodes × 100 ms frames; concurrently poll `/api/v1/detections`, `/api/v1/tracks`, `/api/v1/status` at 10 Hz; assert ingest p99 latency stays within baseline + 10 % and queue depth doesn't regress.

No production code changes expected unless the audit finds a sync call.

---

### 6. BirdNET → labelled-track → IAMF-object regression (Gap D) — effort: **S**

Confirmation pass with a synthetic regression test.

**Trace and assert:** the BirdNET branch in [`minimappr/core/classification.py`](minimappr/core/classification.py) produces `label_id` + `label_category="bird"`; [`assembly.py:227-354`](minimappr/core/assembly.py#L227-L354) propagates `label_category` to the track update; [`tracking.py:122-242`](minimappr/core/tracking.py#L122-L242) accepts labelled detections and re-associates across calls; `iamf_pipeline.py` uses bird-class track waypoints via `ObjectSlotTrajectory.track_id`.

**Test:** new `tests/test_birdnet_track_e2e.py` — synthesize three sequential bird-call windows at slightly different positions (use an existing fixture WAV from `tests/fixtures/`); assert a single track persists with `label_category="bird"` across all three detections and `IamfObjectSlot.track_id` matches.

---

### 7. Wavelength-aware localization confidence (Gap E) — effort: **M**

Per decision: still attempt TDOA but multiply the reported confidence by a wavelength-feasibility factor.

**Modify:**
- [`minimappr/core/ambi_atob.py:232-237`](minimappr/core/ambi_atob.py#L232-L237) — promote `_alias_cutoff_from_positions` to a public helper `alias_cutoff_from_positions(positions, c_sound)`.
- [`minimappr/core/localization.py`](minimappr/core/localization.py) — add module-level helper `dominant_frequency_hz(window, sample_rate)` (Welch PSD or weighted-FFT centroid; deterministic).
- [`minimappr/core/localization_dispatch.py:71-100`](minimappr/core/localization_dispatch.py#L71-L100) — in `LocalizationDispatcher.localize()`:
  - Compute `f_dom` for the analysis window and `f_alias` for the sensor sub-array.
  - `wavelength_factor = clamp(f_alias / max(f_dom, f_alias), penalty_floor, 1.0)`.
  - Multiply final confidence by `wavelength_factor`; attach it to the result so the UI can show "band-aliased: low spatial confidence".
  - When `f_dom > f_alias`, increment `FusionMetrics.localization_band_aliased_count`.
- [`minimappr/models.py`](minimappr/models.py) — add `wavelength_factor: float | None = None` to `LocalizationResult`; surface it on `DetectionEvent` if appropriate.

**Config (new — in `LocalizationConfig` per item 10):** `wavelength_gating_enabled` (default True), `wavelength_penalty_floor` (default 0.25).

**Test:** new `tests/test_wavelength_gating.py` — feed synthetic narrowband signals at 1 kHz (below cutoff, factor ≈ 1.0) and 8 kHz (above 3.4 kHz tetrahedral cutoff, factor near `penalty_floor`); assert counter increments and that confidence is scaled. Extend `tests/test_dispatch_regression.py` with a band-penalty case.

---

### 8. IAMF Parameter_Block_OBU position payload — Python ↔ Rust parity (Gap F) — effort: **M**

The Rust writer at [`minimappr-ingest-sidecar/src/iamf_writer.rs`](minimappr-ingest-sidecar/src/iamf_writer.rs) already encodes `ObjectPosition` (azimuth, elevation, distance_norm) with LINEAR animation across temporal units. The Python writer at [`core/iamf_writer.py:211-229`](minimappr/core/iamf_writer.py#L211-L229) emits `num_subblocks=0` despite the pipeline already computing `positions_per_unit`. Bring Python to parity with Rust.

**Modify:**
- [`minimappr/core/iamf_writer.py:211-229`](minimappr/core/iamf_writer.py#L211-L229) — extend `_parameter_block()` to accept the per-unit `positions: dict[int, dict]` already passed by callers and encode azimuth/elevation/distance fields using **the exact wire format the Rust writer uses**. Stop emitting empty bytes.
- Keep the `iamf_positions.json` sidecar from [`iamf_pipeline.py:325-332`](minimappr/core/iamf_pipeline.py#L325-L332) as a debug artifact.
- Reuse `_xyz_to_spherical()` and `_interpolate_waypoints()` from [`core/iamf_object_slot.py`](minimappr/core/iamf_object_slot.py).

**Tests:**
- Extend `tests/test_iamf_writer.py` — round-trip a known trajectory through encode → parse and assert position fields recover within quantization tolerance.
- New `tests/test_iamf_writer_python_rust_parity.py` (parallel to `test_atob_foa_tetra_bit_identical.py`) — Python writer and Rust writer produce **byte-identical** output for the same trajectory. This is the contract test for item 9 below.
- Extend `tests/test_iamf_pipeline_e2e.py` — synthetic moving source, confirm encoded azimuth sweeps as expected through the full pipeline.

---

### 9. Rust ↔ Python ingest parity hardening (Gap G) — effort: **M**

The two ingest paths have diverged in ways that are partly intentional (Rust = realtime data pump on tmpfs, Python = API server + tests) and partly accidental drift. Lock in the intentional differences with documentation and the accidental ones with code.

**In-scope alignments:**

a. **IAMF writer wire format**: covered by item 8 + the new parity test (`test_iamf_writer_python_rust_parity.py`).

b. **Coverage stats emission**: both compute the same `AudioCoverageStats` struct (`audio_buffer.py:18-28` ≡ `sidecar/src/dsp.rs:7-18`) but Python uses it in-memory while Rust emits it as optional JSON. Pick one shape and document — recommend **emitting from both as a typed JSON block** so any consumer (status endpoint, manifests) sees the same keys. Touch:
   - `minimappr/core/audio_buffer.py` — add `to_json()` that matches the Rust serialization keys.
   - [`minimappr-ingest-sidecar/src/manifests.rs`](minimappr-ingest-sidecar/src/manifests.rs) — populate `coverage_stats: Option<serde_json::Value>` for real (currently set to `None`).

c. **Metrics surface alignment**: Rust's `DspStatusResponse` exposes queue depth + result counts; Python's `FusionMetrics` exposes stage in/out. Each side adds the *other's* small set so a single status dashboard works:
   - Rust: add stage counters (`total_localization_attempts`, `total_classification_attempts`) to `DspStatusResponse` in [`minimappr-ingest-sidecar/src/main.rs`](minimappr-ingest-sidecar/src/main.rs).
   - Python: add `raw_manifest_queue_depth`-equivalent (`ingest_queue_depth`, `ingest_queue_bytes`) to `/api/v1/status`.

d. **Shared config schema documentation**: produce a single table (in code, not a separate doc) of env vars consumed by both sides. The Rust sidecar already reads `MINIMAPPR_LOCALIZATION_WINDOW_SECONDS`, `MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS`, `MINIMAPPR_MAX_SENSOR_BUFFER_SECONDS`, `MINIMAPPR_LOCALIZATION_BAND_MIN_HZ`, `*_MAX_HZ`, `MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M`, `MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M` — wire these to the same `LocalizationConfig` block introduced in item 10 so Python and Rust pull the same defaults.

**Out-of-scope (documented as intentional divergence):**
- Localization algorithm cascade (Python has MUSIC/ESPRIT, Rust does not) — Rust is tetrahedral-optimised; document and move on.
- Classification dual-path (Python runs omni + beamformed, Rust feeds one) — call out in code comments; revisit post-release.
- Storage shape (Python = SQLite, Rust = filesystem manifests + journal) — clean role separation.
- Backpressure model (Python = drop-on-queue-full, Rust = byte-budget 429) — both valid for their roles.

**Test:** `tests/test_iamf_writer_python_rust_parity.py` (from item 8) is the headline parity test. Also extend `tests/test_ingest_process_split.py` to assert both sides produce equivalent `AudioCoverageStats` JSON for the same input frames.

---

### 10. Configuration cleanup (Gap H) — effort: **M**

`config.py` is a 160-field flat mega-class. Restructure into typed nested config blocks, fix risky defaults, and remove redundancy. Server-side scope only — firmware `node_config.h` is out of scope here.

**Restructure `minimappr/config.py`:**
- Group all 160 fields into typed sub-configs: `IngestConfig`, `LocalizationConfig`, `ClassificationConfig`, `TrackingConfig`, `FusionConfig` (already exists — extend), `RetentionConfig`, `RulesConfig`, `FederationConfig`, `SiteConfig`, `ApiConfig`. Most of these already exist as inner dataclasses; finish the migration so `Settings` becomes a thin composition rather than a flat field bag.
- Remove the `Settings.fusion_drop_on_backpressure` → `FusionConfig.drop_on_backpressure` translation layer (one of the two): single source of truth.
- Rename `localization_max_tau_s` → `localization_max_tau_seconds` (the only `_s` outlier).

**Fix risky / unclear defaults:**
- `fusion_*_queue_size` (currently 256 for all four queues) — raise to **1024** for `localization`/`classification` and **512** for `event`/`rules`. Profile in item 5's load test.
- `fusion_drop_on_backpressure=True` is fine but **must log a structured warning** every time it drops (rate-limited to e.g. one log per 5s per queue). Add the log in the same place `triggers_dropped_queue_full` is incremented.
- Document the precedence between `cleanup_policy.json` (per-label TTL overrides) and `RetentionConfig.snippet_retention_seconds` (global default). Cleanup-policy wins; add a `# Precedence: cleanup_policy.json overrides this default per label.` comment on the field.

**Move hard-coded values into config:**
- `_CLASSIFICATION_TIMEOUT_S` in [`core/classification.py:33`](minimappr/core/classification.py#L33) → `ClassificationConfig.stage_timeout_seconds` (used by item 4).
- Any other module-level threshold constants surfaced by the audit (e.g. `_PAIR_OWNERS_MAX_SIZE` in federation — already done — but scan `localization*.py`, `tracking.py`, `classification.py`, `audio_buffer.py` for similar).

**Shared with Rust (item 9d):** the env-var-key set consumed by both must align with the new `LocalizationConfig` / `ClassificationConfig` field names. Rust reads env vars directly today; document the canonical key list in `LocalizationConfig.from_env()` so it stays the source of truth.

**Tests:**
- Extend `tests/test_config.py` (or create one) — load `Settings.from_env({})` with no env vars, assert sub-config types are populated with documented defaults; assert backward-compatible env var keys still parse.
- A small lint-style test that scans `core/*.py` for module-level constants with `_S`, `_SECONDS`, `_MAX`, `_MIN`, `_THRESHOLD` in their names and fails on any new ones — keeps the next mega-class from re-growing.

---

## Critical files to modify

- [`minimappr/config.py`](minimappr/config.py) — restructure + defaults
- [`minimappr/core/localization_dispatch.py`](minimappr/core/localization_dispatch.py)
- [`minimappr/core/localization.py`](minimappr/core/localization.py)
- [`minimappr/core/ambi_atob.py`](minimappr/core/ambi_atob.py) (helper promotion only)
- [`minimappr/core/iamf_writer.py`](minimappr/core/iamf_writer.py) — bring to parity with Rust
- [`minimappr-ingest-sidecar/src/iamf_writer.rs`](minimappr-ingest-sidecar/src/iamf_writer.rs) — reference; minimal changes
- [`minimappr-ingest-sidecar/src/main.rs`](minimappr-ingest-sidecar/src/main.rs) — add stage counters to status
- [`minimappr-ingest-sidecar/src/manifests.rs`](minimappr-ingest-sidecar/src/manifests.rs) — populate `coverage_stats`
- [`minimappr/core/fusion_node.py`](minimappr/core/fusion_node.py) — metrics, timeouts, drop logging
- [`minimappr/core/ingest.py`](minimappr/core/ingest.py) — sequence-gap detection
- [`minimappr/core/node_registry.py`](minimappr/core/node_registry.py)
- [`minimappr/core/audio_buffer.py`](minimappr/core/audio_buffer.py) — add `to_json()`
- [`minimappr/core/classification.py`](minimappr/core/classification.py) — drop module constant
- [`minimappr/storage/db.py`](minimappr/storage/db.py)
- [`minimappr/cleanup_service.py`](minimappr/cleanup_service.py)
- [`minimappr/models.py`](minimappr/models.py)

## Reused utilities (don't reinvent)

- `alias_cutoff_from_positions` (after promotion) — `ambi_atob.py:232-237`
- `LocalCoordinateFrame.local_to_geo` for lat/long enrichment (already wired)
- `TrackManager.update()` in `tracking.py:122-242` for label propagation
- `_xyz_to_spherical`, `_interpolate_waypoints` in `core/iamf_object_slot.py` for IAMF position math
- `cleanup_policy_managed_files` / `cleanup_retention` in `cleanup_service.py` for TTL plumbing
- `FusionMetrics` dataclass for all new counters
- Existing structured logging via the project's logger

## Verification (test-driven, no hardware required)

Each item's tests are self-contained and runnable against synthetic inputs. The release-readiness gate is:

```
.venv/bin/python -m pytest tests/ --ignore=tests/test_soak_harness.py
.venv/bin/python -m pytest \
  tests/test_storage_retention.py \
  tests/test_fusion_node.py \
  tests/test_ingest_sequence_gap.py \
  tests/test_classification_timeout_recovery.py \
  tests/test_ui_concurrency_isolation.py \
  tests/test_birdnet_track_e2e.py \
  tests/test_wavelength_gating.py \
  tests/test_dispatch_regression.py \
  tests/test_iamf_writer.py \
  tests/test_iamf_writer_python_rust_parity.py \
  tests/test_iamf_pipeline_e2e.py \
  tests/test_config.py \
  -v
cargo test --manifest-path minimappr-ingest-sidecar/Cargo.toml
```

All tests green ⇒ server side is release-ready. Hardware-in-the-loop validation with a deployed array (real moving bird, induced firmware drops, end-to-end YouTube export) is **Phase 2**.

## Out of scope

- Firmware-side changes: NACK / sequence-retransmit protocol, WiFi credential handling in `node_config.h`, runtime provisioning. Tracked separately.
- Operator dashboard UI surface for the new metrics — backend exposes them; UI is later.
- Removing the per-node preprocessing lock at `ingest.py:217-225` — defer until benchmarked.
- SRP-PHAT / MUSIC / ESPRIT production hardening — already tracked in TODO.md.
- Multi-node phase-2 verification with real hardware.
