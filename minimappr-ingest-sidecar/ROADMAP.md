# Ingest Sidecar Build Target

Build the ingest sidecar around one storage and processing contract: a Rust capture service appends observations into tmpfs-backed, mmap-readable per-stream rolling journals, publishes typed handles, and ACKs as soon as the append is committed in RAM. Heavy DSP, classification-window rendering, durable persistence, and downstream policy stay off the capture path. This keeps the current MinimapPR retention rules, late-frame semantics, coverage semantics, and BirdNET hybrid expectations while removing file-per-packet churn and avoiding raw operational audio writes to durable disk. Per-packet observation records are not durably written on ingress; they are materialized only when a promotion, review, export, or fault workflow explicitly requires them.

The capture boundary is strict: parse, validate, copy into the tmpfs journal, publish the handle, ACK. The ACK never waits on DSP, Python processing, database writes, fsync of raw audio, or artifact promotion.

## Higher Level Goals

Use Cases: whatever the design, it must support studio-grade ambisonics and IAMF (object tracking recording like Dolby Atmos) recording, speed to text processing of audio inputs, high precision TDOA localization, non-audio data capture (such as temperature and accelerometers), and prepare high quality audio for accurate classification.

Performance Requirements: minimal packet loss (0.01% is the goal), high efficiency processing, minimal writing to disk, and high reliability. Logs and errors need to never build up such that they may eventually swamp the system. However the system does not need to be particularly crash tolerant, if 30 seconds of audio are lost on a crash and reset, this is acceptable, and a worthwhile tradeoff for smaller footprint and simpler system design.

Aim to keep files beneath 500 lines where practical.
Only use widely-used, well-maintained dependencies.

## Status Snapshot

This roadmap should now be read as a status-aware build target rather than as a claim that every downstream migration is already complete. The ingest journal slice is substantially implemented. The two major workstreams that remain outside that closure are the full production Rust DSP worker orchestration and the broader file-size refactor of the existing large Python backend.

Implemented and verified now:

- Rust capture-plane sidecar with journal mode, per-stream segment rotation, mmap-readable handles, admission reserve enforcement, cursor-aware eviction, and pin lease APIs.
- Python journal consumer with durable cursor and exception storage for journal replay.
- Baseline artifact promotion from a journal handle into durable observation and artifact records.
- Derived-cache and manifest stores that define the Rust-to-Python DSP handoff contract.
- Warm-restart sequence recovery, torn-tail index salvage, and cold-restart epoch advancement for the journal lineage.

Partially implemented or intentionally deferred:

- `dsp.rs` covers buffer and parity primitives plus the manifest boundary, but it does not yet replace the current Python localization and classification orchestration end-to-end in production.
- The current Rust worker scaffolding is not yet sufficient for `MINIMAPPR_RUNTIME_PROFILE=birdnet_hybrid_production`: that profile expects `srp_phat`-resolved localized detections plus classifier-ready audio renders with explicit spatial provenance, while the sidecar currently stops short of that contract.
- Multichannel journal storage already supports array-coherent audio capture, but IAMF/ADM/Dolby-class export remains out of scope for this build.
- The transport-neutral envelope can carry non-audio metadata, but the higher-level Python and policy flows for non-audio capture remain follow-on work.
- The sidecar removes file-per-packet churn and bounds journal and cache growth, but the dedicated burst and load benchmark proving the 0.01% packet-loss target remains an open operational gate.
- The broader large-backend decomposition remains separate work and should not block the journal-sidecar slice.

## Goal Alignment

This build already serves the higher-level goals where the ingest boundary is the deciding factor:

- minimal disk writes via tmpfs-backed operational journals and transient derived caches
- reliable bounded growth via admission reserves, eviction rules, lease expiry, and durable consumer cursors
- high-fidelity classifier preparation via mmap-reopenable raw handles, derived-cache entries, manifest provenance, and promotion-time observation materialization
- multichannel capture readiness via per-stream rings that keep channel-coherent media together instead of flattening it into per-channel files

The remaining higher-level goals depend mostly on downstream orchestration rather than on the capture boundary itself:

- full Rust DSP worker orchestration for TDOA, localization, and classifier rendering at production depth
- `birdnet_hybrid_production` readiness where Rust owns ingest and initial audio processing while Python remains the BirdNET inference, tracking, rules, and durable-state plane
- broader speech-to-text and non-audio consumer workflows
- IAMF/ADM/Dolby-class export and scene packaging

## Core Contract

### Observation envelope

The journal entry is a typed in-memory observation envelope plus a journal handle. For audio observations, every committed entry carries:

- `observation_id`
- `journal_epoch`
- `journal_sequence`
- `node_id`
- `stream_id`
- `sensor_type`
- `source_type`
- `transport`
- TOA
- TOR
- `ingest_received_ns`
- `time_quality`
- `clock_domain`
- `sync_source`
- `clock_correction_ns` or `clock_drift_ppm`
- `sample_rate_hz`
- `channel_count`
- `channel_layout`
- `sample_index_start`
- `sample_count`
- `geometry_version`
- `orientation_version`
- `calibration_version`
- `retention_hint`
- `payload_codec`
- `payload_byte_range`
- `integrity_hash`

`journal_epoch` is a monotonically increasing journal namespace identifier assigned when a live journal lineage starts; a cold restart advances the epoch so `journal_sequence` values remain unambiguous within and across recoveries.

`sensor_type` and `transport` are first-class fields in the build target, not deferred schema cleanup. These fields live in the journal entry and short-lived indices by default; they become durable only if a promotion path materializes a retained observation record.

### Journal handle

The media handle resolves to a committed byte range inside a tmpfs-backed, mmap-readable raw-media segment owned by a per-stream rolling ring keyed by `node_id + stream_id`. Each handle includes enough information to reopen the payload without scanning unrelated streams:

- `journal_epoch`
- `segment_id`
- `stream_key`
- `payload_offset_bytes`
- `payload_length_bytes`
- `sample_index_start`
- `sample_count`
- `integrity_hash`

### Storage classes

There are exactly three storage classes:

1. Operational rolling journal: tmpfs-backed, mmap-readable, bounded by memory budget, non-durable raw audio.
2. Transient derived cache: bounded cache for candidate windows, classifier-ready renders, and localized review audio that have not been retained.
3. Promoted retained artifacts: durable outputs governed by existing retention tiers and cleanup rules.

Raw operational audio lives only in the operational journal unless an explicit promotion or capture-session policy pins and exports it.

## Storage Model

### Commit to tmpfs plus mmap

The default and required IPC/storage mechanism for this build is tmpfs-backed segment files opened through mmap. This is not a fallback. It is the only mechanism that satisfies both the Rust capture-plane requirements and the existing Python consumer boundary. Pure in-process memory or anonymous shared-memory designs are out of scope for this build because the Python consumer requires file-backed reopening semantics.

### Per-stream rings

Each `node_id + stream_id` pair owns its own append ring. Rings isolate head-of-line blocking, make eviction local, and keep sample-index ordering coherent for multichannel sources. Array-node audio remains phase-coherent multichannel media in one stream ring rather than being flattened into per-channel artifacts.

### Segment layout

Each stream ring is composed of sealed tmpfs segments with compact headers:

- `segment_id`
- `journal_epoch`
- `stream_key`
- `first_journal_sequence`
- `last_journal_sequence`
- `first_received_ns`
- `last_received_ns`
- optional first and last TOA
- optional first and last TOR
- `entry_count`
- `payload_bytes`
- `pin_count`
- `sealed`

Sealed-segment metadata keeps replay lookup, pinning, and cleanup bounded by segment count rather than entry count.

### Consumer state replaces receipt files

Receipt files are replaced by a durable consumer-state store with two tables:

1. Consumer cursor table: one row per consumer and stream, storing the monotonic watermark (`last_fully_processed_journal_sequence`) and last update time.
2. Consumer exception table: compact rows for entries that failed permanently, expired before processing, or were intentionally skipped.

The raw journal is allowed to disappear. Consumer watermarks and exception rows are durable and must survive process restarts and host restarts.

### Frame dedup disposition

Ingress dedup remains a separate lightweight control-plane record rather than a durable observation record. Keep a compact dedup table keyed by `node_id + boot_session + frame_sequence` so the system can reject replays of the same firmware frame without reintroducing per-channel or per-observation writes. This table exists only to preserve ingest idempotence across retries and restarts; it is not a retained packet log.

### Short-lived ingress indices and aggregate metrics

Instead of durably writing one observation row per packet on ingest, the sidecar maintains bounded short-lived indices over the live journal:

- stream-local sequence to segment-range index
- time-range index keyed by `node_id + stream_id`
- active pin index
- transient manifest index for candidate detections and derived windows

These indices exist only to support live consumers, promotion lookups, and fault handling while the underlying journal ranges remain available.

Durable operational visibility comes from aggregate metrics and rollups rather than per-observation rows:

- ingress bytes/sec and packets/sec by node and stream
- append latency and ACK latency histograms
- consumer lag and watermark deltas
- eviction counts and pin-pressure counters
- overload rejections and degraded-mode counters
- clock-quality and gap-rate summaries

Those metrics are durable as counters, histograms, and time-bucketed summaries, not as packet-by-packet audit logs.

## Capture and Consumer Behavior

### Capture-plane ACK rule

The sidecar hot path does only this:

1. Parse and validate the request envelope.
2. Reserve byte range and next `journal_sequence` in the target per-stream ring.
3. Copy payload bytes into the tmpfs segment.
4. Publish the journal entry and journal handle as committed.
5. ACK the sender.

If any of those steps fail before publish, the append is rolled back and the request is not ACKed.

### Rust DSP plane

Rust DSP workers tail committed journal ranges and emit typed manifests plus derived-audio handles. They own rolling buffering, trigger evaluation, sibling-node grace handling, localization-window extraction, TDOA or GCC-PHAT estimation, localization, and classifier-audio rendering. They do not run on the capture executor.

Current status: the repository contains the journal, derived-cache, manifest, and buffer/parity primitives for this plane, but the full production worker orchestration that would replace the current Python path is still follow-on work.

Near-term cutover requirement: for `MINIMAPPR_RUNTIME_PROFILE=birdnet_hybrid_production`, Rust workers must be able to handle ingest and initial audio processing without waiting for a full Rust BirdNET migration. In that staged design, Rust owns journal append, rolling buffer parity, array-local localization, classifier-render preparation, and review-render preparation; Python continues to own BirdNET inference, tracking, rules, alerting, and durable observation materialization.

### Python boundary

Python consumes:

- journal metadata plus tmpfs-backed handles for raw-frame access where needed
- DSP manifests plus memory handles for classifier-ready audio and localized review audio
- promoted artifact references after retention or review decisions

This keeps Python as the rules, tracking, database, alerting, and BirdNET inference plane while raw packet ingestion stops depending on file-per-packet spooling. Python opens the tmpfs-backed segments through file-backed handles rather than through direct shared-memory access, and only writes durable observation records when a promotion, review, export, or fault workflow materializes them from the live journal entry or derived manifest.

## Pinning, Eviction, and Memory Limits

### Pin and unpin protocol

Eviction is safe only if raw ranges and derived windows have explicit ownership. Every handle can be in one of three retention states:

1. Unpinned: eligible for normal ring eviction.
2. Soft-pinned: temporarily retained for an active consumer or DSP task.
3. Hard-pinned: retained for operator-requested capture sessions, artifact promotion, or active export.

Pinning rules:

- Rust DSP pins source ranges while window extraction or rendering is in progress.
- Python pins derived handles while classifier, review, or persistence work is in progress.
- Operator capture sessions and promotion flows create hard pins until export or retention expiry completes.
- Pins are reference-counted and tied to lease expirations so crashed consumers cannot pin forever.

Unpin happens on successful completion, explicit abandonment, or lease expiry. Eviction may reclaim only fully unpinned segments below the oldest active pin for that stream.

### Memory sizing formula

Size the operational journal with an explicit formula:

$$
\\text{journal\_bytes} = \left(\sum_{streams} \text{peak\_ingress\_bytes\_per\_sec} \times \text{target\_lag\_seconds}\right) \times \text{burst\_multiplier} + \text{pin\_reserve\_bytes} + \text{metadata\_reserve\_bytes}
$$

For PCM-like audio streams:

$$
\\text{ingress\_bytes\_per\_sec} = \text{sample\_rate\_hz} \times \text{channel\_count} \times \text{bytes\_per\_sample}
$$

Required configuration:

- per-stream minimum reservation
- total tmpfs journal budget
- transient derived-cache budget
- hard pin reserve
- admission reserve for one maximum-size append per active stream
- target consumer lag window in seconds
- burst multiplier

### OOM and overload behavior

The sidecar enforces this degradation order:

1. Drop optional DSP work and stop creating new derived windows.
2. Reduce localization cost or spatial weighting work.
3. Fall back to cheaper classifier-audio rendering paths.
4. Evict oldest unpinned journal data above the consumer watermark.
5. Reject new ingest with a clear overload error only when the append path cannot preserve the admission reserve.

Capture never blocks on durable I/O to make progress. If admission reserve is exhausted, the request is rejected rather than partially accepted.

## Durability and Recovery

### Durable versus non-durable state

Durable state:

- consumer cursor table
- consumer exception table
- frame dedup table
- promoted retained artifacts
- policy and cleanup state
- aggregate health metrics and rollups

Non-durable state:

- raw operational journal segments in tmpfs
- per-packet observation envelopes that have not been promoted
- short-lived ingress indices
- transient derived-cache entries that were not promoted
- in-flight DSP work

### Named recovery modes

The build target names three recovery modes explicitly:

1. Warm process restart: tmpfs journal still exists, the sidecar rebuilds in-memory indexes from segment headers, resumes from the last committed segment state, and continues from the durable consumer cursors.
2. Cold host restart: tmpfs journal is gone, unpromoted raw audio is intentionally lost, durable cursors remain, and consumers resume from the next live append in a new `journal_epoch`.
3. Torn-tail recovery: the sidecar finds a partially written or unpublished tail range after a crash, truncates back to the last committed offset, and preserves sequence monotonicity inside the recovered epoch.

The product accepts loss of unpromoted raw audio, unpromoted observation envelopes, and short-lived indices across cold restart. It does not accept loss of durable cursor state, frame dedup state, promoted artifacts, active policy state, or aggregate operational metrics.

## Semantic Requirements

### Audio-buffer parity

Rust DSP behavior must match the existing Python audio-buffer semantics in `minimappr/core/audio_buffer.py`: timeline anchoring, late-frame snap tolerance, overlap merge behavior, zero-fill for gaps, partial trailing windows, full-window availability checks, coverage accounting, and sample-rate-reset handling. These semantics remain the oracle for BirdNET hybrid and snippet fidelity.

### BirdNET hybrid render contract

Classifier-ready renders must carry explicit spatial provenance rather than implicit path assumptions:

- steering solution
- classifier source node
- spatial blend mode
- effective spatial band
- confidence
- fallback reason

The classifier contract is broadband audio with spatial weighting only where the array can localize reliably.

### `birdnet_hybrid_production` runtime-profile contract

The current Python runtime profile is the compatibility target for the Rust worker migration:

- `classifier_backend = birdnet`
- `localization_algorithm = srp_phat`
- `localization_strategy = fixed`
- localized detections are expected to report `localization_method = srp_phat`
- `beamformed_classification_enabled = False` remains the conservative default until journal-derived render parity is proven

That means the Rust sidecar cannot stop at pairwise GCC-PHAT output. To be considered ready for this profile, the Rust workers must provide:

1. array-local localization compatible with the `srp_phat` expectation for Sirith tetrahedral nodes
2. point-node and degraded-localization omni fallback without breaking detection provenance
3. classifier-ready render handles that preserve full-band BirdNET input while applying spatial weighting only where the array has useful aperture
4. explicit BirdNET provenance describing steering solution, blend mode, spatial band, confidence, and fallback reason

The intended hybrid render for Sirith tetrahedral nodes remains:

- array-steered band approximately 1 kHz to 3.4 kHz
- omni contribution outside the spatially reliable band so BirdNET still receives broadband input
- optional high-pass conditioning above roughly 100 Hz before the spatial blend when low-frequency rumble would otherwise dominate the render

Migration rule: keep the profile's current default behavior stable until the Rust-generated manifests and renders satisfy `tests/test_birdnet_hybrid_production.py`. Only after that parity is demonstrated should array-node hybrid renders become the default production path.

### Promotion policy

Candidate windows are transient by default. Durable promotion happens only for:

- retained classifier input audio
- retained localized review audio
- explicit raw capture bundles
- scene manifests and other policy-approved exports

Promotion reuses the existing retention tiers, `large_artifacts` storage path, cleanup policy, and cleanup service. When promotion occurs, the system lazily materializes the retained observation metadata needed for provenance from the live journal entry, DSP manifest, and pinned source ranges rather than from a preexisting per-packet database row.

## Verification Status

| Gate | Status | Evidence | Remaining |
|------|--------|----------|-----------|
| Capture-plane ACK and raw-handle readability | Done at unit and integration level | `src/main.rs::{returns_accepted_after_manifest_is_ready,journal_mode_appends_segment_and_index_files,journal_overload_returns_503}`, `src/ingest_backend.rs::journal_payload_handle_reopens_with_mmap_and_verifies_hash` | Keep the load benchmark separate from the correctness gate |
| Warm and cold recovery semantics | Done | `src/ingest_backend.rs::{journal_recovers_next_sequence_after_restart,journal_recovers_torn_index_tail_by_truncating_uncommitted_payload,journal_recovers_complete_unterminated_index_entry_as_committed,journal_advances_epoch_after_cold_restart}` | None at the correctness gate; load-loss tolerance remains under benchmarking |
| Load and packet-loss benchmarking | Open | No dedicated committed benchmark yet | Measure target burst rates with DSP disabled, degraded, and enabled |
| Memory pressure, reserve enforcement, and eviction safety | Partial | `src/ingest_backend.rs::{journal_rejects_when_no_evictable_segment_can_preserve_reserve,journal_evicts_oldest_sealed_segment_once_cursor_covers_it,pin_lease_blocks_segment_eviction_until_released}`, `src/derived_cache.rs::derived_cache_evicts_oldest_entry_to_preserve_budget` | Add consumer-lag and lease-expiry stress coverage |
| Cursor-store durability | Done for the Python consumer slice | `tests/test_ingest_spool_consumer_journal.py` covers processed, failed, and hash-mismatch journal handling with durable cursor and exception updates | Add a broader end-to-end cold-loss replay scenario if needed |
| Cross-language buffer parity | Partial | `src/dsp.rs::{late_frame_replaces_zero_gap,trailing_window_returns_partial_audio,overlap_merge_overwrites_existing_samples,explicit_sample_index_gaps_mark_missing_coverage,explicit_contiguous_samples_reanchor_after_large_timestamp_correction}` | Add zero-fill edge cases and sample-rate reset parity against the Python oracle |
| BirdNET hybrid production parity | Open | `tests/test_birdnet_hybrid_production.py`, `config.py` runtime-profile defaults, and the current placeholder BirdNET provenance show the required contract more clearly than the current Rust worker output | Add `srp_phat`-compatible array-local localization, hybrid/omni classifier-render handles, explicit fallback semantics, and manifest provenance before enabling Rust-generated hybrid renders by default |
| Promotion materialization | Done for the current journal promotion slice | `tests/test_journal_promotion.py` covers classifier-input promotion plus retained detection references, localized review artifacts, fault capture, and explicit raw-capture bundle cases without pre-packet database writes | Keep extending end-to-end workflow wiring as higher-level consumers land |

Current verification baseline:

- `cargo test -q` in `minimappr-ingest-sidecar`
- `pytest tests/test_sidecar_startup.py tests/test_http_api.py::test_system_diagnostics_includes_sidecar_health tests/test_journal_promotion.py -q`
- `pytest tests/test_ingest_spool_consumer_journal.py tests/test_journal_promotion.py -q`

## Decisions

- Included scope: tmpfs+mmap rolling journal, per-stream rings, durable cursor tables, Rust DSP workers, Python manifest consumption, artifact promotion, cleanup alignment, optional raw capture sessions, and transport-neutral observation envelopes.
- Excluded scope: pure in-process shared-memory transport, durable raw-audio journaling to physical disk, moving BirdNET inference into Rust, and IAMF/ADM/Dolby export work.
- Spool mode fate: legacy spool mode is deprecated and should be removed from production once the tmpfs journal path satisfies the verification gates; it may remain only as a temporary test or rollback harness during transition.
- ACK policy: ACK after parse, validate, copy to tmpfs, publish handle. Do not wait on DSP, Python, DB, or disk.
- Retention policy: raw operational audio is ephemeral unless pinned and promoted by policy, review, or capture-session request.
- Provenance policy: keep TOA, TOR, `time_quality`, geometry, orientation, and calibration immutable in source journal entries; lazily materialize durable observation records only for promoted detections, retained artifacts, exports, and explicit fault records.

## Key Files

| File | Role |
|------|------|
| `minimappr-ingest-sidecar/src/main.rs` | Capture-plane ACK boundary |
| `minimappr-ingest-sidecar/src/ingest_backend.rs` | Journal append, segment rotation, pinning, and cursor integration |
| `minimappr-ingest-sidecar/src/dsp_worker.rs` | Rust worker scaffold for journal-driven initial audio processing |
| `minimappr-ingest-sidecar/src/manifests.rs` | Rust-to-Python DSP manifest and BirdNET provenance contract |
| `minimappr/api/journal_reader.py` | Python mmap-backed journal-handle reopen and integrity verification |
| `minimappr/api/journal_state.py` | Durable consumer cursor and exception storage |
| `minimappr/api/spool_consumer.py` | Python consumer boundary for journal handles and manifests |
| `minimappr/api/transports.py` | Python handoff boundary to preserve provenance and ingestion semantics |
| `minimappr/core/journal_promotion.py` | Journal-handle promotion into durable artifacts and observations |
| `minimappr/core/ingest.py` | Promotion-time observation materialization, preprocessing, and trigger orchestration |
| `minimappr/core/audio_buffer.py` | Semantic oracle for late-frame and coverage behavior |
| `minimappr/core/fusion_node.py` | Localization and classification orchestration |
| `minimappr/core/classification.py` | Classifier-input expectations and fallback behavior |
| `minimappr/core/assembly.py` | Snippet and artifact assembly path |
| `minimappr/storage/db.py` | Durable promoted metadata, artifacts, cursors, and aggregate metric storage |
| `minimappr/cleanup_policy.py` | Retention and export policy |
| `minimappr/cleanup_service.py` | Cleanup orchestration |
| `minimappr/config.py` | Journal budgets, cache budgets, queue sizing, and retention settings |
| `tests/test_ingest_spool_consumer_journal.py` | Journal consumer cursor and exception coverage |
| `tests/test_journal_promotion.py` | Promotion materialization coverage |
| `tests/test_birdnet_hybrid_production.py` | Core behavioral parity suite |
