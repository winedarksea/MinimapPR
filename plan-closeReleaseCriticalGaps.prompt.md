## Plan: Close Release-Critical Gaps

### Objective

Ship one credible v1 workflow before widening platform scope:

- localized BirdNET detections in `birdnet_hybrid_production`
- trusted human review on top of those detections
- export-ready reviewed outputs, including linked media
- validated recording artifacts that are truthful, downloadable, and decodable

This plan is intentionally narrow. It does not prioritize broader Phase 2 platform breadth until this workflow is solid end to end.

### Recommended v1 interpretation

MinimapPR v1 should be understood as:

1. A bird-focused localization and review application.
2. A system that can produce reviewed detections and attach truthful audio and recording artifacts.
3. A system with a clear, tested export story.

It should not be treated as "all planned integrations and all future platform breadth are ready." That framing makes the roadmap less truthful and slows the release.

### Planning principles

- Reuse what already exists before adding new subsystems.
- Prefer confidence-building tests over new feature surface.
- Treat review and export as first-class product workflow, not side utilities.
- Keep the BirdNET beamformed path gated until parity and trust criteria are met.
- Keep the release story honest: mark shipped work as shipped, mark deferred work as deferred, and do not leave stale TODO items describing already-landed features as open.

### Re-baselined current state

The roadmap should be cleaned up before more feature work so the remaining gaps are visible.

#### Clearly landed from nearby code and docs

- detection audio serving is present in the FastAPI app
- heatmap backend and frontend surfaces exist
- recordings API, recording library, and download endpoints exist
- recording UI already exposes saved recording artifacts and downloads

#### Documented as landed and should be marked done or explicitly spot-verified in Phase 0

- app lifespan and `app.state` wiring
- global exception handling and HTTP error-path hardening
- HTTP API coverage baseline
- degraded-audio and sequence-gap metrics used for release validation
- IAMF object-position parameter blocks and tests
- visual recording export path
- SQLite retention and maintenance behavior, including VACUUM and auto-vacuum policy

#### Clearly still open release seams

- BirdNET hybrid trust hardening, especially Rust-versus-Python classifier-render parity
- distributed localization for four separate single-channel nodes as one localization problem
- mixed-node localization coverage for tetrahedral arrays plus omnidirectional point nodes
- human review persistence and API mutations
- export packaging built on reviewed detections
- typed review and export read paths for a more trustworthy library-facing surface
- end-to-end validation that recording and export artifacts are truthful and decodable

### Release definition of done

This plan is complete when all of the following are true:

1. `birdnet_hybrid_production` is trustworthy enough to describe as the supported v1 path.
2. Distributed localization covers both multi-single-mic and mixed-node scenarios.
3. Operators can confirm, correct, note, and optionally promote detections before export.
4. Reviewed detections can be exported in a simple eBird-oriented package with linked media.
5. The main review and export read surfaces use typed models or DTOs instead of dict-heavy returns.
6. Recording and export artifacts are validated by integration tests that check existence and decodability, not just metadata.
7. README, TODO, and release notes explicitly describe the supported v1 workflow and the deliberately deferred items.

### Delivery order

#### Phase 0 - Re-baseline and backlog cleanup

Goal: make the roadmap truthful before building on it.

Work:

- mark already-landed items as complete or verified in `TODO.md` without deleting them
- update release-facing docs so the supported v1 workflow is explicit
- separate "landed but needs confidence" from "not yet implemented"
- remove stale roadmap ambiguity around heatmap, recordings, detection audio, app lifespan, exception handlers, and HTTP API coverage

Outputs:

- `TODO.md` reflects reality
- README and release notes describe the actual current product surface
- this plan becomes the release-driving document rather than a loose outline

Exit criteria:

- no clearly-landed feature remains described as a near-term gap
- deferred breadth is explicitly named as deferred, not silently implied to be part of v1

#### Phase 1 - Make `birdnet_hybrid_production` trustworthy

Goal: raise confidence in the current bird path rather than expanding its scope.

Work:

- extend existing BirdNET hybrid tests to compare Rust-generated classifier renders against the Python reference path
- keep beamformed classification gated by default until parity checks are green
- validate current localization and classification split, provenance fields, and localization-band behavior
- confirm that reported detections remain canonical and truthful when both omni and localized paths are available

Outputs:

- explicit Rust-versus-Python parity coverage
- a clear gating rule for when beamformed classification can be re-enabled by default
- release-ready confidence in the existing BirdNET path

Exit criteria:

- existing BirdNET hybrid tests stay green
- new parity tests are green
- provenance and classification-path behavior is covered by regression tests
- the default gate remains in place until those checks pass

#### Phase 2 - Distributed localization path — DONE (re-scoped 2026-06-16)

Goal: make the primary multi-node deployment story technically true.

Status: the grouping/combine mechanism is implemented on both ingest paths; the original
"current gap" below is resolved. The remaining work is operational (firmware + live
verification), not structural.

What was actually built (supersedes the old "current gap"):

- the Rust DSP worker groups manifests sharing a `cluster_id` into one shared buffer
  (`resolve_cluster_buffer_routing`, dsp_worker.rs:1982); SRP gates on `buffer_channel_count`
  (dsp_worker.rs:1124), so four one-channel nodes become one localization problem
- cross-node timing alignment uses node packet timestamps; receipt-time fallback is refused for
  TDOA correctness (dsp_worker.rs:2024)
- mixed tetrahedral + point topologies are supported and tested
- Python pools sensors across nodes in `_localize_candidate` (fusion_node.py:1020); cluster
  scoping via `cluster_aware_localization` (now default ON)

Exit criteria — MET (unit) / OUTSTANDING (live):

- [x] test proving localization across four separate one-channel nodes:
  `clustered_single_channel_manifests_share_a_tetrahedral_localization_buffer`
  (dsp_worker_tests.rs:1071) + `test_cluster_aware_localization_uses_cluster_sensor_scope`
  (tests/test_cluster_aware_integration.py:122)
- [x] test proving a mixed tetrahedral-plus-point-node scenario:
  `mixed_clustered_tetrahedral_and_point_manifests_share_one_localization_buffer`
  (dsp_worker_tests.rs:1161)
- [x] no regression in the hybrid bird flow (existing suite stays green)
- [ ] **live** re-verification across four physical nodes with distinct GPS positions —
  see `docs/distributed_localization_verification.md` (supersedes the 2026-05-30 test). This,
  plus committing/flashing the firmware per-node position fix, is the only remaining work.

#### Phase 3 - Add the minimal human review workflow

Goal: stop treating BirdNET output as final truth.

Work:

- add review-state persistence for detections and, where useful, tracks
- support confirmation, rejection, label correction, notes, and optional training-set promotion
- add small, first-class API mutations for review updates
- keep the workflow minimal and operationally useful instead of building a large curation subsystem

Outputs:

- reviewable BirdNET detections
- durable review state that downstream export can trust
- HTTP-level coverage for the review mutation flow

Exit criteria:

- operators can confirm a detection, correct its label, attach notes, and optionally mark it for promotion
- review state survives round-trips through persistence and APIs
- HTTP tests cover the main review operations

#### Phase 4 - Build export on top of reviewed detections

Goal: make export a consequence of confirmed review, not a parallel classification path.

Work:

- start from confirmed BirdNET detections only
- package eBird-oriented export output plus linked audio or media handoff
- keep this simple and manual-friendly rather than attempting full automated external upload
- make detection audio and related artifact serving part of the end-to-end path

Outputs:

- reviewed-detection export package
- linked media handoff that matches the exported detections
- one end-to-end review-to-export scenario in tests

Exit criteria:

- a confirmed detection can flow through export packaging with matching metadata and media
- HTTP or end-to-end tests cover the path from review mutation to export retrieval
- the workflow is clearly documented as the supported v1 export path

#### Phase 5 - Harden the package where the v1 workflow touches library surfaces

Goal: get obvious trust wins without treating this as a full public API stabilization effort.

Work:

- convert the main review and export read paths away from dict-heavy returns and into typed models or DTOs
- define the supported entry points for the release-critical workflow
- add one or two examples or documented flows that exercise bird review, export, and recording artifact retrieval

Outputs:

- typed read surfaces where consumers most need reliability
- a smaller, clearer public surface for the release-critical path
- examples that demonstrate the intended workflow

Exit criteria:

- review and export services no longer depend on loose dict-heavy read contracts for their core flow
- at least one documented example covers the bird review and export path
- no deep or speculative refactor is required to claim v1 readiness

#### Phase 6 - Finish confidence-building validation around recordings and exports

Goal: prove the advertised artifacts are real.

Work:

- add integration checks across `ffmpeg` and `iamf-tools` availability
- verify that every artifact listed in the recordings library actually exists
- verify those artifacts are decodable and consistent with the stored metadata
- include sync and truthfulness checks on real outputs where practical

Outputs:

- stronger artifact truthfulness guarantees
- fewer "metadata says export exists but file is broken" failure modes
- a cleaner recording and export story for release notes

Exit criteria:

- integration tests verify existence and decodability of every advertised artifact format
- recording library metadata matches actual on-disk deliverables
- IAMF, visual MP4, and linked download paths are part of the validation story

### Cross-cutting rules for all phases

- Do not reopen broad platform breadth while these phases remain incomplete.
- Do not remove the BirdNET beamformed-classification gate until Phase 1 parity checks are green.
- Prefer extension of the current `tests/test_birdnet_hybrid_production.py` and `tests/test_http_api.py` surfaces before inventing new testing harnesses.
- Reuse the current `DetectionEvent` schema, detection-audio endpoint, recordings API, app lifespan wiring, and existing frontend surfaces wherever possible.

### Recommended PR slices

1. Re-baseline docs and roadmap status.
2. BirdNET parity tests and beamformed-classification gate hardening.
3. Distributed localization for four single-mic nodes.
4. Mixed-node localization coverage.
5. Minimal review-state persistence and review mutation APIs.
6. Reviewed-detection export packaging.
7. Typed review and export read paths.
8. Recording and export artifact integration validation.

This order keeps the work visible, testable, and reversible. It also ensures that export is built on confirmed detections rather than on unreviewed classifier output.

### Primary implementation surfaces

BirdNET hybrid and classification:

- `/Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/config.py`
- `/Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/core/classification.py`
- `/Users/colincatlin/Documents-NoCloud/MinimapPR/tests/test_birdnet_hybrid_production.py`

Distributed localization:

- `/Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr-ingest-sidecar/src/dsp_worker.rs`
- `/Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/core/ingest.py`
- `/Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/core/fusion_node.py`

Review and export workflow:

- `/Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/models.py`
- `/Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/main.py`
- `/Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/storage/db.py`
- `/Users/colincatlin/Documents-NoCloud/MinimapPR/tests/test_http_api.py`

Recordings and artifact validation:

- `/Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/core/iamf_pipeline.py`
- `/Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/core/iamf_writer.py`
- `/Users/colincatlin/Documents-NoCloud/MinimapPR/tests/test_iamf_writer.py`
- `/Users/colincatlin/Documents-NoCloud/MinimapPR/tests/test_iamf_pipeline_e2e.py`

Existing frontend surfaces that should be reused, not rebuilt:

- `/Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr-frontend/src/pages/analysis/heatmap.rs`
- `/Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr-frontend/src/pages/audio/recording.rs`
- `/Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr-frontend/src/pages/audio/recordings_library.rs`

### Verification

1. Keep the existing BirdNET hybrid suite green and add explicit Rust-versus-Python classifier-render parity coverage.
2. Add a distributed four-single-mic localization test and a mixed tetrahedral-plus-point-node localization test.
3. Add HTTP/API tests for review-state mutation, label correction, notes, optional training promotion, and export packaging.
4. Add one end-to-end scenario covering localized bird detection, human confirmation, export packaging, and download of linked media artifacts.
5. Run recording and export integration checks that verify every advertised artifact exists and is decodable.
6. Update README, TODO, and release notes so the supported v1 path and deferred breadth are explicit.

### Decisions

- Included scope: release-critical bird workflow, distributed localization correctness, review and export, and a small amount of library-facing type hardening where it directly improves trust.
- Excluded for now: Home Assistant integration, TAK and CoT, additional federation breadth, advanced tracking work, text intelligence, and effector systems.
- BirdNET beamformed classification stays gated by default until parity checks justify changing that default.
- Export is review-driven. Unreviewed BirdNET output is not the release contract.
- This is not a full public API stabilization effort. Take the obvious wins only.

### What this plan should change in project posture

After this plan is executed, MinimapPR should read as a focused product with one credible v1 workflow, not as a broad architecture document with an unclear release state. The release story should become simpler:

- detect and localize birds
- review and correct detections
- export reviewed results with linked media
- trust the advertised recordings and downloads

Everything else can stay on the roadmap, but it should no longer blur what "ready" means.