## Plan: Long-Range Localization Hardening

Replace the current near-field-only assumptions with an observability-aware long-range localization path that keeps Cartesian outputs but carries anisotropic uncertainty through detection and tracking. The confirmed short-range failure is not acoustic range: Python MUSIC/ESPRIT currently bound tight-array range to `max(20.0, aperture * 450.0)`, which caps a roughly 10 cm tetra around 45-50 m, while Python/Rust SRP-PHAT search only evaluates a small Cartesian box around the microphones. The recommended fix is a shared direction-first tight-array solver, optional bounded range refinement, explicit covariance, and rollout gates that prove the live audio path still stays bounded and non-blocking.

**Steps**
1. Phase 1 — Lock runtime safety contract, root cause, and rollout guardrails.
   - Add a small architecture note or test comment documenting the two current range limiters: `advanced_localization.py` MUSIC/ESPRIT `aperture * 450` range bound and SRP-PHAT Cartesian grid bounds from `localization_search_padding_m`; Rust `srp_phat.rs` has the same bounded-grid behavior.
   - Define non-regression constraints before algorithm work: no new blocking I/O in the live DSP or localization path, no extra disk writes before detection promotion, bounded queue sizes remain intact, localization heartbeat cadence stays at the configured interval, and classifier render cadence stays unchanged.
   - Add a focused long-range config surface that separates near-field Cartesian search settings from far-field settings. Include far-field max range, angular resolution, radial prior/default range, covariance mode, association gate scaling, confidence thresholds, and per-frame compute budget. Keep current defaults as the default behavior until validation passes.
   - Define one shared Python/Rust localization contract before implementation: input sensor geometry and TDOAs, output `position_m`, `steering_direction`, `position_covariance_m2`, `range_observability`, `residual_rms_seconds`, confidence, and algorithm provenance. Use this contract as the parity checklist.
   - Add metrics needed for safe rollout: localization stage timing/p95, raw-manifest queue depth and saturation, DSP compute budget overruns, omni fallback ratio, low-confidence fallback reason counts, localization coverage failures, association miss rate, and near-field vs far-field confidence/range-observability histograms.
   - Depends on nothing; blocks all behavior changes.
2. Phase 2 — Fix the localization model in Python. Depends on step 1. Parallel with step 3.
   - Replace the bounded-box SRP behavior in the Sirith and tight-array path with an angular or direction-first search that scales with angle, not cubic range volume. Do not fix long range by raising `localization_search_padding_m`.
   - Remove the MUSIC and ESPRIT aperture-based hard range cap and replace it with a configurable observability-aware range solver. Keep Cartesian `position_m`, but derive it from direction plus range estimate and emit anisotropic covariance that is tight laterally and elongated along the ray when range is weakly observed.
   - Treat range as weakly observed for single-node tight tetra results unless residual curvature supports it. When range is unobservable, use a configured radial prior/default range for Cartesian projection and make the covariance honestly large along the ray, rather than reporting a precise far-away point.
   - Promote uncertainty into the core localization result. Add covariance or equivalent uncertainty fields to LocalizationResult and LocalizationBranch so downstream components do not have to infer uncertainty from track state later.
   - Wrap fragile least-squares and covariance calculations with explicit `LinAlgError`/ill-conditioning handling. Degrade to direction plus large radial covariance when the math is singular; do not silently clamp to a fake precise range.
   - Keep GCC-PHAT as the fallback path, but do not use it as the primary far-field model for tight arrays.
3. Phase 3 — Bring the Rust sidecar to parity without violating real-time constraints. Depends on step 1. Parallel with step 2.
   - Replace the sidecar’s bounded Cartesian SRP search with the same direction-first or coarse-to-fine strategy used in Python, rather than simply increasing search padding. Do not permit large O(range^3) grids in the live DSP worker.
   - Keep heartbeat localization cheap and deterministic: bearing-first on normal heartbeat frames, with optional heavier range refinement only on classifier-render frames, offline replay, or when the measured compute budget has headroom.
   - Preserve current real-time invariants: localization cadence, forced localization on 4-channel classifier-render frames, bounded raw-manifest channels, propagated backpressure, and no async storage waits in the memory path.
   - Extend the sidecar localization payload and Python manifest bridge to carry uncertainty data alongside position, confidence, and algorithm provenance.
   - Keep BirdNET render selection and audio window semantics unchanged except where low-confidence localization should now use uncertainty-aware thresholds instead of short-range-tuned constants. In particular, do not let far-field covariance force needless omni fallback when bearing is usable for render steering.
4. Phase 4 — Retune fusion, fallback, and tracking around honest uncertainty. Depends on steps 2 and 3.
   - Update FusionNode, `LocalizedClassifierRenderRequest`, and DetectionAssembler to propagate localization covariance directly into DetectionEvent instead of only inheriting covariance from an existing track.
   - Replace the fixed 8 m nearest-neighbor association with a covariance-aware gate. Prefer Mahalanobis or a scaled Euclidean gate derived from `position_covariance_m2`, capped by configured minimum/maximum gates so far-field tracks can remain coherent without making near-field association sloppy.
   - Update tracking filter measurement covariance to consume per-detection covariance when present. The current Kalman path uses a fixed measurement covariance, which will over-trust weak far-field range unless changed.
   - Revisit confidence and fallback logic in both Python and the sidecar. Low-confidence long-range localizations should degrade gracefully based on uncertainty and observability, not because the solver never searched the right region.
   - Harmonize Python and Rust runtime defaults that currently diverge or remain confusing, including trigger and localization gating defaults, while preserving intentional timestamp-policy differences already documented in Rust.
5. Phase 5 — Add regression coverage and performance acceptance tests before enabling by default. Depends on steps 2 through 4.
   - Extend Python and Rust synthetic localization tests to include Sirith tetra sources at 20 m, 50 m, 100 m, and at least one off-axis/elevated case, with acceptance based on angular error plus covariance calibration rather than point error alone.
   - Add paired near-field regressions so the new path cannot silently damage the existing 0.2 m to 5 m behavior or BirdNET hybrid render correctness.
   - Add Python/Rust parity fixtures using the same mic geometry, source positions, sample rate, and synthesized delays. Assert matching sign conventions, comparable steering direction, comparable residuals, and matching uncertainty semantics.
   - Add end-to-end hybrid sidecar tests that preserve cadence and backpressure invariants under load and verify that queue saturation, manifest loss, and classifier render timing do not regress.
   - Run a soak benchmark against real-time streaming: queue depth remains bounded, HTTP 503 behavior matches baseline, localization cadence remains intact, and no new disk writes occur before localized and classified promotion.
6. Phase 6 — Flip defaults conservatively and document operating envelopes. Depends on step 5.
   - After the benchmarks pass, update runtime profile defaults so tight Sirith arrays no longer force short-range-only fixed SRP behavior. Prefer geometry-aware or a dedicated tight-array far-field profile with explicit uncertainty semantics.
   - Document what remains physically weakly observed on a single 5-10 cm tetra: angular precision should improve first, radial uncertainty can stay large at long distance, and multi-node fusion remains the path to narrow range uncertainty further.
   - Update deployment docs with tuning guidance for long-range mode, expected covariance shapes, and rollback knobs.

**Relevant files**
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/core/advanced_localization.py — Python SRP grid bounds, MUSIC and ESPRIT range fitting, and the best seam for replacing hard near-field assumptions.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/core/localization_dispatch.py — geometry-aware routing and the place to stop selecting short-range-tuned algorithms blindly for tight arrays.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/core/localization.py — GCC-PHAT least-squares fallback behavior and its far-field limitations.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/config.py — shared config semantics, runtime profile defaults, and Python and Rust policy alignment.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/models.py — LocalizationResult, DetectionEvent, and TrackState uncertainty plumbing.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/core/fusion_node.py — LocalizationBranch and the point where localization outputs enter the staged pipeline.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/core/assembly.py — direct detection covariance emission.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/core/track_filters.py — fixed measurement covariance that must become measurement-aware for far-field uncertainty.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/core/tracking.py — track creation and covariance propagation.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/core/track_associators.py — fixed Euclidean association gate that should become uncertainty-aware.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/api/rust_dsp_manifests.py — Rust localization manifest ingestion into Python.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr/api/stream_consumer.py — localized render delivery path from the sidecar.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr-ingest-sidecar/src/srp_phat.rs — bounded SRP implementation that currently cannot represent long-range sources safely.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr-ingest-sidecar/src/dsp_worker.rs — cadence, coverage, and confidence gating that must remain real-time safe.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr-ingest-sidecar/src/actors/dsp_compute.rs — SRP execution and fallback reason emission.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr-ingest-sidecar/src/manifests.rs — sidecar localization payload schema.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr-ingest-sidecar/src/dsp_render_output.rs — localized vs omni render selection logic.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr-ingest-sidecar/src/main.rs — sidecar config defaults and env surface.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/minimappr-ingest-sidecar/src/ingest_backend.rs — backpressure and admission control invariants that must not regress while compute changes land.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/tests/test_localization.py — existing Python localization regression surface.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/tests/test_localization_benchmark.py — benchmark extension point for near-field and far-field validation.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/tests/test_runtime_profiles.py — runtime profile behavior and default selection checks.
- /Users/colincatlin/Documents-NoCloud/MinimapPR/tests/test_rust_manifest_handoff_e2e.py — hybrid manifest handoff and sidecar integration regression surface.

**Verification**
1. Python unit and benchmark tests pass for near-field and far-field Sirith cases with covariance-calibrated assertions.
2. Rust sidecar tests pass for near-field and far-field synthetic sources without raising queue saturation, compute-budget, or cadence regressions.
3. End-to-end hybrid manifest handoff tests verify localized renders still arrive on schedule and far-field results do not collapse into omni solely due to bounded search.
4. Load and soak validation shows stable queue depth, unchanged or better HTTP rejection semantics under overload, no added blocking I/O, and no extra live-path disk persistence.
5. Manual review of map and track outputs confirms covariance elongates along range at distance while close-in tracks remain tight and usable.
6. Python and Rust parity fixtures agree on steering direction, TDOA sign conventions, provenance fields, and covariance shape for the same synthetic cases.

**Decisions**
- Cartesian position remains the core output.
- Long-range uncertainty should be represented explicitly as anisotropic covariance, tighter on bearing and broader on range.
- Both Python and Rust are first-class targets; algorithmic behavior and config semantics should stay aligned.
- The initial fix should not rely on huge Cartesian SRP volumes, GPU work, or unbounded compute.
- Single-node tight-array long-range mode may report a projected Cartesian point with large radial covariance; consumers must treat covariance as part of the result, not metadata.
- Included scope: localization math, uncertainty plumbing, gating and association retuning, config harmonization, tests, docs, and rollout guardrails.
- Excluded scope: GPU acceleration, a brand-new bearing-only track type, frontend redesign beyond consuming existing covariance data, and unrelated BirdNET taxonomy tuning.

**Further Considerations**
1. If single-node far-field radial uncertainty remains too large after the solver changes, plan a follow-on multi-node fusion phase rather than forcing more aggressive single-node range claims.
2. If load testing shows the sidecar compute budget is too tight for full far-field refinement on every frame, keep the low-cost direction estimate on the localization heartbeat and run heavier refinement only on classifier-render frames or offline replay.
