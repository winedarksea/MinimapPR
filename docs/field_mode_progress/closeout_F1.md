# Closeout F1: two follow-ups from the closeout review

Two small fixes the user approved after `review_closeout.md`. One commit each on `field-mode-closeout`.

## F1-C5: restored entity covariance uses the publish-time scale (review M1)
Commit `1bfa709`.

- **Problem.** `host_entity_restoration.rs` rebuilt every restored entity's covariance from its contour by
  dividing by the *inside* calibration scale² (0.695²). An entity published outside the layout was drawn at
  the outside scale (0.315²), so it came back at about 0.2× its raw variance.
- **Fix.**
  - The mirror restore (`host_restoration.rs`) now also keeps each island's latest `LayoutModel`. It is used
    only to choose the scale; it is not restored as live layout state.
  - The successor classifies the entity's position inside or outside that layout with the publisher's own
    code: `RegionContext::from_layout_positions` (the centroid and circumradius, now shared with
    `IslandSyncEngine::island_region_context`) and `published_sigma_scale` (split out of
    `published_covariance`). It then divides by that scale².
  - With no mirrored layout, or one with no node positions, it divides by `min(inside, outside)²`. This errs
    wide: a restore that is too tight can fail to associate the bird's next fix, which recreates the
    duplicate entity.
- **Limit.** The layout used is the island's *latest* mirrored one, not necessarily the one in force when that
  entity update was published. An entity close to the circumradius can classify differently if the layout
  moved in between.
- **Tests** (`minimappr-core`, `mesh::host_entity_restoration::tests`). The test helper now publishes through
  the real `published_contour` against a 20 m ring layout.
  - `outside_ring_entity_restores_its_raw_covariance` (new): an entity at (90, 35) m, outside the ring. The
    restored horizontal covariance matches the raw covariance before handover to within 1e-3.
  - `without_a_mirrored_layout_the_restore_errs_wide` (new): no layout, entity inside or outside. The
    restored variance is at least the raw variance.
  - `seeded_entity_keeps_its_wire_id_position_and_covariance` (inside the ring) and
    `times_move_onto_the_new_reference_clock_at_the_first_call` still pass.
  - The mesh-sim successor scenario `successor_restores_entities` (`pr_seeds`) still passes.
- **Sabotage.** With the divisor forced back to the inside scale, both new tests failed and the two existing
  tests passed. Restored afterwards.

## F1-omni: a never-streamed cluster member no longer empties the render (closeout_B1 frame-0 caveat, review N3)
- **Problem.**
  - A cluster member that has not streamed yields empty classification windows.
  - Both render paths truncate to the shortest channel: `omni_mix` and `render_band_split_f32`. So the whole
    render came out empty and was skipped ("empty PCM render").
  - The live default is `classification_audio_source = "beamformed"`. Once SRP clears the confidence floor,
    it takes the **band-split** path, not the omni one. So the bug was not only in the omni path. In the new
    test, before the fix, both paths published 0 renders for 3 localization results.
  - Live, this is the Bluefin cluster: sirith-tetra-07b5 streams and sirith-tetra-1a15 never has.
- **Fix** (`minimappr-ingest-sidecar/src/dsp_render_output.rs`).
  - `compute_render_bytes` first calls `channels_with_data`. It drops empty channels together with their
    mic positions, index for index, so band-split steering stays correct, and then renders on what is left.
    This covers the nearest-node omni, beamformed band-split and omni fallback paths.
  - Partial channels stay in and still truncate. The truncation to the parked geometry in
    `finalize_pending_payload` is unchanged.
  - If no channel has data, the inputs pass through unchanged and the render stays empty, as before.
  - `source_channel_count` in the render metadata now counts the channels actually rendered. It is
    diagnostic only; nothing in Python branches on it.
- **Tests.**
  - New module `dsp_worker_never_streamed_member_tests.rs`. Node a streams 60 frames; node b is declared in
    the cluster but never streams. Each test asserts that every real `localization_result` carries a
    `derived_handle`:
    - `never_streamed_member_does_not_empty_the_beamformed_render`
    - `never_streamed_member_does_not_empty_the_omni_render`
  - The test module reuses `RaceHarness` from the coverage-race tests. Its constructor, clock, deliver
    helper, `race_config` and a few constants are now `pub(super)`.
  - `dsp_render_output::tests::empty_channels_drop_out_with_their_mic_positions` (unit test, new).
  - `render_on_a_parked_trigger_frame_publishes_one_result_carrying_the_render_handle` no longer skips frame 0
    (the B1 caveat). It now asserts that frame 0's render handle is present too.
- **Sabotage.** With `channels_with_data` forced to pass everything through, all three worker tests failed
  (both new ones and the tightened B1 test). Restored afterwards.
- **Not done.** No live check on Bluefin: the sidecar there was not rebuilt or redeployed.

## Suites
| Suite | Baseline at `25f8127` | After both commits |
| --- | --- | --- |
| `cargo test --workspace` | 581 passed, 0 failed, 8 ignored | 586 passed, 0 failed, 8 ignored (+5 new tests) |
| `pytest tests/ --ignore=tests/test_soak_harness.py` | 2084 passed, 9 skipped | 2084 passed, 9 skipped (no Python changes) |
