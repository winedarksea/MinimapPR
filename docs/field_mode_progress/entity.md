# W3-entity: bird-entity model, grace window, API/WS, Heard-nearby tray (S6)

Branch `fm/entity` (base `field-mode` @ 8eca3b4, "Merge fm/core-sync into field-mode").
Everything is behind `entity_reporting_enabled` (default **False**). Flag off: no
coordinator is built, `ReportingFusionPolicy` and per-detection rules run exactly as
before, `GET /api/v1/entities` answers `{"enabled": false, "entities": []}`, the
frontend hides the tray and keeps its old layers.

## What was built

| Piece | File |
| --- | --- |
| Entity record, states, geometry (horizontal sigma, compass bearing, fan from tangential spread, ellipse) | `core/bird_entity_model.py` |
| State machine + sliding absorption window (`find_absorbing_entity`, `observe`, `absorb_unlocalized`, `maintain`) | `core/bird_entity_registry.py` |
| Entity-keyed reporting decision (replaces bucketed `ReportingFusionPolicy` for configured categories) | `core/entity_reporting_policy.py` |
| Notification grace window (asyncio timers, off the hot path) | `core/notification_grace_window.py` |
| FusionNode glue: decide / on_detection_surfaced / maintenance loop / WS events / absorbed-bout partial binding | `core/entity_reporting_coordinator.py` |
| Spatial consistency scores for absorption | `core/bird_entity_matching.py` |
| Records -> API/WS models (geo, ellipse) | `core/bird_entity_projection.py` |
| Sub-config + validation | `core/entity_reporting_config.py` |
| `GET /api/v1/entities`, PATCH validation helper | `api/entity_routes.py` |
| Pydantic `BirdEntity`, `BirdEntityBearing`, `BirdEntityUncertaintyEllipse`, `BirdEntityEvent`, `BirdEntityListResponse` | `models.py` |
| Scenario harness entity mode + 3 entity scenarios; metrics key on `entity_id` | `sim/tracker_detection_scenarios.py`, `sim/entity_scenario_catalog.py`, `sim/replay_entity_metrics.py` |
| Leptos: entity state + WS/poll, map layer (fans, points, ellipses), Heard-nearby tray, spatial-only map | `minimappr-frontend/src/state/entities.rs`, `src/layers/bird_entities.rs`, `src/panels/heard_nearby_tray.rs`, small edits to `ws.rs`, `api.rs`, `state.rs`, `layers/{mod,omni,tracks,detections}.rs`, `workspace/mod.rs`, `styles/workspace-map.css`, `maplibre-interop.js` (layer-retry fix); built dist in `minimappr/frontend/` |
| Browser demo driver | `scripts/dev_bird_entity_demo.py` |

FusionNode changes (≈60 lines, `# Field Mode W3-entity` markers): coordinator built in
`__init__` (None when off), started/stopped with the node; `_assemble_reporting_branch`
calls `entity_reporting.decide` for `applies_to(label_category)` detections; the enrich
path feeds absorbed bouts to the tracker; `_process_rules_and_delivery` hands surfaced
detections to `on_detection_surfaced` instead of evaluating rules; rule evaluation was
extracted into `_evaluate_and_dispatch_rules` (the grace window's callback).
`DetectionProduct.entity_id` and `ReportingDecision.entity_id` carry the entity.

## How it works

**States.** `heard_nearby` (no bearing/position) -> `direction_known` (bearing fan from
the reporting node) -> `pinpointed` (position + 1-sigma major axis
<= `entity_pinpointed_max_horizontal_uncertainty_m`, 15 m). A fix with a larger
uncertainty is direction-known with a range band (`range ± k·σ_radial`, k =
`entity_uncertainty_sigma_multiplier` = 2) and a fan whose half width is
`atan(k·σ_tangential / range)`, clamped to [5°, 45°]; a bearing-only solve (range
unobservable) gives a fan with no range band. Observations never lower the state.
Maintenance (every `entity_maintenance_interval_seconds`) steps back **one** state when
the bound track disappears/drops, when the track's uncertainty passes the pinpoint limit,
or after `entity_spatial_state_hold_seconds` (120 s) without a spatial update; an entity
not heard for `entity_expiry_seconds` (300 s) expires. Stepping to heard-nearby drops the
geometry (the map is spatial-only).

**Which entity (sliding window).** A detection whose track is already bound to an entity
goes to that entity (unless the track has no log-odds belief and the label contradicts
the entity — the legacy CV tracker merges nearby birds of different species). Otherwise:
same species (entity top label or anywhere in its `species_belief`), last heard within
`entity_absorption_window_seconds` (60 s, measured from the entity, so it slides with
each bout), and spatially consistent: fix within 12 m of a pinpointed bird, inside a
direction-known fan (+20°), bearing within 20° for bearing-only solves, or reporting node
within 50 m of the bird (the stand-in for the per-node SNR-profile fit until per-node
levels reach assembly). Unpositioned detections prefer entities already on the map.

**Reporting decision (pre-assembly).** No entity -> `insert` with report window
`[t, t + window)` (never collides on a bucket edge). Unpositioned bout matching an entity
-> `enrich_existing` on the entity's canonical row: no new row, no WS detection, no rules;
the entity pulses ("absorbed" event) and the bout still goes to `resolve_detection_track`
(W2 partial binding updates the track's species belief). Positioned bout matching an
entity -> `upgrade_existing`: assembly runs (tracker gets the fix), the canonical row is
updated in place. Sub-display (two-threshold) detections are never absorbed; they take the
plain insert path and the existing hidden-track rule.

**Rules (post-assembly, after all existing surfacing filters).** Zone-suppressed,
sub-display-without-visible-track and unknown/0.0 detections are dropped before the
entity sees them, so hidden tentative tracks never create entities. Then: a new
heard-nearby entity holds its rules for `entity_notification_grace_seconds` (4 s; 0 =
fire at once); if it gains a bearing/fix inside the window, rules fire once on the
localizing detection; otherwise they fire on the held detection when the window ends;
an entity that expires first never fires. A spatial entity fires at once. After that,
no further rules for that entity (one notification per bird). Categories outside
`entity_reporting_label_categories` (default `wildlife`) keep the legacy path, so drone /
security rules are unchanged.

**API / WS.** `GET /api/v1/entities` -> `BirdEntityListResponse {enabled, generated_ns,
entities[]}`. `/ws/live` carries `{"type": "entity", "event":
"created|updated|absorbed|state_changed|expired", "previous_state", "entity", "server_time_ns"}`.
Each entity: `state`, `species_top_label/probability`, `species_belief`,
`reporting_node_id(s)`, `position_m/position_geo` (spatial states only),
`horizontal_uncertainty_m`, `uncertainty_ellipse` (pinpointed), `bearing` (fan: origin
node + geo, bearing, half width, optional range band), `contour_geo` (reserved for the
W2-onset-srp likelihood contour; always null today), `track_id`, `canonical_detection_id`,
`first/last_heard_ns`, `last_pulse_ns`, counts.

**Frontend.** Tray ("Heard nearby", top-centre under the status ribbon, collapsible):
heard-nearby entities with species, probability, "near <node>", bout count, first/last
heard; header shows "N on map". Map layer "Bird entities": direction-known = amber fan
(annular sector when the range band is known) from the node; pinpointed = green point +
2-sigma ellipse; absorbed bouts pulse the point for 1.5 s. In entity mode the omni halos
and omni ripples are suppressed and tracks bound to an entity are drawn only by the entity
layer (one bird, one thing on screen).

## Config keys (restart-required; one `# --- Field Mode W3-entity ---` block per file)

`entity_reporting_enabled` (False), `entity_reporting_label_categories` ("wildlife"),
`entity_absorption_window_seconds` (60), `entity_absorption_max_fix_distance_m` (12),
`entity_absorption_max_node_distance_m` (50), `entity_absorption_max_bearing_difference_deg` (20),
`entity_pinpointed_max_horizontal_uncertainty_m` (15), `entity_spatial_state_hold_seconds` (120),
`entity_expiry_seconds` (300), `entity_bearing_default_half_width_deg` (15),
`entity_bearing_min_half_width_deg` (5), `entity_bearing_max_half_width_deg` (45),
`entity_uncertainty_sigma_multiplier` (2), `entity_notification_grace_seconds` (4),
`entity_maintenance_interval_seconds` (1), `entity_max_tracked_entities` (512).
Registered in `config.py` (fields, env `MINIMAPPR_<KEY>`, validation,
`entity_reporting_config()`), `settings_store.py`, `main.py` (GET config, PATCH merged-view
validation, restart-required for python and rust ingest), `core/config_groups.py` (new
"Bird Entities" group), `core/pipeline_graph.py` (Tracking node params).

## Measurements

### Entity metrics on the tracker scenarios

`python scripts/measure_tracker_scenarios.py --seeds 0 1 2 3 4` (< 3 s). The harness
(`sim/tracker_detection_scenarios.py`) runs the registry exactly as FusionNode does
(pre-assembly absorption of unpositioned bouts, observe of every surfaced bout) and tags
each detection with `entity_id`; `replay_entity_metrics` keys entities on `entity_id`
when present (else track / omni node+label as before). Scored at 5 m, 1 s slop; means
over 5 seeds. Three new scenarios (`sim/entity_scenario_catalog.py`):
`omni_to_localized_dedup` (test-strategy row: 5 bouts, 2 fail to localize, from two
nodes), `heard_nearby_then_localized`, `two_species_one_omni_each`.
Rows: tracker-only `off` / `all-on` (W2 flags) vs the same with `entity_reporting_enabled`.

| Scenario | Config | Surfaced dets | Visible tracks | Entities | L&C recall | Shown-label acc. | Swap rate | Duplicates | Multi-individual entities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| intermittent_perched_robin | off | 3.00 | 1.00 | 0.00 | 0.38 | 1.00 | n/a | 0.00 | 0.00 |
| intermittent_perched_robin | all-on | 8.00 | 1.00 | 0.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| intermittent_perched_robin | off+entity | 3.00 | 1.00 | 1.00 | 0.38 | 1.00 | n/a | 0.00 | 0.00 |
| intermittent_perched_robin | all-on+entity | 8.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| robin_hops_15m | off | 8.00 | 1.00 | 0.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| robin_hops_15m | all-on | 8.00 | 1.00 | 0.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| robin_hops_15m | off+entity | 8.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| robin_hops_15m | all-on+entity | 8.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| two_species_alternating | off | 10.00 | 4.00 | 0.00 | 0.65 | 1.00 | n/a | 4.20 | 2.20 |
| two_species_alternating | all-on | 12.00 | 2.00 | 0.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| two_species_alternating | off+entity | 10.00 | 4.00 | 2.00 | 0.83 | 1.00 | n/a | 0.00 | 0.00 |
| two_species_alternating | all-on+entity | 12.00 | 2.00 | 2.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| two_species_overlapping | off | 9.00 | 2.40 | 0.00 | 0.63 | 1.00 | 0.00 | 1.40 | 1.00 |
| two_species_overlapping | all-on | 12.00 | 2.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| two_species_overlapping | off+entity | 9.00 | 2.40 | 2.00 | 0.75 | 1.00 | 0.00 | 0.00 | 0.00 |
| two_species_overlapping | all-on+entity | 12.00 | 2.00 | 2.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| weak_fix_weak_id | off | 2.00 | 0.00 | 0.00 | 0.00 | 1.00 | n/a | 0.00 | 0.00 |
| weak_fix_weak_id | all-on | 4.00 | 1.00 | 0.00 | 0.80 | 1.00 | n/a | 0.00 | 0.00 |
| weak_fix_weak_id | off+entity | 2.00 | 0.00 | 1.00 | 0.00 | 1.00 | n/a | 0.00 | 0.00 |
| weak_fix_weak_id | all-on+entity | 4.00 | 1.00 | 1.00 | 0.80 | 1.00 | n/a | 0.00 | 0.00 |
| flock_same_species | off | 9.00 | 5.40 | 0.00 | 0.75 | 1.00 | n/a | 6.00 | 3.60 |
| flock_same_species | all-on | 11.00 | 3.00 | 0.00 | 0.92 | 1.00 | n/a | 1.00 | 1.00 |
| flock_same_species | off+entity | 9.00 | 5.40 | 2.40 | 0.75 | 1.00 | n/a | 2.40 | 2.00 |
| flock_same_species | all-on+entity | 11.00 | 3.00 | 3.00 | 0.92 | 1.00 | n/a | 1.00 | 1.00 |
| omni_to_localized_dedup | off | 5.00 | 1.00 | 0.00 | 0.60 | 1.00 | n/a | 2.00 | 0.00 |
| omni_to_localized_dedup | all-on | 5.00 | 1.00 | 0.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| omni_to_localized_dedup | off+entity | 5.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| omni_to_localized_dedup | all-on+entity | 5.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| heard_nearby_then_localized | off | 3.00 | 1.00 | 0.00 | 0.67 | 1.00 | n/a | 1.00 | 0.00 |
| heard_nearby_then_localized | all-on | 3.00 | 1.00 | 0.00 | 0.67 | 1.00 | n/a | 1.00 | 0.00 |
| heard_nearby_then_localized | off+entity | 3.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| heard_nearby_then_localized | all-on+entity | 3.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| two_species_one_omni_each | off | 6.00 | 2.00 | 0.00 | 0.33 | 1.00 | n/a | 4.00 | 2.00 |
| two_species_one_omni_each | all-on | 6.00 | 2.00 | 0.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| two_species_one_omni_each | off+entity | 6.00 | 2.00 | 2.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| two_species_one_omni_each | all-on+entity | 6.00 | 2.00 | 2.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |

Readings:

- **Duplicates never rise with entities** (gated in `tests/test_entity_scenario_metrics.py`
  for both tracker configs on all 9 scenarios, plus multi-individual entities and swap rate).
- With the W2 tracker all-on, entities match tracker-only on the six tracker scenarios
  and remove the omni/localized duplicate the tracker cannot: `heard_nearby_then_localized`
  1.0 -> 0 duplicates, recall 0.67 -> 1.00 (the heard-nearby entity is promoted, not
  duplicated by the new track).
- On the legacy tracker (`off`), entities fix most of its splits: two-species duplicates
  4.2 -> 0 and 1.4 -> 0, flock 6.0 -> 2.4, omni dedup 2.0 -> 0, recall up everywhere.
  This needed the "track binding only if the species agrees on a belief-less track" rule:
  without it the CV tracker's merged robin/tanager track dragged both birds into one
  entity (1.4 multi-individual entities on `two_species_one_omni_each`).
- Flock all-on keeps the tracker's single ghost-revert mistake (1 duplicate, 1
  multi-individual); the entity layer neither fixes nor worsens it.

### Replay bundles

Not re-run: `server_baseline.md` shows every bundle fix collapsing onto a node until
W2-onset-srp lands, and its stub classifier labels are not `wildlife`, so entity mode
would not apply. Re-measure with `entity_reporting_enabled` once onset-SRP is merged.

## Tests

- `tests/test_bird_entity_registry.py` (12): heard -> direction -> pinpointed as one
  entity, no demotion on a failed bout, absorption within window, sliding window vs
  buckets, species/node/fix refusals, track binding + heard-nearby promotion, large
  uncertainty -> fan with range band, track expiry steps back one state then hold then
  expiry, uncertainty growth demotes, species belief from track log-odds, eviction, geometry.
- `tests/test_entity_reporting_pipeline.py` (7): flag off = legacy policy + rules per
  detection; unlocalized-then-localized inside grace fires once (on the localized
  detection) and never again; never-localized fires after grace; zero grace / expiry
  cancels; hidden (sub-display) and zone-suppressed detections never create entities;
  policy insert / enrich (absorb) / upgrade / sub-display decisions against real storage;
  maintenance WS events and expiry.
- `tests/test_entity_api.py` (4): endpoint disabled by default, lists heard-nearby
  (no position) and pinpointed (geo, track), entity event on `/ws/live`, PATCH
  validation + restart-required.
- `tests/test_entity_scenario_metrics.py` (22): the gates above.
- `minimappr-frontend/src/layers/bird_entities.rs` has `#[cfg(test)]` geometry tests (fan
  and ellipse rings).

## Browser check

`python scripts/dev_bird_entity_demo.py --port 8765` (real app, python ingest, entity
flag on, grace 2 s) registers node-a/b/c and every 6 s drives: a localized robin with a
real tracker track (-> pinpointed), a bearing-only tanager from node-b (range
`range_asymptotic` -> direction known), a house wren omni from node-c (heard nearby),
a one-off chickadee omni from node-a, and a robin omni bout from node-b through
`coordinator.decide` (absorbed). Driven with headless Chromium (Playwright 1.61) at
1440x900 after `trunk build`:

- Tray (top-centre): "Heard nearby 2 · 2 on map" with House Wren 82 % "near node-c ·
  N× · last 0s ago · first …" and Black-Capped Chickadee "near node-a"; after 5 min
  without a bout the chickadee expired and the tray showed 1. Collapse toggle works.
- Map: green robin point with its 2-sigma ellipse (2.9 x 2.4 m) at (6, 12) m; amber
  tanager fan from node-b, bearing 45°, half width 5° (clamped min), default 150 m
  length (range unobservable); no pin at node-c / node-a for the heard-nearby birds; no
  separate robin track marker or legacy grey bearing wedge for the tanager.
- `/api/v1/entities`: robin `absorbed_detection_count` climbing each cycle (absorption
  working, no new rows/cards); no page errors besides blocked OSM tiles (offline).

Fixes found by the browser check: (1) serde_wasm_bindgen turns `serde_json` objects into
JS `Map`s that MapLibre cannot read — the entity layer now passes JSON text (the
existing acoustic layer uses the same `to_value` pattern and is probably affected
too; left alone); (2) `maplibre-interop.js` `upsertGeojsonLayer` waited on the one-shot
`load` event when `isStyleLoaded()` was false, so a second generic layer added right
after a first was silently dropped — it now retries on a short timer and keeps
`setLayerData` calls that arrive before the source exists; (3) the legacy detection
layer drew a second wedge for the tanager — in entity mode it now skips species the
entity layer draws.

Pre-existing, not changed: in headless Chromium the COP map never auto-initialised
(`#mmp-map` stayed empty until the driver called `mapInterop.init`; then toggling the
theme re-ran the layer effects); and live `/ws/live` detection payloads have no
top-level `label`, so the Detections panel shows "—" rows until the 3 s poll refreshes
them.

## Open issues / follow-ups

- **Memory-only, per process.** Entities live in the FusionNode process; under
  `process_role=api` the endpoint answers `enabled: false` (WS events still arrive through
  the live hub). A restart forgets entities (like `species_belief`).
- **SNR-profile fit is a node-distance bound** (50 m) until per-node in-band levels reach
  assembly (W3-attribution / onset-SRP); same gap as the tracker's node-ranking constraint.
- **`contour_geo` is always null**: no localizer emits a likelihood contour yet; the map
  draws the covariance ellipse (pinpointed) or the covariance-derived fan (direction known).
- **Rules fire once per entity.** Rules that expect every bout (e.g. counting) see one
  event per bird for entity categories; other categories unchanged.
- **Grace window is not gated on node count** (plan: solo / couples alert at once).
  Single-node sites should set `entity_notification_grace_seconds=0`.
- **Absorbed bouts update the canonical row's branch evidence** but not its position;
  positioned bouts overwrite the canonical row in place (DB history is one row per
  entity, plus `track_updates`).
- **Pre-render report-window gate** (`fusion_report_window_localized_emission_cap`) still
  uses the legacy bucket bounds; harmless (it only skips classifier work) but not
  entity-aware.
- The tray is top-centre (both docks can cover the edges); the legend has no entity
  entries yet.
