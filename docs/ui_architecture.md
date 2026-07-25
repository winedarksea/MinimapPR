# MinimapPR UI Architecture

This document is the implementation contract for the COP/C2 frontend overhaul. It keeps the current Leptos CSR app shippable while moving the operator experience toward a full-bleed map workspace with audio, detections, uncertainty, alerts, and effectors as first-class operational surfaces.

## Design Tokens

The token system is MD3 at the foundation and C2-specific at the semantic layer. UI chrome uses neutral surface, outline, shape, density, and elevation tokens; saturated color is reserved for operational meaning.

Existing token groups:

- MD3 color roles: `--md-sys-color-primary`, `secondary`, `tertiary`, `error`, surface tiers, `on-surface`, outline, scrim.
- C2 status roles: `--mmp-sys-color-ok`, `warn`, `danger`, `info`, `armed` with container and on-color variants.
- Map entity roles: `--mmp-sys-color-map-node`, `effector`, `track`, `track-coasting`, `track-dropped`, `detection`, `bearing`, `omni`.
- Typography: `--md-sys-typescale-plain`, `mono`, and the MD3 display/headline/title/body/label scale.
- Shape, elevation, state-layer, density, and legacy alias tokens.

New operational token groups added for the overhaul:

- Uncertainty: `--mmp-uncertainty-bearing`, `ellipse`, `halo`, `range`, `selected`, `low-confidence`.
- Severity: `--mmp-severity-critical`, `high`, `medium`, `low`, `info` plus `--mmp-severity-*-container`.
- Freshness: `--mmp-freshness-live-opacity`, `recent-opacity`, `stale-opacity`, `expired-opacity`.
- Workspace dimensions: `--mmp-workspace-dock-width-*`, `--mmp-workspace-bottom-dock-height`, `--mmp-workspace-inspector-width`, `--mmp-workspace-toolbar-size`.
- Motion: `--mmp-motion-duration-fast`, `standard`, `slow`, `--mmp-motion-easing-standard`, `emphasized`.
- Z-index ladder: `--mmp-z-map`, `layers`, `docks`, `toolbar`, `inspector`, `context-menu`, `modal`.

Color budget rules:

- Critical alerts are the only persistent looping animation.
- Warning/critical colors never style baseline navigation or decorative chrome.
- Operational symbols use redundant shape plus color: node square, track diamond, detection ring, bearing wedge, effector wedge/arc, omni halo.
- Coordinates, times, counters, confidence, bearings, and IDs use tabular mono figures.

## Information Architecture

`/cop` is the live workspace. The map is the persistent canvas; side information is docked, collapsible, and spatially stable. Analysis, audio recording/library, and settings stay as routed full pages because they are slower, task-specific workflows.

Workspace surfaces:

- Map canvas: live nodes, tracks, detections, uncertainty, zones, effectors, and overlays.
- Left dock: node health and device registration/status.
- Right dock: tracks, detections, alerts, and modality drawers.
- Bottom dock: audio pipeline/review and timeline-adjacent live summaries when present.
- Inspector: pinned entity detail for node, track, detection, alert, effector, or zone.
- Toolbar: layers, draw/edit spatial tools, follow-live, density/layout controls.

## Drawer Registry

The workspace uses a single drawer registry:

`DrawerId` covers `Nodes`, `Tracks`, `Detections`, `Alerts`, `Audio`, `Rf`, `Seismic`, and `Speech`. Each drawer definition owns title, icon, default dock, badge count, availability signal, and render function.

Availability gates future modality UI. With no registered RF, seismic, radar, or speech device, those drawers are absent. When devices are registered locally or later arrive from backend registries, the same signals make drawers visible.

Layout is persisted as `mmp.workspace.v1` with open/collapsed state, dock sizes, and drawer dock assignment. Missing or corrupt preferences fall back to a COP-compatible default: nodes left; tracks, detections, alerts right; audio hidden.

## Map Layer Registry

Layers are mounted once and controlled by persisted visibility in `mmp.layers.v1`. A layer definition has id, title, group, default visibility, availability, and a mount function.

As built, the main COP map lifecycle lives in `src/map/mod.rs`, while state-to-map effects are split under `src/layers/` for nodes, tracks, detections, uncertainty, effectors, omni halos, zones, overlays, heatmap, acoustic maps, and future modalities. `src/layers/mod.rs` owns the `LayerDef` catalog used by the workspace layer controls and the mount function that installs effects once. The operator-facing visibility state persists through `mmp.layers.v1`.

Core layers:

- Nodes: health-coded node squares and audio status.
- Tracks: diamonds, TQI sizing, velocity vectors, coasting/dropped treatment.
- Detections: rings or clustered circles with confidence and age styling.
- Uncertainty: covariance ellipses, bearing wedges, node-only halos.
- Effectors: camera marker, FOV wedge, armed/pending state.
- Zones: polygon geometry and occupancy.
- Overlays: floorplans and GeoJSON/image overlays.
- Heat/acoustic: live analytic heatmap and acoustic map slot.
- Future modalities: radar, RF, seismic.

## JavaScript Interop Contract

The long-term map backend is `globalThis.mapInterop`. Phase 1 keeps Leaflet-equivalent function names during migration so Rust effects can move independently.

Core lifecycle:

- `init(lat, lon, zoom)`, `resize()`, `panTo(lat, lon)`, `flyTo(lat, lon, zoom?)`, `fitBoundsLatLons(points)`.
- `setTheme(theme)`, `setSelectionCallback(callback)`, `setContextMenuCallback(callback)`.
- `highlightCopItem(kind, id)`, `clearCopHighlight()`.

Entity functions preserve the current marker contract:

- Nodes, omni halos/ripples, tracks, velocity vectors, detections, bearing-only detections, effectors, zones, GDOP, uncertainty, and heatmap.

Generic layer functions:

- `ensureLayer(layerId, specJson)`, `setLayerData(layerId, dataJson)`, `setLayerVisible(layerId, visible)`, `removeLayer(layerId)`.
- `setImageOverlay(id, url, corners, opacity)`.
- `startZoneDraw(callback)`, `startZoneEdit(zoneId, latlngs, callback)`, `cancelZoneDraw()`.

## Extension Contracts

New modality:

1. Add a device schema entry with position/orientation requirements.
2. Add a feed signal in state.
3. Add a mock generator until backend ingest exists.
4. Add a map layer module.
5. Add a drawer or inspector surface.

New map layer:

1. Define a `LayerDef`.
2. Convert state to GeoJSON or image source data in Rust.
3. Use `mapInterop` for paint/source lifecycle.
4. Register it in the layer registry.

New device type:

1. Add `DeviceKind` and schema fields.
2. Add form validation.
3. Persist records to `mmp.devices.v1` until a backend registry exists.
4. Gate drawers/layers on device presence.

## Mock Data Seam

Mocks are explicit operator-facing placeholders. Every mock-fed panel or map artifact carries a persistent `MOCK - awaiting backend` badge. Feed structs live with state/schema types rather than inside mock modules so real backends can swap in without rewriting presentation components.

## Size Audit

Release build snapshot from `NO_COLOR=false trunk build --release` on 2026-07-07:

- WASM: `minimappr-frontend-7184b1b5e01b0aaa_bg.wasm` = 2.4 MB.
- JS glue: `minimappr-frontend-7184b1b5e01b0aaa.js` = 69 KB.
- Map interop: `maplibre-interop.js` = 32 KB.
- Audio interop: `audio-interop.js` = 9.4 KB.
- App CSS: `style-71f3a87d4b48e645.css` = 117 KB.
- Local fonts CSS: `fonts-9bff27209ed48361.css` = 552 B.
- Vendored MapLibre: `maplibre-gl.js` = 1.0 MB, `maplibre-gl.css` = 68 KB.
- Vendored fonts: Inter Latin WOFF2 = 47 KB, Material Symbols Rounded TTF = 1.1 MB.

## Audio-First Requirements

The UI must keep audio pipeline usability central:

- Per-node pipeline controls expose RMS, stage lag, queue/drops, gain/filter overrides, and dispatch provenance.
- Localization views always show method, capability tier, covariance/bearing uncertainty, freshness, and confidence.
- Detection review keeps audio playback, waveform/evidence, label correction, and training promotion in one flow.
- Recording status is live and visible globally while a capture session is active.

## Phase Changelog

- Phase 0: Created this architecture contract and added the missing operational token groups to the CSS root/light theme.
- Phase 1: Vendored MapLibre GL JS 5.24.0 locally, replaced CDN Leaflet/leaflet.heat with local MapLibre assets, retargeted Rust map bindings to `globalThis.mapInterop`, added `maplibre-interop.js` with the current COP/heatmap marker surface plus generic layer stubs, renamed the map container to `mmp-map`, removed `leaflet-interop.js`, and added Node tests for pure interop math/fallback behavior.
- Phase 2 slice: Routed `/cop` through `workspace::MapWorkspace`, added docked drawer chrome, persisted drawer open/collapsed state to `mmp.workspace.v1`, mounted the existing node/track/detection/alert panes over a full-map canvas, added a floating live status ribbon, and added a first pinned-selection inspector for tracks, detections, and alerts.
- Phase 2 slice: Added frontend zone models and polling for existing `/api/v1/zones` plus `/api/v1/zones/occupancy`, rendered zones on the COP map, made zone polygons selectable through `mapInterop`, added a Zones drawer with occupancy state, and expanded the inspector to cover nodes, effectors, and zones.
- Phase 3 slice: Added `mapInterop.setBearingWedge/removeBearingWedge/clearBearingWedges`, tested wedge geometry, and wired bearing-only detections to render node-anchored uncertainty fans with range and angular uncertainty derived from detection metadata when available.
- Phase 3 slice: Moved COP detections off DOM markers and onto a clustered MapLibre GeoJSON source with confidence-driven point radius/opacity and click-to-select behavior. Cluster count text remains deferred because the current offline raster-only MapLibre style intentionally has no glyph endpoint.
- Phase 1/3 hardening slice: Added live MapLibre basemap theme retinting, removed stale node markers when the site switches out of geodetic mode, and handled `rules_updated` websocket events so rules saves no longer produce parse warnings.
- Phase 2/5 hardening slice: Added persisted `mmp.layers.v1` visibility controls for nodes, tracks, detections, effectors, omni halos, zones, overlays, and future-modality artifacts. Map effects now tear down their rendered artifacts when a layer is disabled, providing the operator-facing behavior of the planned layer registry before the source files are fully split into `src/layers/*`.
- Phase 7 slice: Added reusable RMS meter and strip-chart components, then upgraded the Pipeline settings page with a visual mic -> stage chain -> output flow and clearer live RMS/stage health presentation while preserving the existing `/api/v1/pipeline/nodes` and PATCH contracts.
- Phase 4 slice: Added the minimal rules CRUD backend enabler: `GET/PUT /api/v1/rules`, rules Pydantic models, `RuleDef/RuleCondition.to_dict()`, default rules serialization, atomic active-rules-file writes, `rules_updated` websocket broadcast, and targeted API tests proving defaults, validation, round-trip persistence, and engine hot reload.
- Phase 4 slice: Added typed frontend rules API helpers and `/settings/rules`, with structured rule/action controls, frontend validation, raw JSON escape hatch, source/path status, and unit tests for raw parsing plus confidence validation.
- Phase 6 slice: Added effector arm/disarm and safety controls end-to-end: persisted `EffectorSafetyConfig`, manager interlocks for require-arm, per-effector slew interval, and no-go zones, HTTP endpoints, focused backend tests, and a Settings -> Effectors safety card UI.
- Phase 7 slice: Added live recording status broadcast over `/ws/live` for capture session updates, immediate REST start/stop broadcasts, frontend `recording_status` handling, and a persistent top-bar REC chip linked to `/audio/record`.
- Phase 4 slice: Added map overlay backend enabler: `MINIMAPPR_MAP_OVERLAY_DIR`, `map_overlays` storage table/helpers, `MapOverlay*` API models, multipart `GET/POST/PATCH/DELETE /api/v1/overlays` plus content serving with path-containment checks, mutation broadcasts, and focused overlay API tests.
- Phase 5 slice: Added typed frontend overlay helpers and `/settings/overlays` for multipart floorplan/SVG/GeoJSON upload, bounds/opacity editing, enable toggles, direct content links, delete controls, and bounds parser unit tests.
- Phase 5 slice: Added live COP overlay rendering: global overlay state polling, `overlay_updated` websocket refresh, MapLibre image/SVG georeferenced overlays, GeoJSON fill/line/point rendering, and cleanup when overlays are disabled or deleted.
- Phase 5 slice: Added a COP heatmap layer control backed by `/api/v1/analytics/heatmap`, with 5m/1h/24h/7d windows, live refresh on the app poll cadence, bin-count/error feedback, and MapLibre heatmap cleanup when disabled.
- Phase 6 slice: Reworked the COP alert drawer into a SOC-style triage queue with open/status filters, critical counters, severity-first cards, detection/track evidence pivots when alert references are present, payload disclosure, and Ack/Dismiss/Escalate actions using the existing alert status API.
- Phase 7 slice: Added detection review directly into the audio analysis workflow: review state chips, label/category correction with analytics label suggestions, notes, training/export promotion, Confirm/Correct/Reject actions via `PATCH /api/v1/detections/{id}/review`, and optimistic metadata refresh alongside playback, waveform, and spectrogram.
- Phase 8 slice: Added a local future-modality device registry persisted at `mmp.devices.v1`, Settings -> Devices registration/import/export for 24G radar, SDR/RF, seismic, and speech nodes, validation for radar position/orientation, and hidden-until-enabled RF/seismic/speech COP drawers stamped as mock backend-pending surfaces.
- Phase 8 slice: Added typed client-side modality feeds plus deterministic mock generators for registered RF/radar, seismic, and speech devices, then replaced generic placeholder drawer cards with RF spectrum/emitter, seismic strip-chart/meter, and speech transcript ticker panels that preserve the mock-to-real backend swap seam.
- Phase 8 slice: Added a combined `future-modalities` MapLibre GeoJSON layer fed by the local device registry and mock modality feeds, including registered device points, mock RF emitter points, and radar FOV wedges using the generic `mapInterop.ensureLayer/setLayerData` contract.
- Phase 8 slice: Added Home Assistant placeholder configuration end-to-end: `hass_*` settings/env keys, redacted config snapshots, PATCH allowlist and validation, focused backend tests, and a Settings -> Integrations page that persists HA URL/token/MQTT fields while marking live bridge behavior as backend pending.
- Phase 2 slice: Made the Home Assistant bridge real (see `docs/home_assistant_integration.md`). `pages/settings/integrations.rs` became a directory module (`mod.rs` shell, `hass_form.rs`, `hass_status.rs`); the "backend pending" badge and the mock "Planned Bridge Outputs" card are gone, replaced by a live `connection_state` badge, `last_connect_error` subline, backend-driven counters, and Republish/Purge discovery actions (purge behind a two-click confirm). Added `LiveEvent::HassStatus` plus `AppState.hass_status`, and a `#[serde(other)] Unknown` catch-all variant so a server event this build has never heard of no longer fails the whole enum and spams parse warnings.
- Phase 6 slice: Added the first COP map context menu: MapLibre right-click now opens a workspace menu with copy-coordinates, center-map, starter zone creation, and guarded camera slew actions. Zone creation opens a compact editor seeded from the clicked lat/lon, saves a valid geodetic square through `PUT /api/v1/zones/{id}`, updates the Zones drawer immediately, and pins the new zone. Slew-to-map converts the clicked lat/lon through the configured site origin into local ENU meters before calling the existing effector aim endpoint; disarmed cameras and missing site origins show operator-visible disabled reasons.
- Phase 5/6 hardening slice: Replaced the MapLibre zone draw/edit stubs with a click-to-vertex draw session and draggable vertex edit session. Enter accepts the polygon, Escape cancels, and temporary draft layers are cleaned up on completion/cancel.
- Phase 9 slice: Vendored local UI fonts (`Inter` Latin WOFF2 and static `Material Symbols Rounded`) under `vendor/fonts`, replaced Google Fonts CDN links with `fonts.css`, and kept the source/version notes beside the assets for offline deployment audits.
- Phase 9 slice: Ran the release frontend size audit and recorded optimized WASM, JS, CSS, vendored MapLibre, and local font artifact sizes in this document.
- Phase 2 hardening slice: Added persisted left/right dock widths to `mmp.workspace.v1`, clamp bounds, mouse-drag resize handles, and inspector/layer-control positioning that follows the resized docks.
- Phase 2 hardening slice: Split the former monolithic `style.css` into ordered modules under `styles/`: tokens/base/shell, workspace map+docks+inspector, shared components/map, and page-specific common/alerts/forms/analysis/diagnostics/audio/recording/pipeline/settings/compact row styles. Every CSS module is now below 500 lines while preserving the original cascade order.
- Phase 2 hardening slice: Hid the global shell status strip on `/cop` so the full-bleed workspace uses its floating status ribbon without duplicating vertical status chrome; routed pages still keep the persistent strip.
- Phase 2/3/5/8 hardening slice: Extracted COP state-to-map effects from `src/map/mod.rs` into `src/layers/{nodes,tracks,detections,uncertainty,effectors,omni,zones,overlays,heatmap,future_modalities}.rs`, leaving the map component focused on MapLibre lifecycle, callbacks, first-load centering, and legend markup.
- Phase 2 hardening slice: Added a real `LayerDef` catalog in `src/layers/mod.rs` with id/title/group/default visibility and visibility accessors, then rendered workspace layer toggles from that registry instead of duplicating hardcoded toggle logic in `MapLayerControls`.
- Phase 5 hardening slice: Added the acoustic map slot end-to-end on the frontend: typed `AcousticMapLayer`/sample state, mock live SPL-density generation from positioned audio nodes, an `src/layers/acoustic.rs` GeoJSON renderer using `mapInterop.ensureLayer/setLayerData`, and a persisted layer toggle.
- State hardening slice: Split the former frontend state model cluster into `src/state.rs`, `src/state/models.rs`, and `src/state/feeds.rs`, preserving the existing `crate::state::*` public paths while keeping app signals, backend DTOs, and future-modality feed structs in separate focused modules under the 500-line target.
- Verification hardening slice: Added a Playwright COP smoke test (`npm run test:smoke`) that loads `/cop` through Trunk, verifies the full-bleed workspace, MapLibre canvas, docks, ribbon, registry-backed layer toggles including Acoustic, and absence of the old global status strip. The smoke uses SwiftShader for stable headless MapLibre WebGL.
- Dev-server hardening slice: Changed the Trunk `/ws` proxy backend from `ws://127.0.0.1:8000/ws` to `http://127.0.0.1:8000/ws`, allowing Trunk to proxy websocket upgrades instead of rejecting the backend URL scheme before it reaches the server.
- Verification hardening slice: Ran the COP smoke against a real API-only backend on `127.0.0.1:8000` with temporary storage plus Trunk on `127.0.0.1:18080`; the UI rendered and the backend accepted `/ws/live` websocket connections through the Trunk proxy.
