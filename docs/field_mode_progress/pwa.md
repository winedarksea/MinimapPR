# W4-pwa: notes

Branch: `fm/pwa`, off `field-mode` at `292e2dd` (Merge fm/attribution).

## What landed

| Path | Contents |
|---|---|
| `minimappr-frontend/Trunk.field.toml`, `field.html` | A separate trunk target for the Field PWA. It builds the same crate with `--features field-pwa` into `dist-field/` (gitignored). The COP build (`Trunk.toml` → `../minimappr/frontend`) is untouched. |
| `minimappr-frontend/src/main.rs` | Under `field-pwa`, mounts `field::FieldApp` instead of the COP `App` and allows dead code in that build. The default build has the same module graph as before. |
| `minimappr-frontend/src/field/` | The Field PWA itself:<br>• `capture_controller`: mic → Stage-1 → BirdNET → feed/relay orchestration<br>• `stage1_capture_pipeline`: core Stage-1, rolling buffer, classification windows, FLAC/MMCL clips<br>• `birdnet_input_resampler`: Kaiser-windowed sinc, capture rate → 32 kHz<br>• `relay_session_controller`: host, join by code (SPAKE2), sealed send<br>• `field_message_builders`: Heartbeat and DetectionEvent `FieldMessage`s<br>• `relay_message_presenter`<br>• `browser_bindings`: wasm-bindgen imports of the JS interop<br>• `field_signals`, `field_panels`, `field_configuration` (all constants) |
| `minimappr-frontend/src/mock/field_shim.rs` | The seam: a local shim standing in for `api.rs`/`ws.rs`. It writes local (or relay-received) detections into the shared `AppState.detections` feed and maps relay phase onto `AppState.ws_status`, so shared components render field data unchanged. |
| `minimappr-frontend/field/` | • `field-mic-worklet.js` and `field-capture-interop.js`: AudioWorklet capture, ported (see Attribution)<br>• `field-birdnet-interop.js`: ORT Web wrapper with Cache-API model storage<br>• `field-sw.js`: offline service worker<br>• `field.webmanifest`, `field.css`<br>• `field-dev-server.mjs`: local static server, model mount and same-origin `/v1` relay proxy |
| `minimappr-transport/src/web_websocket/` (feature `web-websocket`, wasm32 only) | The browser adapter for the sans-IO `RelayClientSession`:<br>• `WebRelayDriver`: browser WebSocket, `bufferedAmount` gate, reconnect with the session's backoff, stale-socket generation guard<br>• `RelayFetchClient`: relay HTTP API over `fetch`<br>• `BrowserCryptoRandomSource`: `crypto.getRandomValues` as a `rand_core` CSPRNG |
| `minimappr-frontend/tests/field-pwa.spec.js` and `tests/field/*.cjs` | Playwright browser checks plus a harness that starts local `wrangler dev` and the dev server, and builds the fake-mic fixture. |
| `minimappr-frontend/package.json` | `onnxruntime-web` 1.31.1, pinned exactly (released 2026-10-26), as a devDependency. Trunk copies 3 files from `node_modules`. |

Nothing changed in `minimappr-core`, `field-relay-worker`, `TODO.md`, the top-level docs or firmware. The root `Cargo.lock` changed only because of the new optional web-sys/wasm-bindgen deps of `minimappr-transport`.

## Build and run

```sh
cd minimappr-frontend
npm ci                                            # playwright + onnxruntime-web
(cd ../field-relay-worker && npm ci)              # pinned wrangler (local only)
python ../scripts/fetch_birdnet_v3.py             # run from repo root: data/models/birdnet_v3/ (gitignored)
trunk build --release --config Trunk.field.toml   # -> dist-field/
npx playwright test tests/field-pwa.spec.js --reporter=line   # starts wrangler dev + dev server itself
# manual: wrangler dev --local in field-relay-worker, then
FIELD_RELAY_ORIGIN=http://127.0.0.1:8787 node field/field-dev-server.mjs   # http://127.0.0.1:18090
```

## Browser checks (desktop Chromium via Playwright, headless)

RESULTS_PLACEHOLDER

## Design notes

- **Audio path.** The AudioWorklet posts 20 ms chunks tagged with `currentFrame`, which becomes the device-clock sample index.
  - A quantum with no input, or a frame discontinuity, flushes a partial chunk. The index gap then reaches Rust as an exact gap. Rust restarts its rolling buffer and drops the open trigger context, so a clip never splices across a gap. Gaps are counted and shown.
  - Stage-1 runs inline on the main thread (cheap).
  - BirdNET runs one window at a time. Windows queue up to `MAXIMUM_QUEUED_CLASSIFICATION_WINDOWS`; the oldest is dropped and counted.
  - Capture is never blocked by inference.
- **Windows.** 3 s from 0.5 s before the Stage-1 open, then every 3 s while the trigger stays open. The 3 s choice follows birdnet_v3.md's window-dilution table.
- **Resampling.** Capture rate → 32 kHz uses a Kaiser-windowed sinc (32 taps × 256 phases, DC-normalised), not the linear interpolation the metal-detecting app used. Linear interpolation aliases 16–24 kHz into the band BirdNET scores. A non-48 kHz context (Safari may give 44.1 kHz) still classifies. Clips are then disabled, because the core decimator is exact 48→24 kHz only.
- **No level gate anywhere.** Stage-1 is relative to the noise floor, and the BirdNET report threshold (`BIRDNET_REPORT_MINIMUM_SCORE` = 0.25) is an untuned V3 placeholder.
- **Model storage.** The page itself stores the model and labels in Cache Storage (`mmp-field-models-v1`), cache-first. Reason: the first load's fetches happen before the service worker controls the page, so a SW-only cache would miss them and the model would not be offline after one visit. The SW uses the same cache name, and shell-version bumps never purge it.
- **Service worker.** Adapted from the metal-detecting app's `sw.js`: network-first for navigations and stable-named code, cache-first for hashed and large assets. One change: `install` parses `index.html` for the trunk-hashed wasm/js/css it references and precaches them with the stable runtime files (interop JS, worklet, ORT runtime). The shell therefore opens offline after **one** online load; the original app needed a second visit.
- **Relay.**
  - The PWA must be same-origin with the relay. The Worker sends no CORS headers, and the plan serves the PWA and the relay from one Worker. Locally, `field-dev-server.mjs` proxies `/v1/*` HTTP and WebSocket upgrades to `wrangler dev`.
  - Host flow: mints a session id and session key, then `POST /v1/sessions`. The 6-digit secret is minted in the browser and never sent; only the lookup half goes to the Worker. The host answers SPAKE2 flights with `HostRelayJoinCodeResponder`.
  - Joiner flow: `POST /v1/join` with the lookup half, binds a provisional peer id, runs `JoinerRelayKeyExchange`, installs the key, then sends sealed `Heartbeat`/`DetectionEvent`s. Joiners send to the host; the host broadcasts.
  - Sender sequence numbers are seeded from wall-clock ms, so a page reload does not regress them (open issue in relay.md).
- **`bufferedAmount` gate.** `WebRelayDriver` writes one frame at a time and pauses while `bufferedAmount` > 12 KiB. A control frame then waits behind at most about two 4 KiB chunks, not a whole clip sitting in the browser's send buffer. This closes the relay.md browser item.
- **Closure lifetime.** A socket's closures can be the code running when that socket dies. They are parked and freed on the next timer tick, never dropped from inside themselves.

## Attribution (code ported from the user's metal-detecting PWA, read-only source)

`~/Documents-NoCloud/audio_app/On Device Audio App`:
- `web/app-extra/mic-worklet.js` → `field/field-mic-worklet.js`. Changed: the `currentFrame` stamp, a preallocated buffer, partial-chunk flush on gaps.
- `webApp/.../web/audio/MicCapture.kt` (`getUserMediaStrictlyVerified`) and `WebMicrophoneInputSource.kt` (`buildInfCaptureGraph`) → `field/field-capture-interop.js`. Kept: processing-off constraints, the iOS `resume()`, the muted sink. Dropped: the pass-through monitor and the ScriptProcessor fallback.
- `webApp/.../web/inference/WebOnnxClassifier.kt` and `resources/index.html` (global `ort`, WASM EP, single thread without COOP/COEP) → `field/field-birdnet-interop.js`.
- `web/app-extra/sw.js` → `field/field-sw.js` (strategy; see above).

Each file carries an attribution header comment.

## Deviations

- **ORT 1.31.1, not the audio app's vendored 1.18.** V3 is opset 18 with mixed FP16. It is an npm devDependency rather than vendored, to keep about 15 MB of wasm out of git; trunk copies `ort.wasm.min.js` and `ort-wasm-simd-threaded.{mjs,wasm}`.
- **FP16 pruned, not INT8.** The plan row says INT8, but W1 rejected INT8 (the quantized conv stack broke scores).
- **The web adapter lives in `minimappr-transport`** (feature `web-websocket`, cfg wasm32), alongside the native one, rather than in the frontend. The feature is off by default, so `cargo test --workspace` and the native build are unchanged.
- **Detection labels on the wire are V3 class indices** (`SpeciesScore.label_id` = row of the V3 labels CSV). The session's agreed label table is therefore the V3 CSV.

## Open issues

OPEN_ISSUES_PLACEHOLDER
