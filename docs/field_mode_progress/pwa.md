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
| `minimappr-frontend/tests/field-pwa.spec.js`, `tests/field-pwa-webkit.spec.js` and `tests/field/*.cjs` | Playwright browser checks plus a harness that starts local `wrangler dev` and the dev server, and builds the fake-mic fixture. |
| `minimappr-frontend/package.json` | `onnxruntime-web` 1.31.1, pinned exactly (released 2026-10-26), as a devDependency. Trunk copies 3 files from `node_modules`. |

Nothing changed in `minimappr-core`, `field-relay-worker`, `TODO.md`, the top-level docs or firmware. The root `Cargo.lock` changed only because of the new optional web-sys/wasm-bindgen deps of `minimappr-transport`.

## Build and run

```sh
cd minimappr-frontend
npm ci                                            # playwright + onnxruntime-web
(cd ../field-relay-worker && npm ci)              # pinned wrangler (local only)
(cd .. && python scripts/fetch_birdnet_v3.py)     # data/models/birdnet_v3/ (gitignored)
trunk build --release --config Trunk.field.toml   # -> dist-field/
npm run test:field    # Chromium + WebKit specs; each starts wrangler dev + dev server itself
# manual: wrangler dev --local in field-relay-worker, then
FIELD_RELAY_ORIGIN=http://127.0.0.1:8787 node field/field-dev-server.mjs   # http://127.0.0.1:18090
```

## Browser checks (desktop Chromium via Playwright, headless)

The commands are `npm run test:field` (Chromium and WebKit), or `npx playwright test tests/field-pwa.spec.js`. Each run starts its own `wrangler dev --local` and dev server, and they are torn down afterwards. Raw numbers are written to `test-results/field-pwa-measurements.json` and `test-results/field-pwa-webkit-measurements.json`.

### Chromium (Playwright chromium-1228, headless, Apple Silicon laptop shared with other agents' builds): 3/3 pass

| Check | Result |
|---|---|
| **Offline after one online load** | First load registers the SW, which precaches the shell. Then `context.setOffline(true)`, reload. The shell renders, the page is SW-controlled, the model loads from Cache Storage (`modelFromCache: true`) and no page errors occur. |
| **Fake mic** | `--use-file-for-fake-audio-capture` with a fixture built from committed clips: 6 s of forest ambience, then the American Robin song `amro_song_xc132864`, looped. The AudioContext came up at 48000 Hz (the fake track reports 44.1 kHz; Chrome resamples). Processing was off: AEC, NS and AGC all `false`. |
| **Stage-1** (core, in wasm) | Triggered on the bird at a contrast of 15.5 dB. |
| **BirdNET labels it** | **American Robin (*Turdus migratorius*) 0.87**. The detection appears in the shared `AppState` feed through the shim. A FLAC+MMCL clip of the 1.5 s window was **47 085 B**. |
| **Two tabs, one relay session** | Host tab: join code `XXXX-NNNNNN`, connected as peer 1. Joiner tab: typed the code, SPAKE2 ran over the relay, and the session key was installed. A sealed Heartbeat went joiner → host and host → joiner (broadcast), and each side logged it. A third tab with a wrong secret got `join failed` and no key. |
| **Capture gaps** | 1, always at stream start: frames 128 → 384, i.e. two render quanta with no input while the fake track warmed up. None after. |

### Measurements

| What | Value |
|---|---|
| App wasm (release, `opt-level=s`, LTO, wasm-opt `s`) | **928 KB** (`*_bg.wasm`). Includes Leptos plus the unreferenced COP modules (LTO-stripped), `minimappr-core` (Stage-1, rustfft, FLAC encode, postcard, XChaCha20-Poly1305, SPAKE2, HKDF), `minimappr-transport` and the relay wire. |
| ORT Web runtime | `ort-wasm-simd-threaded.wasm` 14.8 MB, `ort.wasm.min.js` 54 KB |
| Model | `BirdNET+_V3.0_Global_12K_FP16_pruned.onnx` 72.3 MB, plus an 8.0 MB labels CSV (one-time download, then Cache Storage) |
| Model fetch (localhost, cold) | 610–650 ms. From Cache Storage: 80–95 ms. |
| `InferenceSession.create` (WASM EP) | 575–673 ms cold; 420–525 ms on the offline reload |
| Inference, one 3 s window, 1 thread (no COOP/COEP) | **367–372 ms** (scratch probe on an idle moment: 281 ms) |
| Inference, 4 threads (`FIELD_CROSS_ORIGIN_ISOLATE=1`, COOP/COEP) | **180 ms** |
| Resample 3 s, 48 → 32 kHz (Rust/wasm) | 3–4 ms |
| Memory after load + capture + inference | Renderer RSS **~304–338 MB**; JS heap 18–25 MB. The ORT wasm heap is outside the JS heap. |

WebGPU EP was not tried: headless, and FP16 WebGPU support is not in the iPhone target's Safari baseline.

### WebKit (Playwright webkit-2311, WebKit 26.5): 3/3 pass, minus the mic

- The model loads (cold fetch 0.4–4.6 s, session 523–581 ms). BirdNET inference on a synthetic 3 s window takes **312–330 ms** single-threaded.
- The two-tab join by code and the sealed exchange work.
- With the page server process killed (real connection refused), the reload renders from the SW and the model loads from cache (`modelFromCache: true`). `context.setOffline` + reload is an internal error in Playwright WebKit when an SW is present, so the server is stopped instead.
- **Mic: not automatable.** Playwright WebKit has no `microphone` permission ("Unknown permission: microphone") and no fake-file capture. `getUserMedia` stays pending, so the status stays "Requesting microphone". That says nothing about real Safari.
- **Real Safari was not driven.** `safaridriver` needs the user to enable "Allow Remote Automation" (Develop menu) once, and Safari's mic prompt can't be pre-granted from automation. The capture graph is the metal-detecting app's, which already runs on iPhone Safari.

## Design notes

- **Audio path.** The AudioWorklet posts 20 ms chunks tagged with `currentFrame`, which becomes the device-clock sample index.
  - A quantum with no input, or a frame discontinuity, flushes a partial chunk. The index gap then reaches Rust as an exact gap. Rust restarts its rolling buffer and drops the open trigger context, so a clip never splices across a gap. Gaps are counted and shown.
  - Stage-1 runs inline on the main thread (cheap).
  - BirdNET runs one window at a time. Windows queue up to `MAXIMUM_QUEUED_CLASSIFICATION_WINDOWS`; the oldest is dropped and counted.
  - Capture is never blocked by inference.
- **Windows.** 3 s from 0.5 s before the Stage-1 open, then every 3 s while the trigger stays open. The 3 s choice follows birdnet_v3.md's window-dilution table.
- **Resampling.** Capture rate → 32 kHz uses a Kaiser-windowed sinc (32 taps × 256 phases, DC-normalised), not the linear interpolation the metal-detecting app used. Linear interpolation aliases 16–24 kHz into the band BirdNET scores. A non-48 kHz context (Safari may give 44.1 kHz) still classifies. Clips are then disabled, because the core decimator is exact 48→24 kHz only.
- **No level gate anywhere.** Stage-1 is relative to the noise floor, and the BirdNET report threshold (`BIRDNET_REPORT_MINIMUM_SCORE` = 0.25) is an untuned V3 placeholder.
- **Model storage.** The page itself stores the model and labels in Cache Storage (`mmp-field-models-v1`), cache-first. Reason: the first load's fetches happen before the service worker controls the page, so a SW-only cache would miss them and the model would not be offline after one visit. The SW keeps that cache across shell-version bumps and does not intercept `/models/` (intercepting would write the 72 MB body twice). The page calls `navigator.storage.persist()`. A failed store (quota) is not fatal: the model runs online and the status says it is not stored offline. A cached model that ORT rejects (truncated write) is deleted and refetched once.
- **Service worker.** Adapted from the metal-detecting app's `sw.js`: network-first for navigations and stable-named code, cache-first for hashed and large assets. One change: `install` parses `index.html` for the trunk-hashed wasm/js/css it references and precaches them with the stable runtime files (interop JS, worklet, ORT runtime). The shell therefore opens offline after **one** online load; the original app needed a second visit. Review additions: a fresh `index.html` prunes hashed assets referenced by neither it nor the previous shell (bounded growth across deploys); a changed `ort.wasm.min.js` drops the cache-first `/ort-wasm-*` companions (an ORT upgrade never pairs a new loader with a stale wasm); navigations fall back to the cached shell after 4 s (lie-fi) while the fetch refreshes in the background. Shell cache is `mmp-field-shell-v2`.
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

- **Real-iPhone validation outstanding:** Safari capture rate (44.1 vs 48 kHz; clips switch off below 48 kHz), whether processing is really off, ORT Web single-thread latency on iPhone (desktop is 280–370 ms per 3 s window), and memory headroom (about 300 MB renderer RSS on desktop). iOS may evict a tab well below that on older phones. The plan's fallback is Mic-only, which already happens when the model fails to load.
- **Threads need COOP/COEP.** 4 threads halve inference (372 → 180 ms). The production Worker serving the PWA should send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. Same-origin relay WebSockets are unaffected. It is unverified whether iOS Safari gives ORT threads under isolation.
- **Model download is 72 MB plus 8 MB of labels.** The CSV has about 40 translated-name columns; an English-only label JSON (about 600 KB) would cut it. A regional species subset (birdnet_v3.md open item) would trim the head too.
- **Browser-side join failure detail.** A refused WebSocket upgrade (401/409/410) is just `error` + close 1006 in the browser, so the PWA cannot tell a stolen peer id from an expired session. The driver keeps reconnecting with backoff. A pre-flight `GET` status endpoint on the Worker would fix that; it was not added because the Worker is out of this stream's area.
- **Joiners keep their provisional peer id** (≥ 0x8000_0000). `JoinAccepted` / host-assigned ids and the reconnect with the assigned id are not wired in the PWA yet.
- **The fixed 0.25 report threshold** and the 3 s windows are V3 placeholders, to be re-tuned with W3-birdnet-cutover.
- **Stage-1 and BirdNET run on the main thread.** Inference is async in ORT (WASM EP proxies to its own worker only when threaded), but a busy UI thread could still delay worklet messages. If real phones show capture gaps, the next step is moving Stage-1, the rolling buffer and the ORT session into a dedicated Worker. Worklet chunks already carry `currentFrame`, so a gap is detected and counted, never spliced.
- **`playwright install webkit` garbage-collected the old `chromium-1223` / `chromium_headless_shell-1223` from `~/Library/Caches/ms-playwright`.** Only 1228 is used by this repo. Another project pinned to an older Playwright would need `npx playwright install` again.
- **No Rust unit tests in the frontend crate.** It is wasm-only and has no Rust test harness. The pipeline and resampler are covered end to end by the browser checks.
