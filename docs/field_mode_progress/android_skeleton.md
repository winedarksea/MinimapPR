# W4-android-skeleton: notes

Branch: `fm/android-skeleton`, off `field-mode` at `292e2dd` (Merge fm/attribution).

## What landed

| Path | Contents |
|---|---|
| `minimappr-ffi/` (new root workspace member) | A thin UniFFI 0.29.4 surface over `minimappr-core`, with four groups: the Stage-1 detector (`FieldStage1Detector`), the clip codec (`encode_clip_from_rolling_buffer`, `decode_clip`), protocol frames (`describe_frame`, `canonicalize_frame`, `known_message_kind_ids`, `field_protocol_version`), and the mesh node (`FieldMeshNode`, which wraps `FieldNodeStateMachine` plus a PCM rolling buffer and Stage 1). It has 15 Rust unit tests. |
| `field-app/` (new; not a Cargo or Python member) | A Kotlin Multiplatform + Compose project with one `:shared` module (targets: Android, host JVM for tests, and iOS arm64 + simulator arm64 when Xcode is present) and one `:androidApp` (activity + microphone foreground service). |
| `Cargo.toml` | Adds `minimappr-ffi` to `members`. `Cargo.lock` gains only new packages (uniffi and its deps). |

The FFI crate builds on the host, so it joined the root workspace. `cargo test -p minimappr-ffi` passes.

## FFI surface decisions

- **Field messages cross the FFI as encoded frames (`Vec<u8>`).** Mirroring the 33-kind `FieldMessage` enum in UniFFI types would tie the Kotlin surface to every protocol change. The shell routes frames between the transport and `FieldMeshNode::receive_frame`, and inspects only the header (`describe_frame`).
- **PCM crosses as little-endian bytes.** UniFFI lowers `Vec<i16>` to a boxed `List<Short>`, which costs 48k allocations per second at 48 kHz. `Vec<u8>` lowers to a `ByteArray`. The packing helpers are in `field-app/.../capture/Pcm16LittleEndianPacking.kt` and `minimappr-ffi/src/pcm16_bytes.rs`.
- **The rolling buffer lives in Rust, inside `FieldMeshNode`.** The node state machine's `NodeAudioWindowSource` (readiness answers, clip requests) reads it directly, so clips never copy audio back across the FFI.
  - A sample-index discontinuity restarts the buffer and counts a gap. A clip that spans a gap would carry wrong timing.
  - Compaction uses 25 % slack, so the memmove happens every few seconds instead of every 20 ms block.
- `FIELD_FFI_SURFACE_VERSION = 1` (`field_ffi_surface_version()`). The activity calls it at launch, so a missing or mismatched `.so` fails at startup rather than on the first capture block.
- Every object guards its core type with a `Mutex` because UniFFI objects must be `Send + Sync`. There is one lock per call, and nothing holds it across calls.

## Toolchain decisions

- **JDK:** the only system JDK is Homebrew OpenJDK 27, and Gradle 8.14 and AGP 8.x cannot run on it.
  - `field-app/scripts/fetch_local_jdk.sh` downloads Eclipse Temurin 21 (21.0.13.1+2, about 200 MB, SHA-256 checked against the Adoptium API) into `field-app/.local-jdk/`, which is gitignored.
  - `field-app/scripts/gradlew_local.sh` sets `JAVA_HOME` to that JDK and `ANDROID_HOME` to `~/Library/Android/sdk`. Neither variable is overridden if it is already set, and no dotfiles or system packages are touched.
  - A Gradle daemon toolchain or foojay would also need the wrapper client to start on JDK 27, so I chose the explicit local JDK.
- **Gradle 8.14.3 wrapper, AGP 8.11.1, Kotlin 2.2.20, Compose Multiplatform 1.9.0, `com.android.library` for `:shared`.**
  - Gobley 0.3.7's own CI runs Kotlin 2.1 / AGP 8.7 with the classic library plugin.
  - The audio app is on AGP 9.1, but AGP 9's built-in Kotlin and `com.android.kotlin.multiplatform.library` are not supported by Gobley 0.3.x, so I did not follow it there.
- **Gobley over plain uniffi-bindgen:** Gobley worked on the first attempt.
  - `dev.gobley.cargo` builds the crate for each Android ABI through the NDK (27.0.12077973), copies the `.so` files into the AAR, and builds the host `.dylib` for JVM and Android unit tests.
  - `dev.gobley.uniffi` generates KMP bindings (package `com.minimappr.field.ffi`).
  - Gobley 0.3.7 pins `uniffi_bindgen =0.29.4`, so the crate pins `uniffi = "=0.29.4"`. The newest uniffi is 0.32, so upgrading means waiting for a Gobley release.
  - The first build runs `cargo install gobley-uniffi-bindgen` once, which takes several minutes.
- **ABIs:** `arm64-v8a` (phones) and `x86_64` (emulators). armv7 and x86 are left out to halve Rust build time. Add them to both `abiFilters` blocks if they are ever needed.
- **minSdk 31** (the audio app's tested floor), **target/compileSdk 36** (the installed platform).

## Copied from the audio app (read-only source, MIT, (c) 2026 Colin Catlin)

Each copied piece carries an attribution comment in its KDoc.

| Field app file | Source | Taken | Not taken |
|---|---|---|---|
| `shared/src/androidMain/.../capture/FieldAudioSourceSelection.kt` | `audio/source/AndroidAudioCaptureSource.kt` (`preferredDetectorAudioSource`) | UNPROCESSED-if-advertised check | — ; **added** the plan's VOICE_RECOGNITION middle step, with MIC as the last resort |
| `shared/src/androidMain/.../capture/AndroidMicrophoneCaptureLoop.kt` | `MicrophoneAudioInputSource.kt` + `recording/AudioRecordingSession.kt` | `AudioRecord.Builder` setup, buffer sizing from `getMinBufferSize`, `READ_BLOCKING` loop on `Dispatchers.IO` | WAV writing, RMS/waveform flows; **added** sample-count indices, gap flagging, source fallback chain |
| `shared/src/commonMain/.../capture/DropWhileBusyBackpressureGate.kt` | `inference/InferenceController.kt` (`handleFrame` in-flight drop + `droppedFrames`) | Drop-newest-while-busy plus a counter | **The RMS energy gate was deliberately not copied.** AGENTS.md forbids absolute level gates that would block faint long-range calls, and Stage 1 is relative to the noise floor instead |

Placement rule: the gate sits only in front of the skippable classifier stage. Every capture block reaches the Rust rolling buffer, because a dropped block would be a TDOA gap. The classifier is still a placeholder that acquires and releases at once, so the gate's counters are live but idle.

## Android foreground service

- `FieldCaptureForegroundService` uses `foregroundServiceType="microphone"`.
  - Permissions: `RECORD_AUDIO`, `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_MICROPHONE`, `POST_NOTIFICATIONS`.
  - Its notification channel has low importance and the notification carries a Stop action.
- Lifecycle:
  - It is started only from the visible activity, because Android 14+ refuses a microphone FGS started from the background.
  - It calls `ServiceCompat.startForeground` immediately and checks `RECORD_AUDIO` before doing so, because without it `startForeground` throws on API 34+.
  - It returns `START_NOT_STICKY`: an OS restart with no visible activity would not be allowed to reopen the mic, so a silent "restart" would misreport the state.
  - `onDestroy` stops capture, cancels the scope and closes the Rust node.
- Status flows through the process-wide `FieldServiceStatusBoard` (a StateFlow) to the shared Compose `FieldApp` screen.
- The node identity is a placeholder (`skeletonSoloNodeSetup`: node 2, expected host 1, zero session id) until QR join lands. There is no transport yet (`discardingOutboundFrameTransport`), so the node keeps sending JoinRequests (counted in the UI) and then enters solo, as a phone that cannot reach its host should.

## Results (this branch, host = Apple Silicon Mac, Command Line Tools only)

| Command | Result |
|---|---|
| `cargo test -p minimappr-ffi` | 15 passed |
| `field-app/scripts/gradlew_local.sh :shared:jvmTest` | 9 passed (4 capture/backpressure, 5 FFI through the real Rust dylib) |
| `... :shared:allTests :shared:testDebugUnitTest :androidApp:assembleDebug :androidApp:assembleDebugAndroidTest` | BUILD SUCCESSFUL. `allTests` = jvmTest (iOS tests are absent without Xcode). Android unit tests: debug 9/9, release 9/9 |
| `... :androidApp:connectedDebugAndroidTest` on emulator `field_smoke_api35` (API 35 `default` arm64-v8a image) | 1/1 passed: app launches, the FFI answers, the service reaches RUNNING with capture from `VOICE_RECOGNITION`, samples advance, the first outbound frame is a JoinRequest (kind 1), and Stop returns to STOPPED |
| `cargo rustc -p minimappr-ffi --target aarch64-apple-ios --lib --crate-type staticlib` | Builds (staticlib needs no SDK linker) |

APK sizes:
- Debug APK: 52 MB. It is dominated by the unstripped debug `libminimappr_ffi.so`: 9.2 MB for arm64 and 16.8 MB for x86_64.
- Release APK (`assembleRelease`): 24.6 MB, unsigned and not minified. The Rust library is 1.4 MB for arm64 and 2.0 MB for x86_64. Most of the remainder is Compose, Material3 and JNA, which R8 would cut. An arm64-only split would drop x86_64.

Emulator: `sdkmanager` installed `emulator` 37.3.10 (1.3 GB) and `system-images;android-35;default;arm64-v8a` (1.7 GB). No account sign-in was needed. The AVD `field_smoke_api35` was created with `avdmanager` (device `pixel_6`) and run headless (`-no-window -no-audio`). AudioRecord still opened and delivered samples.

## iOS

- Full Xcode is not installed (`xcode-select -p` is `/Library/Developer/CommandLineTools`), so Kotlin/Native cannot link an iOS framework.
- `shared/build.gradle.kts` declares `iosArm64` and `iosSimulatorArm64` (static framework `FieldShared`) only when `xcrun --sdk iphonesimulator --show-sdk-path` succeeds. Otherwise it logs "iOS targets skipped". This keeps the Android build green on hosts with Command Line Tools only.
- `shared/src/iosMain/.../MainViewController.kt` is the Compose entry point for a future Xcode project.
- What was verified: the Rust staticlib for `aarch64-apple-ios`. Not verified: the Kotlin/Native compile and link, Gobley's cinterop, and the simulator target (`aarch64-apple-ios-sim` is not installed; Gobley's `rustUpTargetAdd*` task would add it).

## Open issues

1. **Readiness answers are partly placeholders.** `speech_free` and `handling_noise_free` are reported `true`, and `in_band_level_over_floor_db` is `0.0`, because the shell has no VAD and no motion blanking yet. The host's clip selection will not down-weight this phone until those land.
2. **Overrun detection is coarse.** Only failed or zero reads flag a gap. Missing pieces:
   - comparing `AudioRecord.getTimestamp` frame positions against the sample count;
   - AAudio's `getXRunCount` (Phase 1 picks AudioRecord or AAudio).
3. **Missing features:** no `mediaPlayback` FGS type and no chirp playback yet (the plan needs both types once chirps play). Also missing: mic pinning (`setPreferredDevice` to the built-in mic), sensors, thermal headroom polling (the FFI `report_thermal_headroom` exists, unused), a transport (the relay client in `minimappr-transport` is not yet behind the FFI) and QR join.
4. **No classifier.** BirdNET+ V3.0 ONNX is not wired; `can_classify = false` in the skeleton setup.
5. **The release build is not minified.** Gobley generates JNA ProGuard rules, but R8 has not been exercised.
6. **The first Gradle build is slow** (about 23 min here): the `gobley-uniffi-bindgen` install plus Rust builds for 2 Android ABIs and the host, with parallel agents competing for CPU. Incremental builds take about 1 min.
7. **Gradle shows `kotlin.native.ignoreDisabledTargets`** although iOS targets are omitted rather than disabled on CLT hosts. The setting is harmless and is kept for hosts where Kotlin itself disables them.

## Deviations

- `allTests` covers the host-JVM target rather than the Android unit tests, because KMP's `allTests` does not include AGP unit-test tasks. The Android unit tests run through `:shared:testDebugUnitTest` (both pass, same 9 tests).
- The iOS framework could not be built: there is no Xcode, as described in the iOS section above.
