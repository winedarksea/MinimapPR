# Firmware

Embedded firmware projects and shared libraries for MinimapPR nodes.

## Layout

Three runtime families live here. Which one a node uses follows from its MCU and build system:

| Library | Build system | Purpose |
|---|---|---|
| `lib/minimap_node_runtime` | CMake (Pico SDK + ESP-IDF) | node runtime — protocol, runner, clock discipline, GPS/PPS capture, transport QoS, sensor interfaces. The portable subset builds for both platforms; the GPS/sensor half is RP2350-only |
| `lib/minimap_node_core` | PlatformIO | original Arduino/ESP32 runtime — protocol builder, HTTP publisher, WiFi helpers |
| `lib/minimap_audio_pico` | CMake + PlatformIO | RP2040/RP2350 PIO audio sources (TDM-4, PDM, mono I2S) and the PDM CIC/halfband decimator |
| `lib/minimap_audio_esp32` | PlatformIO | ESP32 I2S audio source drivers (Arduino `driver/i2s.h`, for `point_single_mic`) |
| `lib/minimap_audio_esp32s3` | CMake (ESP-IDF) | ESP32-S3 I2S/TDM capture + internal loopback self-test |
| `lib/minimap_transport_cyw43` | CMake (Pico SDK) | Pico W / Pico 2 W CYW43 WiFi uplink |
| `lib/minimap_transport_espc5` | CMake (Pico SDK) | ESP32-C5 companion-radio uplink (for non-W Pico boards) |
| `lib/minimap_transport_esp_wifi` | CMake (ESP-IDF) | ESP-IDF station WiFi, blocking-socket HTTP uplink, SNTP, node stats server |
| `lib/minimap_ui_tft_st7789` | CMake (ESP-IDF) | 240x135 ST7789 status page |
| `lib/cmake/mmpr_nodecfg.cmake` | CMake | forwards a node's `node_config.h` settings into the shared libraries |

| Node target | Board | Audio | Uplink |
|---|---|---|---|
| `nodes/sirith_tetrahedral` | `pico2_w` (RP2350) | 4-mic TDM via ADAU7112, 32 kHz | CYW43 WiFi |
| `nodes/sirith_tetra_s3` | Adafruit ESP32-S3 TFT Feather | 4-mic TDM via ADAU7112, 32 kHz (synthetic by default) | ESP32-S3 integrated WiFi |
| `nodes/sirith_planar` | `pico2` (RP2350) | 5-mic PDM planar array, 48 kHz | ESP32-C5 companion |
| `nodes/point_single_mic` | `esp32dev` | single I2S mic | ESP32 WiFi |

`minimap_node_runtime` builds for both the Pico SDK and ESP-IDF: its `CMakeLists.txt` registers the
platform-portable subset (`NodeProtocol`, `NodeRunner`, `NodeClock`, `Base64`, the synthetic/silence
audio sources) as an IDF component when `ESP_PLATFORM` is set, and the pico-sdk `INTERFACE` library
otherwise. The only platform-specific calls in that subset live behind
[`mmpr/PlatformTime.h`](lib/minimap_node_runtime/include/mmpr/PlatformTime.h) — a monotonic clock,
two delays, and a free-heap reading. Adding a third platform means adding a branch there, not forking
the protocol or runner code.

## Shared Runtime (`minimap_node_runtime`)

Used by every Sirith node, on either platform:

- node protocol payload builder (`POST /api/v1/ingest/frame`) and base64 PCM encoding
- `NodeClock` frame timestamping, `GpsPpsTimerCapture` PPS discipline, and `ClockHoldoverModel`
  for drift estimation across PPS gaps
- `NmeaGpsSource` UART time/position ingest
- `AudioTransportQos` — backpressure and drop policy on the uplink
- `IUplinkTransport` abstraction (CYW43 or ESP32-C5 behind the same interface)
- `IAudioSource` + `NodeRunner` publisher loop
- optional sensors: SHT4x environmental, LSM6 temperature, LIS2MDL magnetometer with
  `MagAutoOrientation` smoothing, piezo buzzer

`minimap_node_core` is the older PlatformIO-only equivalent, still used by the ESP32 point node.

## Sirith Tetrahedral (`nodes/sirith_tetrahedral`)

Four-channel TDM capture from an ADAU7112 on a Pico 2 W (RP2350). The Pico drives `BCLK`/`FSYNC`
in master mode, samples `SDATA`, and reconstructs four slots per frame via PIO.

Also supports optional BLE scanning (`MMPR_ENABLE_BLE_SCAN`), posting observations to
`/api/v1/ingest/ble`.

### Configure

Edit `nodes/sirith_tetrahedral/include/node_config.h`:

- `MMPR_NODECFG_AUDIO_INPUT_MODE` — `0` TDM 4-mic, `1` I2S mono, `3` synthetic (no hardware)
- WiFi credentials and `MMPR_NODECFG_SERVER_BASE_URL`
- audio rate/channels/frame size (`32000` Hz, `4`, `512` by default)
- TDM sample edge, capture bit offset, and data-pin bias
- `MMPR_NODECFG_GPS_PPS_PIN` and GPS UART settings
- BLE scan intervals and audio-queue watermarks
- optional LSM6 IMU temperature telemetry (`metadata.temperature_c`, `metadata.temperature_source`)

### Build and flash

This target uses the Raspberry Pi Pico SDK via CMake, not PlatformIO. The easiest path is the
Raspberry Pi Pico VS Code extension, which provisions the toolchain under `~/.pico-sdk/` and wires
up the *Compile Project* / *Run Project* tasks in `.vscode/tasks.json`.

From the command line, using that same toolchain:

```bash
cd firmware/nodes/sirith_tetrahedral
~/.pico-sdk/cmake/v3.31.5/bin/cmake -S . -B build -G Ninja \
  -DCMAKE_MAKE_PROGRAM=~/.pico-sdk/ninja/v1.12.1/ninja
~/.pico-sdk/ninja/v1.12.1/ninja -C build
~/.pico-sdk/picotool/2.2.0-a4/picotool/picotool load build/sirith_tetrahedral.uf2 -fx
```

Adjust the version directories to match what your `~/.pico-sdk/` actually contains — the extension
pins them per install.

## Sirith Tetra S3 (`nodes/sirith_tetra_s3`)

The same tetrahedral node on an ESP32-S3 instead of a Pico 2 W: same node type (`sirith_tetra`), same
4-sensor geometry, same MMB3 wire format, same 32 kHz / 4 ch / 512-sample audio parameters. It exists
to answer one question — the live deployment measures ~45 % audio frame loss on the Pico node with
firmware reporting `PUBLISH_QUEUE_OVERFLOW`, and the prime suspect is the CYW43439 gSPI uplink
against a never-measured 1.5 Mbit/s goodput floor. Swapping the radio for an integrated WiFi MAC and
re-running the same measurement is the cheapest way to test that.

Because the audio parameters and the counters are identical, the existing tooling does the comparison
unmodified — see [nodes/sirith_tetra_s3/README.md](nodes/sirith_tetra_s3/README.md) for build, flash,
loopback and measurement procedure.

Defaults to **synthetic** audio (`MMPR_NODECFG_AUDIO_INPUT_MODE=3`), so a bare Feather with nothing
attached streams valid MMB3 at the real data rate; build with `-DMMPR_NODECFG_AUDIO_INPUT_MODE=0` for
real TDM capture. The server needs no changes: there is no registration handshake, and the full
`NodeDescriptor` rides in every batch.

## Sirith Planar (`nodes/sirith_planar`)

Five-mic coplanar PDM array on a plain Pico 2 (RP2350): four corner mics at a 25 mm radius plus one
center mic, all at `z = 0`. PDM is captured over three data lines (2+2+1 sharing) and decimated on
core 1 by `PdmCicDecimator` — see [lib/minimap_audio_pico/PDM_DESIGN.md](lib/minimap_audio_pico/PDM_DESIGN.md).

The board is deliberately `pico2` rather than `pico2_w` so no CYW43/lwIP/btstack is linked in; WiFi
is offloaded to an ESP32-C5 companion over `minimap_transport_espc5`.

Coplanar arrays have a mirror ambiguity about their plane, so the server applies a `half_space`
constraint (`upper` by default for `sirith_planar` nodes) to fold mirror-symmetric solutions.

Configure via `nodes/sirith_planar/include/node_config.h`; build exactly as for the tetrahedral node,
substituting the target name. Hardware notes are in
[nodes/sirith_planar/HARDWARE_REVIEW.md](nodes/sirith_planar/HARDWARE_REVIEW.md).

## Point Node Reference (`nodes/point_single_mic`)

A single-mic ESP32 target on the PlatformIO/Arduino runtime — the simplest node that speaks the
protocol, and the reference for adding new ESP32-class hardware.

```bash
cd firmware/nodes/point_single_mic
pio run
pio run -t upload
pio device monitor
```

## Host Tests

The pico-free model code (clock holdover, transport QoS, PDM decimation) builds and runs off-target
with a plain host toolchain:

```bash
cd firmware/nodes/sirith_tetrahedral/tests/host   # or nodes/sirith_planar/tests/host
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

> **macOS note:** if the Command Line Tools libc++ headers are broken in your environment, the host
> build fails at `#include <vector>` and similar. Point the compiler at the SDK's copy explicitly:
> `-DCMAKE_CXX_FLAGS="-isystem $(xcrun --show-sdk-path)/usr/include/c++/v1"`.

`lib/minimap_transport_espc5/tests/host` follows the same pattern.

## Timestamping

Nodes timestamp frames from `NodeClock`, disciplined in order of preference:

1. **GPS PPS** — `GpsPpsTimerCapture` latches the RP2350 timer on each PPS edge; `NmeaGpsSource`
   supplies absolute time from the UART sentences. Reports `time_quality: gps_locked`.
2. **Holdover** — when PPS drops out, `ClockHoldoverModel` extrapolates from the measured drift rate
   rather than falling straight back to free-running.
3. **NTP/SNTP** (ESP32 targets) — `EspSntpClient` latches each sync and the runner feeds it to
   `NodeClock::applyNtpObservation`, so frames report `ntp_disciplined` rather than `free_running`
   (which the server down-weights in fusion). Without any of the above, frames are anchored to the
   build timestamp and report `free_running`.

Time quality is sent per frame and used by the server to weight fusion.

## Adding a Node Type

**Pico SDK (CMake) family** — copy `nodes/sirith_planar` as the template:

1. Create `nodes/<new_node>/` with `CMakeLists.txt`, `pico_sdk_import.cmake`, `include/node_config.h`,
   and a `<new_node>.cpp` main.
2. `add_subdirectory` the runtime, an audio lib, and a transport lib from `../../lib`, and include
   `lib/cmake/mmpr_nodecfg.cmake` to forward your `node_config.h` settings.
3. Implement an `IAudioSource` for the hardware in `lib/minimap_audio_pico` or a new hardware lib.
4. Reuse `NodeRunner`, `NodeClock`, and the `IUplinkTransport` implementations.

**ESP-IDF (CMake) family** — copy `nodes/sirith_tetra_s3` as the template:

1. Create `nodes/<new_node>/` with `CMakeLists.txt`, `sdkconfig.defaults`, `include/node_config.h`,
   and `main/{CMakeLists.txt,<new_node>.cpp}`.
2. List the shared libraries in `EXTRA_COMPONENT_DIRS`. A directory named there that contains its own
   `CMakeLists.txt` registers as one component named after the directory, so only the libraries you
   name are ever scanned — that is what keeps the pico-only libraries out of an IDF build.
3. `CONFIG_COMPILER_CXX_EXCEPTIONS=y` is mandatory, not a preference: `NodeRunner` catches
   `std::bad_alloc` around its publish-batch vector growth. `CONFIG_FREERTOS_HZ=1000` matters too —
   at the IDF default of 100 Hz the runner's 1 ms idle nap becomes a 10 ms stall.
4. Implement an `IAudioSource` for the hardware and an `IUplinkTransport` if the uplink is new.

**PlatformIO family** — copy `nodes/point_single_mic`:

1. Create `nodes/<new_node>/` with `platformio.ini`, `src/main.cpp`, `include/node_config.h`.
2. Set `lib_extra_dirs = ../../lib` so `minimap_node_core` and the audio libs resolve.
3. Implement an `IAudioSource` in `lib/minimap_audio_esp32` or a new hardware lib.

Either way, fill the `NodeDescriptor` with real geometry and capabilities — the server derives
`sensor_offsets_m`, half-space constraints, and localization strategy from it. That keeps protocol
and transport behavior consistent across all node types.
