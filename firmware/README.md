# Firmware

This directory contains embedded firmware projects and shared libraries for MinimapPR nodes.

## Layout
- `lib/minimap_node_core`: shared runtime used by all node types
- `lib/minimap_audio_esp32`: ESP32 audio source drivers
- `lib/minimap_audio_pico`: RP2040/RP2350 Pico audio source drivers
- `nodes/sirith_tetra`: Sirith tetrahedral firmware target (4-channel, dual I2S)
- `nodes/sirith_tetra_pico`: Sirith tetrahedral Pico W firmware target (4-channel, TDM)
- `nodes/point_single_mic`: reference point-node firmware target

## Shared Runtime (`minimap_node_core`)
Common pieces used by every node:
- node protocol payload builder (`POST /api/v1/ingest/frame`)
- base64 encoding for PCM payloads
- HTTP publish transport
- node frame clock / timestamping
- WiFi connectivity helpers
- generic node runner (`IAudioSource` + publisher loop)
- optional environmental telemetry hook (non-fatal if unavailable)

This is the extension point for future node families.

## Sirith Firmware (`nodes/sirith_tetra`)
Current implementation captures 4 channels using **dual stereo I2S**:
- I2S bus 0: channels 0 and 1
- I2S bus 1: channels 2 and 3

This matches Sirith hardware configuration where ADAU7112 is set to two I2S outputs. TDM mode is not yet implemented in this first firmware drop.

### Configure
Edit `nodes/sirith_tetra/include/node_config.h`:
- WiFi credentials
- backend URL
- node position and sensor offsets
- I2S pin mapping
- optional LSM6 IMU temperature telemetry (`metadata.temperature_c`, `metadata.temperature_source`)

### Build and Upload
From `firmware/nodes/sirith_tetra`:
```bash
pio run
pio run -t upload
pio device monitor
```

## Sirith Pico Firmware (`nodes/sirith_tetra_pico`)
Pico W / Pico 2 W target for Sirith in **TDM-4 mode** using RP2040/RP2350 PIO:
- Pico drives `BCLK` and `FSYNC` (master mode)
- Pico samples `SDATA` and reconstructs four 32-bit slots per frame
- default slot map expects top microphone (MK4) in Slot 3

### Default pin map
- `TX` (GPS optional): `GP12`
- `RX` (GPS optional): `GP13`
- `PPS` (GPS optional): `GP10`
- `SCL`: `GP19`
- `SDA`: `GP18`
- `TDM SDATA`: `GP7`
- `TDM BCLK`: `GP8`
- `TDM WS/FSYNC`: `GP9`

### Configure
Edit `nodes/sirith_tetra_pico/include/node_config.h`:
- WiFi credentials and backend endpoint
- TDM slot/channel mapping
- optional GPS and I2C pins
- base-plane rotation (`kBasePlaneRotationSteps`) for manual orientation calibration
- optional LIS2MDLTR auto-orientation smoothing and stability thresholds
- optional LSM6 IMU temperature telemetry (`metadata.temperature_c`, `metadata.temperature_source`)

### Build and Upload
From `firmware/nodes/sirith_tetra_pico`:
```bash
pio run -e sirith_tetra_pico_w_rp2040
pio run -e sirith_tetra_pico_w_rp2040 -t upload
pio device monitor
```
This target uses the `maxgerhardt/platform-raspberrypi` PlatformIO platform in `platformio.ini` for Pico W / Pico 2 W board definitions.

## Point Node Reference (`nodes/point_single_mic`)
A single-mic ESP32 target built on the same shared runtime and protocol.

From `firmware/nodes/point_single_mic`:
```bash
pio run
pio run -t upload
pio device monitor
```

## Timestamping Notes
Current firmware uses:
- NTP sync at startup (if enabled)
- monotonic frame stepping for frame start timestamps

Planned next step is GPS PPS and UART time discipline, already aligned with `NodeClock` extension points.

## Adding More Node Types
1. Create `nodes/<new_node>/` with `platformio.ini`, `src/main.cpp`, `include/node_config.h`.
2. Implement an `IAudioSource` for that hardware in `lib/minimap_audio_esp32` or a new hardware lib.
3. Reuse `NodeRunner`, `NodeClock`, and `HttpFramePublisher` from `minimap_node_core`.
4. Fill `NodeDescriptor` with node geometry/capabilities.

That keeps protocol and transport behavior consistent across all node types.
