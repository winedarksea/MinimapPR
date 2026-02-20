# Firmware

This directory contains embedded firmware projects and shared libraries for MinimapPR nodes.

## Layout
- `lib/minimap_node_core`: shared runtime used by all node types
- `lib/minimap_audio_esp32`: ESP32 audio source drivers
- `nodes/sirith_tetra`: Sirith tetrahedral firmware target (4-channel, dual I2S)
- `nodes/point_single_mic`: reference point-node firmware target

## Shared Runtime (`minimap_node_core`)
Common pieces used by every node:
- node protocol payload builder (`POST /api/v1/ingest/frame`)
- base64 encoding for PCM payloads
- HTTP publish transport
- node frame clock / timestamping
- WiFi connectivity helpers
- generic node runner (`IAudioSource` + publisher loop)

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

### Build and Upload
From `firmware/nodes/sirith_tetra`:
```bash
pio run
pio run -t upload
pio device monitor
```

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
