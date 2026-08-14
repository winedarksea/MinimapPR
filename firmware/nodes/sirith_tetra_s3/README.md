# sirith_tetra_s3 — tetrahedral node on an ESP32-S3

The same node as [`nodes/sirith_tetrahedral`](../sirith_tetrahedral) with the radio and the capture
peripheral swapped. Same node type (`sirith_tetra`, binary code `1`), same 4-sensor geometry, same
MMB3 wire format, same 32 kHz / 4 ch / 512-samples-per-frame audio. Only the node id, the `hardware`
string and the `firmware` string differ, so the server cannot tell the two apart in the pipeline —
which is the point.

## Why it exists

The live Bluefin deployment measures **~45 % audio frame loss** on the Pico 2 W node, with firmware
reporting `PUBLISH_QUEUE_OVERFLOW`. The prime suspect is the CYW43439 gSPI uplink: the audio payload
is 2.05 Mbit/s (32 kHz × 4 ch × 16-bit) against an *assumed, never-measured* goodput floor of
1.5 Mbit/s baked into the Pico's `node_config.h` (`kAssumedFloorKbitPerSec`).
[`docs/audio_uplink_bandwidth_review.md`](../../../docs/audio_uplink_bandwidth_review.md) closes with
"measure first", and every proposed software lever was rejected as counterproductive or a large
project.

Swapping the radio is the cheapest way to test the hypothesis. The ESP32-S3 has an integrated Wi-Fi
MAC (tens of Mbit/s TCP, not a 2 Mbit/s SPI shim) and an I²S peripheral with native TDM support, so
it replaces both the CYW43 and the PIO TDM receiver. **Goal: stream real 4-channel TDM audio into
MinimapPR with no frame loss, and produce a number to compare against the Pico.**

Everything built here for Wi-Fi + HTTP uplink under ESP-IDF is directly reusable as the uplink half
of the ESP32-C5 SPI→Wi-Fi bridge the board redesign plans — hence
`lib/minimap_transport_esp_wifi` carries no S3-specific code.

## What is shared, not forked

`NodeProtocol` (the MMB3 serializer), `NodeRunner` (capture → packetize → queue → batch → publish,
adaptive backoff, drop policy, every `RunnerStats` counter the measurement rests on) and `NodeClock`
build from `lib/minimap_node_runtime` unchanged. The only platform-specific calls in that subset go
through [`mmpr/PlatformTime.h`](../../lib/minimap_node_runtime/include/mmpr/PlatformTime.h).
PCM slot decoding is shared too, via
[`mmpr/PcmSlotDecode.h`](../../lib/minimap_node_runtime/include/mmpr/PcmSlotDecode.h) — the ESP node's
value depends on being bit-comparable with the Pico on the same input, which only holds if both run
the same arithmetic.

New for this node:

| Library | Contents |
|---|---|
| `lib/minimap_audio_esp32s3` | `EspI2sTdmSource` (`IAudioSource`), `EspI2sTdmLoopback` (self-test) |
| `lib/minimap_transport_esp_wifi` | `EspWiFiSupport`, `EspHttpFramePublisher` (`IUplinkTransport`), `EspSntpClient`, `EspNodeStatsServer` |
| `lib/minimap_ui_tft_st7789` | `TftStatusDisplay` + an embedded 8×8 font |

## Task model

| Task | Core | Job |
|---|---|---|
| capture | 1 | blocking `i2s_channel_read` → decode slots → int16 interleaved → ring + `AudioCaptureTimestamp` |
| runner | 1 | `NodeRunner::loopOnce()` — drains the ring, packetizes, drives the transport |
| publisher | 0 | blocking socket connect/send/recv, signalled by the transport |
| stats / TFT | 0 | `:8082` JSON listener and the 2 Hz status page |

`NodeRunner` stays single-threaded (runner task only), preserving its existing invariants. The
capture task is the DMA-IRQ equivalent, which is what keeps `availableFrames()` /
`ringFramesHighWater()` meaning the same thing they mean on the Pico.

## Toolchain

ESP-IDF v6.0.2 under `~/.espressif`. Per shell:

```bash
source ~/.espressif/tools/activate_idf_v6.0.2.sh
#   or: . ~/.espressif/v6.0.2/esp-idf/export.sh
```

The Feather enumerates as `/dev/cu.usbmodem1101` (native USB-CDC, no bridge chip).

## Build, flash, monitor

```bash
cd firmware/nodes/sirith_tetra_s3
idf.py set-target esp32s3          # first time only
idf.py build
idf.py -p /dev/cu.usbmodem1101 flash monitor
```

`sdkconfig.defaults` is the checked-in source of truth; `sdkconfig` is generated and gitignored. After
editing the defaults, `rm -f sdkconfig && idf.py build`.

Two of those defaults are load-bearing rather than preferences:

- `CONFIG_COMPILER_CXX_EXCEPTIONS=y` — `NodeRunner` catches `std::bad_alloc` around its publish-batch
  vector growth, and ESP-IDF disables exceptions by default.
- `CONFIG_FREERTOS_HZ=1000` — at the IDF default of 100 Hz the runner's 1 ms idle nap becomes a 10 ms
  stall.

### Configuration overrides

`node_config.h` knobs can be overridden on the command line without editing the file; the project's
`CMakeLists.txt` forwards a whitelist of `MMPR_NODECFG_*` names as compile definitions:

```bash
idf.py -DMMPR_NODECFG_AUDIO_INPUT_MODE=0 build          # real TDM capture
idf.py -DMMPR_NODECFG_ENABLE_I2S_LOOPBACK=1 build       # loopback self-test at boot
idf.py -DMMPR_NODECFG_ENABLE_TFT_STATUS=0 build         # headless
idf.py -DMMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ=48000 build  # after loss is at zero
```

**The default build is synthetic audio** (`MMPR_NODECFG_AUDIO_INPUT_MODE=3`). A bare Feather with
nothing attached streams valid MMB3 at exactly the real data rate, which is all the uplink
measurement needs. Set mode `0` once the ADAU7112 is wired.

Wi-Fi credentials and the ingest URL live at the top of
[`include/node_config.h`](include/node_config.h).

## Pin map (provisional)

| Signal | GPIO | Note |
|---|---|---|
| TDM BCLK (out) | 5 (D5) | S3 is master |
| TDM FSYNC/WS (out) | 6 (D6) | no BCLK+1 adjacency constraint, unlike the Pico's PIO |
| TDM SDATA (in) | 9 (D9) | ADAU7112 tri-states between slots |
| Loopback DOUT | 10 (D10) | self-test only; jumper D10 → D9 |

Occupied by the board: 7, 39, 40, 45, 21, 35/36 (TFT), 41/42 (I²C/STEMMA), 33/34 (NeoPixel). Free
after the above: 1, 2, 8, 11–18. The S3 routes I²S through the GPIO matrix, so any of these can be
reassigned when the tetrahedral board mounts.

`TFT_I2C_POWER` (GPIO 21) must be driven **high** or the panel is dead — the display rail is gated
behind it, and without it every SPI transaction "succeeds" onto a dark screen.

## Loopback self-test (no ADAU7112 required)

Validates TDM framing and channel ordering on a bare board:

1. Jumper **D10 → D9**.
2. `idf.py -DMMPR_NODECFG_AUDIO_INPUT_MODE=0 -DMMPR_NODECFG_ENABLE_I2S_LOOPBACK=1 build flash monitor`
3. At boot a TX channel on the same I²S port (full duplex, sharing the RX channel's BCLK/FSYNC)
   emits `0x111111` in slot 0, `0x222222` in slot 1, `0x333333` in slot 2, `0x444444` in slot 3. The
   test asserts every slot decodes to the right PCM16 value in the right output-channel position —
   with the default slot map (`{1, 0, 3, 2}`) that is `8738, 4369, 17476, 13107` on channels 0–3.

```
I (…) mmpr-i2s-lb: PASS 4 frames, channels decode to 8738 4369 17476 13107 (slots 1 0 3 2)
```

It catches a wrong `outputChannelToSlot` map, a wrong slot count, a wrong `validBits` /
`captureBitOffset` decode, and a dead data path. It **cannot** catch a physical mic ↔ slot strapping
error, or a `ws_width`/`ws_pol` that disagrees with the real ADAU7112 — TX and RX share the slot
config, so the test is self-consistent by construction with respect to the actual waveform. Those
need mics attached and a bearing sanity check through the pipeline.

## Status page

Ten lines at 2 Hz on the 240×135 TFT. Lines carrying a non-zero fault counter turn red, so "no loss"
is a colour rather than a reading exercise:

```
sirith-tetra-s3-4f2a 00:14:22
wifi catlin        -52 dBm
ip   192.168.8.141
clk  ntp       +2.4ppm
aud 32000Hz/4ch synthetic
ring  3/13  q  2/19  hw 5
pub 12840 lat  41 ewma  38
 drop0 ovf0 gap0
 http  202 fail 0/0
cap51203 heap231k ntp3
```

## The measurement

The node serves the same stats JSON as the Pico's `NodeControlServer`, on the same port and path, so
the existing tooling works unmodified:

```bash
curl -s http://<feather-ip>:8082/api/v1/stats | python -m json.tool
python scripts/node_stress_report.py                                  # transport_health per node
python scripts/collect_ingest_regression_metrics.py --node-base-url http://<feather-ip>:8082
```

Run the ESP node and the Pico node against `192.168.8.103:8081` and compare. Pass criteria over a
≥30-minute run at 32 kHz / 4 ch:

- `frames_dropped` and `queue_overflows` at **0** (the Pico shows sustained `PUBLISH_QUEUE_OVERFLOW`);
- no `"Detected ingest frame sequence gap"` in the server log;
- `ring_frames_high_water` well under `ring_frames_capacity`;
- `publish_latency_ewma_ms` stable, not climbing — a climbing EWMA is the 24-minute-pipeline-lag
  signature from the earlier live-box work.

The server also synthesizes CBIT reports from these counters, so
`CBIT_WARN: PUBLISH_QUEUE_OVERFLOW` / `AUDIO_CAPTURE_DROPPED` on the node's BIT history is a second,
independent read on the same question.

**Sanity on the audio itself:** with mics attached, pull a snippet through the normal pipeline and
confirm localization produces sane bearings. That catches a physical channel-order or slot-mapping
error the loopback test cannot.

## Protocol correctness without hardware

Point the node at the sidecar and confirm it appears in `GET /api/v1/nodes` as `online` with
`node_type: sirith_tetra`. The MMB3 encoder is shared with the Pico so the format is right by
construction, but the acceptance criteria in `minimappr/api/binary_ingest.py` are worth checking
explicitly: `channels == len(sensor_offsets_m)` (4),
`end_sample_index == start_sample_index + samples_per_channel`, and zero trailing bytes.
`tests/test_ingest_binary_geo.py::_binary_ingest_payload_mmb3` is the Python-side reference for a
byte-level diff.

## Known limitations

- **`snapshotProducerState()` returns false.** The IDF I²S driver does not expose the live
  in-descriptor DMA transfer count that the Pico reads for sub-sample PPS alignment, so GPS/PPS
  discipline is out of scope. Block-boundary accuracy comes from stamping `esp_timer_get_time()` in
  the driver's `on_recv` callback rather than at `read()` return. Filling the gap means a PCNT unit
  on FSYNC, shorter DMA descriptors plus interpolation, or locking the audio clock to a
  GPS-disciplined TCXO — and on the C5/C6/P4 (which have ETM, unlike the S3) it stops being a puzzle
  at all.
- **413 and 400 are not specially retried.** `IUplinkTransport` has no "shrink the batch" or "never
  retry" seam; both are logged with an actionable message and fall through to
  `kPublishBatchMaxRetries` attempts then discard. A 503's `Retry-After` *is* honoured, clamped to
  1.5 s, by pausing the publisher task rather than by failing the batch.
- **No GPS/PPS, I²C sensors, BLE scanning, piezo, or `FailureSnapshot` watchdog breadcrumbs.** All
  deferred; none of them affect the uplink measurement.
- **The TFT gaps (`xGap`/`yGap`) and the ST7789 mirror/swap flags are bench-unverified.** Wrong
  values show as a shifted or wrapped image, not a crash.
- **Higher sample rates are deliberately held at 32 kHz** for comparability. Once loss is at zero,
  the 48/96 kHz sweep is the natural follow-up and the config knob already exists.
