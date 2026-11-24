# Pico 2 W + CYW43439 multichannel capacity — results

*Bench run 2026-11-23 on the USB-attached Pico 2 W (live node `sirith-tetra-07b5`),
plan: "Pico 2 W multichannel lossless audio capacity test". Raw data:
`bench_results/2026-11-23/` (Stage 1 `stage1_tmr50.jsonl`, Stage 2/3 `*.json` +
`stage2_table.md`, serial logs alongside).*

## Verdict

**The CYW43439 radio is not the limit. The firmware was.**

- The radio and gSPI bus sustain **10.8–13.5 Mbit/s** of TCP goodput (BLE on/off),
  with peaks above 17 Mbit/s. That's ≥ 1.5× every config except C5 with BLE on,
  which is marginal.
- The production firmware collapses at **every** config, C1 (2.82 Mbit/s) included.
  This bench reproduces the field's ~45 % loss as a timeout/reconnect death spiral.
- Three lwIP/publisher settings change the picture (loss falls to 0–0.7 %; per-config numbers below):
  - cap bytes in flight at 8×MSS
  - shorten lwIP's TCP timer
  - make the HTTP timeout longer than a loss recovery

  Average goodput then matches or exceeds the payload rate for every config, and
  publish errors drop to zero.
- What's left is occasional 0.6–1.2 s stalls in a heavily contended 2.4 GHz channel.
  They outlast the RAM-limited queue. No config passed the strict 0-gap criterion
  for a full run with BLE on; **C2 (48 k × 4) lost 0 frames in 10 min with BLE on**.

| Config | Payload | Stage 1 gate (1.5×) | Prod firmware | Fixed firmware, BLE on | Fixed firmware, BLE off | Limiting factor |
|---|---|---|---|---|---|---|
| C1 44.1k×4 | 2.82 | pass | **FAIL** (4,160 lost/5 min) | 454 lost/10 min (1 episode) | — | radio environment (burst stalls) |
| C2 48k×4 | 3.07 | pass | not run (C1 and C5 bracket it) | **0 lost/10 min** ¹ | — | — (queue high-water 79/83) |
| C3 48k×5 | 3.84 | pass | not run (bracketed) | 1,150 lost/10 min (2 episodes) | **34 lost/10 min** | radio environment |
| C4 48k×6 | 4.61 | pass | not run | not run | not run | — |
| C5 48k×8 | 6.14 | marginal (BLE on) / pass (off) | **FAIL** (25,829 lost/5 min) | 5,691 lost/10 min | **375 lost/10 min** | link + RAM-capped queue |

¹ Flagged FAIL by the harness only for two ~21 ms timestamp steps (NTP re-anchor, see
Timing below) and queue high-water ≥ 50 %; zero sequence/sample gaps, bit-exact.

Every frame that arrived in every run was **bit-exact** against the synthetic pattern
(0 bit errors in ~600 k frames). All loss is whole frames dropped at the node's queue.

## Environment (read this before generalising)

- Only the "near AP" position was measured (RSSI −34 to −40 dBm). The Pico is
  USB-tethered to the Mac, so the field position (−65 to −75 dBm) is **not covered**.
- The 2.4 GHz channel is busy. The AP is on channel 4 (20 MHz), overlapping
  neighbours on channels 5/6 (−48 to −64 dBm), with a noise floor around −82 dBm. The
  other live Pico node (`sirith-tetra-1a15`) streams on the same channel and is itself
  stuck in the production timeout spiral. MAC retry rate measured 24–34 %.
- The receiver for Stages 1–3 was the Mac's wired port (192.168.8.199). Bluefin was at
  load ~140 at the start, which would have contaminated "server wait".

## Stage 1 — raw link ceiling (`firmware/nodes/bench_cyw43_iperf`)

A custom TCP blaster rather than lwiperf: lwiperf's client has a fixed 10 s run and no
per-second report, and Bluefin (ostree) has no iperf2. It shares the production
`lwipopts.h`, `WiFiSupport` and `BleRssiScanner`. Each variant is 3 × 60 s, saturating,
with TCP_TMR_INTERVAL 50 ms; driven by `scripts/bench/cyw43_link_ceiling.py`.

| sys MHz | gSPI div | BLE | in-flight cap | mean Mbit/s | worst run | TCP rexmit (3 min) | chip rxnobuf |
|---|---|---|---|---|---|---|---|
| 150 | 2 | off | 8×MSS | **13.47** | 12.68 | 270 | 0 |
| 150 | 2 | on | 8×MSS | **10.77** | 9.00 | 386 | 0 |
| 150 | 3 | on | 8×MSS | 11.49 | 11.12 | 95 | 0 |
| 200 | 2 | on | 8×MSS | 11.93 | 11.55 | 382 | 0 |
| 150 | 2 | on | none (32×MSS, production) | 7.85 | 7.66 | 418 (2 min) | **11,680** |

- **The production send buffer starves the chip.** With `TCP_SND_BUF = 32×MSS`
  (46 KB in flight), the CYW43439's shared TX/RX packet pool fills with queued TX
  frames, and the chip drops inbound frames: our TCP ACKs (wl counter `rxnobuf`,
  ~30 % of inbound in some runs). TCP then advances in RTO-paced bursts. Capping in
  flight at 8–16×MSS removes the drops: 2.5 → 14–18 Mbit/s in the uncapped/capped
  sweep, with the 250 ms default timer.
- **lwIP's RTO floor is about 3 slow-timer ticks:** 1.5 s at the default
  `TCP_TMR_INTERVAL` of 250 ms, against a ~3 ms LAN RTT. A 50 ms timer cuts the worst
  ACK stall from ~1.4 s to ~0.3–0.9 s.
- BLE passive scan (3 % duty) costs ~20 % of goodput. gSPI divider 3 (25 MHz) gives
  the same throughput with far fewer retransmits. 200 MHz sys clock buys nothing.
- Ruled out: TX-credit stalls in the cyw43 driver. They exist (the driver discards
  inbound frames while stalled, `cyw43_ll.c` `cyw43_sdpcm_send_common`), but were
  measured at ~2 ms/s. Raising the MAC retry limits (SRL 7 / LRL 4 → 15) changed
  nothing.
- Not run: the `threadsafe_background` arch variant (supported via `-DBENCH_ARCH`).

## Stage 2 — production firmware, synthetic audio, bit-exact sink

Sink: `scripts/bench/mmb3_capacity_sink.py`, with unit tests in
`tests/test_mmb3_capacity_sink.py`. Driver: `scripts/bench/pico_capacity_run.py`.

With production defaults (auto HTTP timeout ~425 ms, 32×MSS send buffer, 250 ms TCP
timer, ~300 ms queue), the node enters a **death spiral**:
1. Any loss that fast retransmit doesn't repair waits for the 1.5 s RTO.
2. `HttpFramePublisher`'s `totalDeadline = timeoutMs` (425 ms) fires first and
   `tcp_abort`s the connection.
3. The 300 ms queue overflows, and the next 28 KB batch burst re-overfills the chip
   pool.

Results: C1 lost 4,160 frames in 5 min (182 publish errors); C5 lost 25,829 in 5 min,
with goodput down to 0.54 Mbit/s. Server-wait stayed ~4 ms, so the link/protocol
interaction is the problem, not the server.

## Stage 3 — firmware iterations (it2 values are now the production defaults; all remain `-D` overridable)

| Iteration | Change | C3 BLE on | C5 BLE on |
|---|---|---|---|
| base | production | (spiral) | 25,829 lost/5 min, 717 errs |
| fixall | `TCP_SND_BUF_MSS=8`, `TCP_TMR_INTERVAL_MS=50`, `HTTP_TIMEOUT_MS=2000` | 3,717 lost, **0 errs** | 3,022 lost, **0 errs**, goodput 6.21 |
| it2 | + `TCP_TMR_INTERVAL_MS=25`, `TARGET_QUEUE_MS=1000`, `AUDIO_BUFFER_RAM_CEILING_KB=360` | 1,150 lost | 5,691 lost ² |
| it2, BLE off | same | **34 lost** | — |
| it3, BLE off | + `PUBLISH_BATCH_FRAMES=8` | — | **375 lost** |

² C5 at 8 ch has only ~450 ms of queue even at 360 KB, and its sustained HTTP goodput
with BLE on (5.8 Mbit/s) sits just under its 6.14 payload.

The RAM ceiling could be raised because **free heap was measured for the first time**:
173–253 KB free at the old 240 KB BLE-on ceiling. `freeHeapBytes()` was a stub returning 0
on RP2350; it is now implemented from `mallinfo()`. At 360 KB, 43–62 KB stays free.

The plan's remaining Stage 3 levers were not needed to reach this point, and wouldn't
fix the residual stalls:
- tighter Wi-Fi servicing — the main loop already polls every iteration
- overlapping requests — server-wait is ~4 ms, not the bottleneck
- core1 capture

The candidates for the residual stalls are environmental: channel choice, and stopping
the neighbouring node's retry storm.

## Stage 4 — real Rust sidecar on Bluefin (C2, BLE on, it2 fixes, 5.5 min)

The live server was used as-is (it already runs `MINIMAPPR_DIRECT_INGEST_ENABLED=false`).
The node streamed as `bench-pico-07b5`.

- **Node:** 0 queue overflows and 0 publish errors. Server wait was 4 ms EWMA,
  35 ms max.
- **Server:** `packet_loss_total` was unchanged (155,545 before and after), with no
  sequence gaps for the bench node. `overload_rejections` was 0, and ~30 k frames
  were received.

**Ingest capacity: PASS.**

**Separate server finding (pre-existing, not caused by the bench):** on the live box
`frames_processed` sat at 98,945 for the whole afternoon. `/api/v1/dsp/status` showed
`last_processed_ns` ~6.2 h behind `last_heartbeat_ns`, with 125 raw manifests queued.
Ingest is accepting audio, but the DSP worker is not advancing.

## Stage 5 — real ADAU7112 TDM (4 ch, BLE on, it2 fixes, 10 min each)

| Config | frames lost | node capture drops | capture ring high-water | publish errors | timestamp jumps |
|---|---|---|---|---|---|
| C1 44.1 kHz | **0** | 0 | 2/18 | 0 | 12 |
| C2 48 kHz | 123 (0.2 %, 6 episodes) | 0 | 2/19 | 0 | 0 |

Real capture DMA and the fixed publisher coexist: the ring never got past 2 frames,
and no capture blocks were dropped. Bit-exactness doesn't apply to live audio.
Timestamp jumps on C1 again track clock discipline (see Timing). Not done: the
30-minute runs and the 120 s calibration capture / clap GCC-PHAT check.

## Bugs found and fixed along the way

1. **GPS PPS capture IRQ storm → watchdog reset (pre-existing, affects field nodes).**
   `GpsPpsTimerCapture::initPioStateMachine()` enabled the PIO1 "RX FIFO not empty"
   IRQ before assigning `pio_`, `sm_` and `configured_`. If the state machine had
   already pushed (PPS pin high at start), `onIrq()` returned without draining, the
   level IRQ re-entered forever, and `begin()` never returned. The result was a 16 s
   watchdog reset at every lazy PPS arm (GPS baud lock, ~12 s after boot) until a
   start happened to be clean. It first looked like "`:8082/api/v1/stats` kills the
   node", because the curl timing coincided. The July production image reproduces it.
   Diagnosed with a temporary SysTick PC sampler and HardFault hook writing to
   `__uninitialized_ram`. Fixed by publishing state before enabling the IRQ.
2. **`cyw43_wifi_get_rssi()` from the node main loop hangs with BLE on.** An attempt to
   report real RSSI on the Pico was reverted, and the comment now records why. RSSI in
   this report comes from the Stage 1 bench, which doesn't run btstack traffic
   concurrently.
3. **`PUBLISH_BATCH_FRAMES` override tripped the RAM static_assert,** because the
   queue was sized for the auto batch. The derivation now budgets the effective batch.

## Timing finding (separate from capacity)

Indoors the GPS sees PPS but no fix, so the node is NTP-disciplined. NTP re-anchors
step `start_time_ns` by **~21 ms** (twice, 1 s apart, during the C2 run). The sink
flags these as timestamp jumps. For TDOA this is a real defect in GPS-less operation,
worth its own look (slew instead of step); it isn't a link-capacity issue.

## Firmware changes made

- `node_config.h`: `kAssumedFloorKbitPerSec` 1500 → **9000** (measured, BLE on, worst run).
  It is paired with an auto HTTP-timeout floor of 2000 ms, replacing the old
  [150, 850] ms clamp. On its own, a higher floor would have *shortened* the timeout
  further below the RTO. The production default timeout moves from ~425 ms to 2000 ms.
- **New production defaults (the it2 config):** `TCP_SND_BUF` 8×MSS (was 32),
  `TCP_TMR_INTERVAL` 25 ms (was lwIP's 250), publish queue target 1000 ms (was 300),
  audio RAM ceiling 360 KB for BLE on and off (was 240/300). Each is overridable
  with `-D`.
- `HttpFramePublisher` / `NodeRunner` / `:8082` stats: upload vs server-wait split
  (`publish_upload_*`, `publish_server_wait_*`), and `bytes_sent`.
- `freeHeapBytes()` implemented on RP2350.
- Override knobs: `MMPR_NODECFG_TCP_SND_BUF_MSS`, `MMPR_NODECFG_TCP_TMR_INTERVAL_MS`
  (`lwipopts.h`); `MMPR_NODECFG_TARGET_QUEUE_MS`, `MMPR_NODECFG_AUDIO_BUFFER_RAM_CEILING_KB`
  (`node_config.h`).
- CMake: `-DMMPR_NODECFG_ENABLE_BLE_SCAN=0|1` override.
- Synthetic mode:
  - advertises N-sensor geometry (tetra at 4 channels, planar ring otherwise)
  - reserves no DMA ring
  - uses node id prefix `bench-pico-`, so a bench stream never overwrites a live node
- PPS IRQ ordering fix and batch-budget fix (above).

## Recommendation

1. **Done (now the defaults):** `TCP_SND_BUF` 8×MSS, `TCP_TMR_INTERVAL` 25 ms, HTTP
   timeout ≥ 2 s, queue target 1 s, audio RAM ceiling 360 KB. Flash every Pico node.
   The first build (TDM 32 kHz, BLE on) on node 07b5 ran 3.5 min against the live
   server with 0 overflows, 0 publish errors and no resets; queue high-water was 47/57,
   absorbing a 765 ms stall, with ~60 KB heap free.
2. Watch `heap_free_bytes` on the deployed nodes. The 360 KB ceiling leaves ~45–60 KB
   free.
3. **Sample rate / channels:** 48 kHz × 4 (C2) is the highest config that ran clean
   with BLE on in this environment. 48 k × 5 is viable with BLE off (0.06 % loss) or
   with a cleaner channel. 8-channel / 24-bit-equivalent (C5) is radio-viable on paper
   but not with BLE on and the current RAM budget.
4. **Environment:** move the AP off channel 4 (to 1, 6 or 11, and the least crowded
   of those). Deploy the fixed firmware to *all* Pico nodes; one spiralling node
   pollutes the shared channel for the others.
5. Re-run at a field-position RSSI (−65 to −75 dBm) before relying on any of this
   beyond the near-AP case.
