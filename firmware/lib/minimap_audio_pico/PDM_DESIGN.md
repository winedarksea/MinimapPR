# sirith_planar PDM capture + decimation design (Phase 2)

Covers the 5-mic PDM planar array's capture and decimation chain: PIO
capture, the CIC^4/half-band decimator (`PdmCicDecimator`), the PIO+DMA+core-1
source (`PdmPlanarSource`), and the timestamp convention. See the firmware
plan's design decisions D2-D5 for the original requirements; this document
records what was actually implemented and measured, including where reality
diverged from the plan's rough estimates.

## Clocking (D2)

PDM chip rate is fixed at 3.072 MHz. An integer `clk_sys` / chip-rate ratio is
required -- a fractional PIO `clkdiv` dithers edges by ~6.5 ns, which is not
acceptable for a studio-grade capture path.

- Bench (Pico 2 W, stock 12 MHz crystal): `clk_sys = 153.6 MHz = 50 x
  3.072 MHz` (a 2.4% overclock over the RP2350's nominal 150 MHz -- flagged,
  not bench-validated against flash timing at the time of writing).
- Final board (RP2354A + 12.288 MHz 2.5 ppm TCXO): `clk_sys = 122.88 MHz =
  40 x 3.072 MHz`, no overclock.

`sirith_planar.cpp`'s `setClockForPdm()` calls `set_sys_clock_khz(153600,
false)` before any PIO init, and only when PDM audio is active
(`MMPR_NODECFG_AUDIO_INPUT_MODE=2`) -- synthetic-audio bring-up
(`=3`) intentionally skips the clock change. `mmpr_pdm_rx.pio`'s SM runs at
`clkdiv=1.0` in both cases; the per-half-period cycle count (25 for the
bench overclock, 20 for the final TCXO clock) is a **host-loaded runtime
parameter** (`Y`, computed from `clock_get_hz(clk_sys)`), not a hardcoded
delay immediate, so the same compiled program is correct on both clocks --
this mirrors the `mmpr_sirith_tdm_in` Y-preload idiom already used in
`mmpr_audio_rx.pio`.

**Not bench-verified:** the actual 153.6 MHz overclock's effect on flash
read timing (Risk #3 in the firmware plan); the settle-loop cycle count's
alignment with the mic's actual data-valid window (~130-155 ns after the
clock edge per `HARDWARE_REVIEW.md`) has not been checked against a scope.

## CIC^4 R=32 + half-band decimate-by-2 (D3)

Direct decimate-by-64 CIC (order 4) leaves -10.6 dB of passband droop at
20 kHz (per the plan); splitting into CIC^4 R=32 (-2.5 dB droop, measured --
see below) followed by a compensated decimate-by-2 stage recovers a flat
response to 20 kHz.

### Stage B: CIC^4 R=32, LUT-based polyphase FIR

Mathematically, CIC^4 decimate-by-R is the R-fold self-convolution of a
length-R boxcar, four times -- a 125-tap FIR (`4*(32-1)+1`). Rather than the
classic recursive integrator/comb structure (4 running-sum integrators at
the full 3.072 MHz chip rate, decimate, 4 comb/differentiator stages), this
is implemented as a byte-at-a-time LUT: the 125 taps are padded to 128 (16
bytes) and `gen_pdm_filters.py` precomputes, for each of the 16
byte-positions in that window and each of the 256 possible byte values, the
byte's signed (bipolar +-1 chip) contribution to the CIC sum. Producing one
CIC output is then 16 table lookups + adds per channel, not up to 128
individual bit operations -- this is what makes 5 channels at 3.072 MHz
tractable on a Cortex-M33.

- LUT size: 16 phases x 256 entries x 4 bytes (int32) = **16 KiB**, matching
  the plan's ~16 KB target.
- Throughput: one CIC evaluation per 4 completed bytes (32 chips) per
  channel = 96 kHz x 16 adds x 5 channels ~= **7.7M ops/s**, well inside the
  ~10M ops/s budget the plan cites (the half-band stage adds more; see
  below).
- Bit growth: CIC^4 R=32 has DC gain `R^4 = 2^20` exactly (confirmed by the
  host test asserting `sum(taps) == 1048576`). int32 accumulators have
  ample headroom (plan cites 21 bits; actual is 20 + sign).
- Measured droop at 20 kHz: **-2.51 dB** (plan's estimate: -2.6 dB) --
  cross-checked two independent ways in the host test: the closed-form CIC
  magnitude formula, and a DFT evaluation of the taps *decoded back out of
  the shipped LUT* (a finite-difference trick: flipping one chip bit changes
  the LUT sum by exactly `2*tap`, letting the test recover the taps
  in true chronological order and confirm the LUT is exactly linear).

### Stage C: droop-compensated decimate-by-2 FIR

`gen_pdm_filters.py`'s `design_halfband()` builds this via
`scipy.signal.firls` (weighted least squares against a piecewise-linear
target: the inverse of the measured CIC droop across 0-20 kHz, tapering to
0 in a stopband starting at 27.5 kHz -- the frequencies that actually fold
into the 0-20 kHz passband after decimate-by-2 are 28-48 kHz, so 20-28 kHz is
transition slack).

**Tap-count tradeoff (documented in code and here because it is a real,
deliberate deviation from the plan's rough estimate):** D4 pins the total
group delay to ~260 us, which at CIC R=32 means the half-band stage gets a
budget of about 47 taps (`(N-1)/4` us at 48 kHz; more taps directly costs
more delay). An early attempt at a flat-plus-taper design via
`scipy.signal.firwin2` at 47 taps could not hit +-0.1 dB passband ripple at
any reasonable stopband edge -- `firwin2`'s frequency-sampled windowing
tracks a smooth compensation curve poorly (up to ~1 dB of Gibbs-like error).
Switching to `firls`'s piecewise-linear multi-band fit resolves this at the
*same* 47-tap budget:

| Metric | Achieved | Target |
|---|---|---|
| Passband ripple (combined CIC+halfband, 0-20 kHz) | -0.081 / +0.037 dB | +-0.1 dB |
| Alias rejection (28-48 kHz fold band) | ~80 dB | >=90 dB (aspirational) |
| Group delay (CIC + halfband) | 259.77 us | ~260 us |

Alias rejection lands at ~80 dB rather than the plan's aspirational >=90 dB
-- a deliberate choice to hold the group-delay figure, since D4's timestamp
correction and cross-node TDOA bias depend on that constant being right.
80 dB is not a hard architectural ceiling: `design_halfband()`'s `stop_lo`
and `stop_weight` parameters trade rejection for either ripple or delay: e.g.
71 taps (~385 us delay) reaches ~97 dB rejection with much tighter ripple.
Revisit if bench testing shows 80 dB is insufficient and the delay budget
has slack (see the docstring in `gen_pdm_filters.py`'s `design_halfband()`).

Coefficients are shipped as Q20 fixed-point `int32` (`kHalfbandTapsQ` in the
generated `PdmFilterCoeffs.h`); quantization error is <5e-7 per tap (checked
in the generator's ad hoc verification while tuning), far below the
half-band stage's own ripple budget.

### Stage D: DC block + dither

A first-order DC blocker (`y[n] = x[n] - x[n-1] + a*y[n-1]`, `a` derived
from a ~10 Hz corner at 48 kHz) removes the PDM stream's average bias, then
triangular-PDF dither (sum of two independent uniform PRNG draws) is added
before truncation to `int16`. Both run at the final 48 kHz rate (cheap:
5 channels x 48 kHz x a handful of double-precision ops), implemented with
plain `double` arithmetic rather than fixed point -- the RP2350's Cortex-M33
has a hardware FPU and this is a small fraction of Stage B's op count, so
simplicity/correctness was prioritized over further optimization here.

### Regenerating the coefficient tables

`firmware/scripts/gen_pdm_filters.py` is the single source of truth for both
filters. Run `python3 firmware/scripts/gen_pdm_filters.py --write` (requires
numpy + scipy) to regenerate
`firmware/lib/minimap_audio_pico/include/mmpr/PdmFilterCoeffs.h`; the script
prints a design report (droop, ripple, alias rejection, group delay) that
should be diffed against this document if the design (R, order, tap counts)
ever changes.

## Timestamps (D4)

The raw PDM chip index -> decimated (48 kHz) sample index conversion is an
exact division by 64 (`kCicDecimation * kHalfbandDecimation = 32 * 2`).
`PdmPlanarSource::snapshotProducerState()` performs this conversion from the
live DMA transfer-count register (core-0-only state; see D5 below) and then
subtracts `PdmCicDecimator::groupDelayMicroseconds()` (259.77 us, expressed
in decimated-sample units at the configured output rate) so the reported
sample position is **acoustic-arrival-time**, not raw-capture-time.
`readFrame()`'s per-frame `AudioCaptureTimestamp` uses the same convention
via the decimated-sample counter core 1 maintains as it commits full frames.

**Sign convention:** the group delay is *subtracted* (the acoustic event
happened `groupDelayMicroseconds()` earlier than when the filtered sample
carrying its energy becomes available) -- getting this backwards would
introduce a systematic, appear-instantaneous ~260 us bias in any
planar-to-tetra cross-node TDOA solve (Risk #6 in the firmware plan). This
has not been cross-checked against a live dual-node acoustic-impulse test
(see "Not yet bench-validated" below); the host tests only verify the
*magnitude* of the constant against the analytic formula, not its sign in
the live DMA-snapshot path.

## Core split (D5)

`GpsPpsTimerCapture::onIrq()` (`minimap_node_runtime`) calls
`audioSource_->snapshotProducerState()` directly from a PIO IRQ context on
core 0 (this is pre-existing behavior, unchanged by Phase 2 -- see
`GpsPpsTimerCapture.cpp`). This pins all producer-state/timestamp
bookkeeping to core 0: `PdmPlanarSource`'s DMA IRQ (`onDmaIrq()`), its
`nextRawWordIndex_`/`blockStartRawWordIndex_` bookkeeping, and
`snapshotProducerState()` itself are all core-0-only and never touch
anything core 1 writes.

Core 1 (launched via `pico_multicore`'s `multicore_launch_core1()`, the
first multicore use in this codebase) is a **pure DSP consumer**:
`core1DecodeLoop()` pops raw PDM word-block descriptors signaled by the
inter-core SIO FIFO (`multicore_fifo_push/pop_timeout_us`, non-blocking on
the push side so the DMA IRQ never stalls; core 1 falls back to polling
`rawBlockReadIndex_`/`rawBlockWriteIndex_` if a doorbell is ever dropped),
runs `PdmCicDecimator::processRawWords()`, and writes decimated int16 frames
into a second ring that core 0's `readFrame()` drains. Core 1 never touches
DMA/PIO/IRQ registers.

Ring sizes are each a single named constant for easy bumping:
`kPdmRawWordsPerBlock` / `kPdmRawBlockRingSize` (raw capture,
`PdmPlanarSource.h`) and `PdmPlanarConfig::ringFrames` (decimated output,
sized via the existing `kAudioRingFrames` derivation in `node_config.h`).

**Not yet bench-validated:** actual core-0/core-1 hand-off latency and
whether `kPdmRawWordsPerBlock=256` (1024 PDM periods, ~333 us of audio per
block) leaves enough slack for core 1 to keep up while occasionally
servicing other Phase 3 (ESP32-C5 link) work -- see Risk #1 in the firmware
plan. The SPSC ring discipline has not been stress-tested beyond the host
CIC/halfband math tests (which don't exercise the ring/doorbell code at
all, since that requires real hardware).

## PIO bit layout (Stage A)

`mmpr_pdm_rx.pio` produces 24-bit autopush words: 4 PDM clock periods, 2
samples each (rising + falling half-cycle), 3 data-line bits per sample.
`PdmCicDecimator::processRawWords()` documents and implements the exact
bit-to-channel mapping this assumes (chronologically-first group in the
most-significant bits; within a group, bit0=line0/GP1, bit1=line1/GP2,
bit2=line2/GP3; line0 rising->ch0, falling->ch1; line1 rising->ch2,
falling->ch3; line2 rising->ch4, falling unused). This is a
**firmware-internal convention** -- there is no external spec it must
match, only internal self-consistency between the PIO program and the
deinterleave code, which is unverifiable without a logic analyzer on real
hardware. The host test `testDeinterleaveChannelMap` exercises the
deinterleave logic (not the PIO program) against a synthetic word stream
with a known bit pattern.

## What is and isn't verified

**Host-tested (bit-exact / deterministic, see
`nodes/sirith_planar/tests/host/test_pdm_cic_decimator.cpp`):**
- CIC^4 LUT linearity and exact DC gain (`2^20`).
- CIC droop at 20 kHz vs. the closed-form formula (two independent
  derivations agree to <0.01 dB).
- Combined CIC+halfband passband ripple and alias rejection vs. the shipped
  coefficient tables.
- Generated group-delay constant vs. an independently-recomputed analytic
  formula.
- `PdmCicDecimator`'s LUT-based Stage B is bit-exact against an
  independent direct-convolution reference (a different code path computing
  the same linear operator from decoded taps).
- Stage A deinterleave channel routing (`processRawWords`).
- End-to-end delta-sigma-modulated sanity checks (DC/silence, rough
  passband flatness).

**NOT verified without a bench (flagged throughout the code as
HARDWARE-DEPENDENT):**
- Real PIO state-machine timing (settle-loop cycle count vs. actual mic
  data-valid window).
- The 153.6 MHz bench overclock's effect on flash read timing.
- DMA IRQ / core-1 hand-off latency and ring-buffer sizing under real load.
- The group-delay correction's *sign* in the live `snapshotProducerState()`
  path (only its magnitude is host-tested).
- Erratum RP2350-E9 mitigation (external pull-downs) -- firmware only
  avoids the internal pull-down; the external resistors are a board-level
  concern tracked in `HARDWARE_REVIEW.md`.
