#pragma once

// PdmCicDecimator -- Stage B/C/D of the sirith_planar PDM capture pipeline
// (D3 in the firmware plan). Pure, portable C++: no pico-sdk headers, no
// MCU-specific intrinsics, so it can be compiled and unit-tested on the
// host (see nodes/sirith_planar/tests/host/test_pdm_cic_decimator.cpp) and
// then linked unmodified into the RP2350 firmware.
//
// Pipeline per channel:
//   Stage B: CIC^4 decimate-by-32 (3.072 MHz chip rate -> 96 kHz), realized
//            as a LUT-based polyphase FIR (see PdmFilterCoeffs.h) instead of
//            the classic recursive integrator/comb structure -- mathematically
//            identical (both compute the same boxcar^4 convolution), but this
//            form does one 32-bit table lookup + add per input byte rather
//            than per input bit, which is what makes 5 channels at 3.072 MHz
//            tractable on a Cortex-M33 core.
//   Stage C: droop-compensated FIR decimate-by-2 (96 kHz -> 48 kHz),
//            compensating the CIC passband droop up to 20 kHz.
//   Stage D: DC block (~10 Hz corner) + TPDF dither -> int16.
//
// This class only implements Stages A(low-level)/B/C/D at the chip-bit
// level (pushChipBit / pushChipByte); the PIO raw-word deinterleave (the
// hardware-format-dependent half of "Stage A" in the plan) lives in
// PdmPlanarSource, which unpacks captured PIO words into per-channel chip
// bytes and feeds them here. Keeping that split means the part that must be
// bit-exact (this file) never depends on anything hardware-specific.
//
// D4 timestamps: kGroupDelayMicroseconds (see PdmFilterCoeffs.h) is the
// constant total group delay of stages B+C. Callers (PdmPlanarSource)
// subtract it from capture timestamps so published sample times represent
// acoustic-arrival-time, not raw-capture-time.

#include <cstddef>
#include <cstdint>

#include "mmpr/PdmFilterCoeffs.h"

namespace mmpr {

// Bumping the channel count is the one place a firmware build needs to touch
// besides node_config.h; everything else in this file is channel-count-generic.
static constexpr size_t kPdmMaxChannels = 5;

class PdmCicDecimator {
 public:
  // If true, add triangular-PDF dither before the final int16 truncation
  // (standard practice for audio quality). Host tests that need bit-exact,
  // reproducible output disable this.
  struct Config {
    bool enableDcBlock = true;
    bool enableDither = true;
    uint32_t ditherSeed = 0x9E3779B9u;
  };

  PdmCicDecimator() : PdmCicDecimator(Config{}) {}
  explicit PdmCicDecimator(const Config& config);

  void reset();

  // Push one raw PDM chip bit (0 or 1) for `channel`, sampled at the chip
  // rate (3.072 MHz nominal). Internally accumulates 8 chips into a byte
  // before doing any CIC work, so most calls are cheap (a shift + counter
  // increment); every 8th call for a channel does one CIC window evaluation
  // (~16 LUT adds), and every 64th call (R_total = 32*2) may additionally
  // produce one decimated 48 kHz output sample.
  //
  // Returns true and writes *outSample if a decimated sample was produced.
  bool pushChipBit(size_t channel, int bit, int16_t* outSample);

  // Convenience: push a full byte (bit 0 = oldest chip in the byte, bit 7 =
  // newest -- i.e. bits arrive MSB-first into the shift register and this
  // is the already-assembled result). Equivalent to 8 calls to pushChipBit
  // but skips the intermediate per-bit byte assembly.
  bool pushChipByte(size_t channel, uint8_t byteValue, int16_t* outSample);

  // Stage A: deinterleave raw PIO capture words and feed the 5 channels.
  //
  // Word format (see node_config.h's "Deinterleave -> output channel map"
  // comment, and PDM_DESIGN.md): each 24-bit word packs 4 PDM clock periods,
  // 6 bits each (2 half-cycles x 3 data lines = `in pins,3` sampled twice
  // per period). Within a word, the first-in-time 3-bit group occupies the
  // most-significant bits: group g (g=0..7, g even=rising-half, g odd=
  // falling-half of period g/2) occupies bits [24-3*(g+1) .. 23-3*g]. Within
  // a 3-bit group, bit0=line0 (GP1), bit1=line1 (GP2), bit2=line2 (GP3).
  // Channel map: line0 rising->ch0, falling->ch1; line1 rising->ch2,
  // falling->ch3; line2 rising->ch4, falling-> discarded (line2's falling
  // half carries no mic in the 2+2+1 sharing scheme).
  //
  // This exact bit layout is a firmware-internal convention (there is no
  // external spec to match); PdmPlanarSource's mmpr_pdm_rx.pio must produce
  // words in this layout, which is not hardware-verifiable without a bench
  // -- see PDM_DESIGN.md. The CIC/halfband math downstream of this function
  // does not depend on the layout being "correct" in any absolute sense,
  // only on this function and the PIO program agreeing with each other.
  //
  // Writes up to maxOutSamplesPerChannel decimated samples per channel,
  // interleaved as [s0ch0, s0ch1, s0ch2, s0ch3, s0ch4, s1ch0, ...] into
  // interleavedOut. Returns the number of samples produced per channel.
  size_t processRawWords(
      const uint32_t* words,
      size_t wordCount,
      int16_t* interleavedOut,
      size_t maxOutSamplesPerChannel);

  // Test/diagnostic hook: the most recent raw CIC^4 (Stage B, 96 kHz-domain,
  // pre-halfband) output pushed for `channel`. Lets host tests verify Stage
  // B bit-exactness independent of Stage C/D. Not used by the hot path.
  int32_t debugLastCicOutput(size_t channel) const {
    return channel < kPdmMaxChannels ? channels_[channel].halfbandDelay[0] : 0;
  }

  static constexpr size_t maxChannels() { return kPdmMaxChannels; }
  static constexpr int totalDecimation() {
    return pdm_filters::kCicDecimation * pdm_filters::kHalfbandDecimation;
  }
  static constexpr double groupDelayMicroseconds() {
    return pdm_filters::kGroupDelayMicroseconds;
  }

 private:
  struct ChannelState {
    // Rolling byte history for the CIC window: history_[0] is the newest
    // completed byte, history_[kCicLutPhases-1] the oldest. Shifted on every
    // new completed byte (every 8 chips).
    uint8_t history[pdm_filters::kCicLutPhases] = {};
    uint8_t partialByte = 0;
    int partialBitCount = 0;
    // CIC^4 R=32 decimation = 4 completed bytes (32 chips) per evaluation of
    // the LUT window; this counts completed bytes since the last evaluation.
    int bytesSinceCicEval = 0;

    // Stage C (halfband) delay line, most-recent-first, and a running count
    // of CIC-stage (96 kHz) samples produced, used to decide when a
    // decimate-by-2 halfband output is due.
    int32_t halfbandDelay[pdm_filters::kHalfbandTapCount] = {};
    int cicSamplesProduced = 0;

    // Stage D: DC blocker state and dither PRNG state. Stage D runs at the
    // final 48 kHz rate (5 channels x 48 kHz is cheap), so it is implemented
    // with double arithmetic for simplicity/correctness rather than fixed
    // point -- the RP2350's Cortex-M33 has a hardware single/double FPU, and
    // this is a small fraction of the CIC stage's op count.
    double dcPrevX = 0.0;
    double dcPrevY = 0.0;
    uint32_t ditherState = 0;
  };

  int32_t evaluateCicWindow(const ChannelState& state) const;
  int32_t evaluateHalfband(const ChannelState& state) const;
  int16_t applyDcBlockAndDither(ChannelState& state, int32_t halfbandOutQ20);
  uint32_t nextDither(ChannelState& state);

  Config config_;
  ChannelState channels_[kPdmMaxChannels];
};

}  // namespace mmpr
