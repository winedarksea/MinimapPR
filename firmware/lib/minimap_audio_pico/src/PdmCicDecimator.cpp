#include "mmpr/PdmCicDecimator.h"

#include <cmath>

namespace mmpr {
namespace {

// R=32 decimation for Stage B == 4 completed bytes (32 chips) per CIC
// window evaluation.
constexpr int kBytesPerCicEval = pdm_filters::kCicDecimation / 8;
static_assert(kBytesPerCicEval * 8 == pdm_filters::kCicDecimation,
              "CIC decimation must be a whole number of bytes");

// Total decimation gain contributed by Stage B is exactly
// kCicDecimation^kCicOrder (a power of two, since kCicDecimation is a power
// of two) -- this is how many fractional bits of headroom Stage B leaves
// above the incoming +-1 chip domain. Stage C's coefficients are normalized
// (via droop compensation) to ~unity DC gain in that same domain, so the
// combined domain stays at this many bits before Stage D brings it down to
// int16. Kept as a runtime-computed constexpr (rather than hand-derived) so
// a future change to kCicDecimation/kCicOrder can't silently desync it.
constexpr int log2Exact(int value) {
  int bits = 0;
  while (value > 1) {
    value >>= 1;
    ++bits;
  }
  return bits;
}
constexpr int kCicDomainBits = pdm_filters::kCicOrder * log2Exact(pdm_filters::kCicDecimation);
// int16 full scale is 2^15; leave the rest as the final output shift.
constexpr int kOutputShiftBits = kCicDomainBits - 15;
static_assert(kOutputShiftBits > 0, "CIC domain must have more headroom than int16");

// DC block corner frequency and dither amplitude, at the final 48 kHz rate.
constexpr double kDcBlockCornerHz = 10.0;

double dcBlockCoefficient(double sampleRateHz) {
  return 1.0 - (2.0 * M_PI * kDcBlockCornerHz / sampleRateHz);
}

}  // namespace

PdmCicDecimator::PdmCicDecimator(const Config& config) : config_(config) {
  reset();
}

void PdmCicDecimator::reset() {
  for (size_t ch = 0; ch < kPdmMaxChannels; ++ch) {
    channels_[ch] = ChannelState{};
    channels_[ch].ditherState = config_.ditherSeed + static_cast<uint32_t>(ch) * 0x2545F491u;
    if (channels_[ch].ditherState == 0) {
      channels_[ch].ditherState = 0xA5A5A5A5u;
    }
  }
}

int32_t PdmCicDecimator::evaluateCicWindow(const ChannelState& state) const {
  int64_t sum = 0;
  for (int phase = 0; phase < pdm_filters::kCicLutPhases; ++phase) {
    sum += pdm_filters::kCicLutFlat[phase * 256 + state.history[phase]];
  }
  return static_cast<int32_t>(sum);
}

int32_t PdmCicDecimator::evaluateHalfband(const ChannelState& state) const {
  int64_t sum = 0;
  for (int i = 0; i < pdm_filters::kHalfbandTapCount; ++i) {
    sum += static_cast<int64_t>(pdm_filters::kHalfbandTapsQ[i]) *
           static_cast<int64_t>(state.halfbandDelay[i]);
  }
  // Round-to-nearest when rescaling out of the Q(kHalfbandCoeffScaleBits)
  // fixed-point domain back to the CIC-output domain.
  const int64_t half = int64_t{1} << (pdm_filters::kHalfbandCoeffScaleBits - 1);
  const int64_t rescaled = (sum >= 0) ? ((sum + half) >> pdm_filters::kHalfbandCoeffScaleBits)
                                      : -(((-sum) + half) >> pdm_filters::kHalfbandCoeffScaleBits);
  return static_cast<int32_t>(rescaled);
}

uint32_t PdmCicDecimator::nextDither(ChannelState& state) {
  // xorshift32 -- deterministic, cheap, good enough statistically for TPDF
  // dither (this is not a cryptographic use).
  uint32_t x = state.ditherState;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  state.ditherState = x;
  return x;
}

int16_t PdmCicDecimator::applyDcBlockAndDither(ChannelState& state, int32_t halfbandOutQ20) {
  double sample = static_cast<double>(halfbandOutQ20);

  if (config_.enableDcBlock) {
    static const double kA = dcBlockCoefficient(48000.0);
    const double y = sample - state.dcPrevX + kA * state.dcPrevY;
    state.dcPrevX = sample;
    state.dcPrevY = y;
    sample = y;
  }

  // Bring the CIC-domain sample down to int16 range.
  double scaled = std::ldexp(sample, -kOutputShiftBits);

  if (config_.enableDither) {
    // Triangular PDF dither: sum of two independent uniforms in [-0.5, 0.5)
    // LSB, added before truncation.
    const double u1 = static_cast<double>(nextDither(state)) / 4294967296.0 - 0.5;
    const double u2 = static_cast<double>(nextDither(state)) / 4294967296.0 - 0.5;
    scaled += (u1 + u2);
  }

  double rounded = std::floor(scaled + 0.5);
  if (rounded > 32767.0) rounded = 32767.0;
  if (rounded < -32768.0) rounded = -32768.0;
  return static_cast<int16_t>(rounded);
}

bool PdmCicDecimator::pushChipByte(size_t channel, uint8_t byteValue, int16_t* outSample) {
  if (channel >= kPdmMaxChannels) {
    return false;
  }
  ChannelState& state = channels_[channel];

  for (int i = pdm_filters::kCicLutPhases - 1; i > 0; --i) {
    state.history[i] = state.history[i - 1];
  }
  state.history[0] = byteValue;

  if (++state.bytesSinceCicEval < kBytesPerCicEval) {
    return false;
  }
  state.bytesSinceCicEval = 0;

  const int32_t cicOut = evaluateCicWindow(state);

  for (int i = pdm_filters::kHalfbandTapCount - 1; i > 0; --i) {
    state.halfbandDelay[i] = state.halfbandDelay[i - 1];
  }
  state.halfbandDelay[0] = cicOut;

  ++state.cicSamplesProduced;
  // Decimate-by-2: only every other 96 kHz sample yields a 48 kHz output.
  if ((state.cicSamplesProduced & 1) != 1) {
    return false;
  }

  const int32_t halfbandOut = evaluateHalfband(state);
  if (outSample != nullptr) {
    *outSample = applyDcBlockAndDither(state, halfbandOut);
  }
  return true;
}

namespace {
// Channel indices for the 2+2+1 PDM sharing scheme (see node_config.h).
constexpr size_t kChNe = 0;   // line0 rising
constexpr size_t kChNw = 1;   // line0 falling
constexpr size_t kChSw = 2;   // line1 rising
constexpr size_t kChSe = 3;   // line1 falling
constexpr size_t kChCenter = 4;  // line2 rising (line2 falling is unused)
}  // namespace

size_t PdmCicDecimator::processRawWords(
    const uint32_t* words,
    size_t wordCount,
    int16_t* interleavedOut,
    size_t maxOutSamplesPerChannel) {
  if (words == nullptr || interleavedOut == nullptr) {
    return 0;
  }

  size_t produced = 0;
  for (size_t w = 0; w < wordCount; ++w) {
    const uint32_t word = words[w];
    for (int period = 0; period < 4; ++period) {
      const int risingGroupIdx = period * 2;
      const int fallingGroupIdx = period * 2 + 1;
      const uint32_t risingGroup = (word >> (24 - 3 * (risingGroupIdx + 1))) & 0x7u;
      const uint32_t fallingGroup = (word >> (24 - 3 * (fallingGroupIdx + 1))) & 0x7u;

      int16_t sample[kPdmMaxChannels] = {};
      bool produced5[kPdmMaxChannels] = {};

      produced5[kChNe] = pushChipBit(kChNe, static_cast<int>(risingGroup & 0x1u), &sample[kChNe]);
      produced5[kChSw] = pushChipBit(kChSw, static_cast<int>((risingGroup >> 1) & 0x1u), &sample[kChSw]);
      produced5[kChCenter] =
          pushChipBit(kChCenter, static_cast<int>((risingGroup >> 2) & 0x1u), &sample[kChCenter]);

      produced5[kChNw] = pushChipBit(kChNw, static_cast<int>(fallingGroup & 0x1u), &sample[kChNw]);
      produced5[kChSe] = pushChipBit(kChSe, static_cast<int>((fallingGroup >> 1) & 0x1u), &sample[kChSe]);
      // fallingGroup bit2 (line2 falling) has no mic in the 2+2+1 scheme; discarded.

      // All 5 channels advance in lockstep (one chip pushed per channel per
      // period), so they always decimate together; only act if they did.
      if (produced5[kChNe]) {
        if (produced >= maxOutSamplesPerChannel) {
          return produced;
        }
        for (size_t ch = 0; ch < kPdmMaxChannels; ++ch) {
          interleavedOut[produced * kPdmMaxChannels + ch] = sample[ch];
        }
        ++produced;
      }
    }
  }
  return produced;
}

bool PdmCicDecimator::pushChipBit(size_t channel, int bit, int16_t* outSample) {
  if (channel >= kPdmMaxChannels) {
    return false;
  }
  ChannelState& state = channels_[channel];

  // Newest chip shifts into bit 7; existing bits move toward bit 0 (must
  // match the convention baked into kCicLutFlat by gen_pdm_filters.py).
  state.partialByte = static_cast<uint8_t>((state.partialByte >> 1) | ((bit & 1) << 7));
  if (++state.partialBitCount < 8) {
    return false;
  }
  state.partialBitCount = 0;
  const uint8_t completedByte = state.partialByte;
  state.partialByte = 0;
  return pushChipByte(channel, completedByte, outSample);
}

}  // namespace mmpr
