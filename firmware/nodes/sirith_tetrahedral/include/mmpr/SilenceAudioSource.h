#pragma once

#include <cstddef>
#include <cstdint>

#include "mmpr/IAudioSource.h"

namespace mmpr {

class SilenceAudioSource final : public IAudioSource {
 public:
  SilenceAudioSource(uint32_t sampleRateHz, size_t frameSamples, uint8_t channels);

  bool begin() override;
  uint32_t sampleRateHz() const override { return sampleRateHz_; }
  uint8_t channels() const override { return channels_; }
  size_t frameSamples() const override { return frameSamples_; }
  bool readFrame(
      int16_t* interleavedOut,
      size_t samplesPerChannel,
      AudioCaptureTimestamp* captureTimestamp = nullptr) override;

 private:
  uint32_t sampleRateHz_;
  size_t frameSamples_;
  uint8_t channels_;
  uint64_t frameDurationUs_ = 0;
  int64_t nextFrameAtUs_ = 0;
};

}  // namespace mmpr
