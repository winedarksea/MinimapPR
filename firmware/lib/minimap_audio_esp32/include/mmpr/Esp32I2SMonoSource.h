#pragma once

#include <driver/i2s.h>

#include <cstddef>
#include <cstdint>

#include "mmpr/IAudioSource.h"

namespace mmpr {

struct Esp32I2SPins {
  int bclk;
  int ws;
  int dataIn;
};

struct Esp32I2SMonoConfig {
  i2s_port_t port = I2S_NUM_0;
  uint32_t sampleRateHz = 16000;
  size_t frameSamples = 1024;
  int32_t sampleShiftBits = 16;
  uint8_t dmaBufferCount = 8;
  uint16_t dmaBufferLength = 256;
  bool useApll = false;
  uint8_t stereoChannelIndex = 0;
};

class Esp32I2SMonoSource final : public IAudioSource {
 public:
  Esp32I2SMonoSource(const Esp32I2SPins& pins, const Esp32I2SMonoConfig& config);

  bool begin() override;
  uint32_t sampleRateHz() const override { return config_.sampleRateHz; }
  uint8_t channels() const override { return 1; }
  size_t frameSamples() const override { return config_.frameSamples; }
  bool readFrame(int16_t* interleavedOut, size_t samplesPerChannel) override;

 private:
  int16_t toPcm16(int32_t raw) const;

  Esp32I2SPins pins_;
  Esp32I2SMonoConfig config_;

  int32_t* rawStereo_ = nullptr;
};

}  // namespace mmpr
