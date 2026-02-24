#pragma once

#include <driver/i2s.h>

#include <cstddef>
#include <cstdint>

#include "mmpr/IAudioSource.h"

namespace mmpr {

struct SirithDualI2SPins {
  int bclk0;
  int ws0;
  int dataIn0;

  int bclk1;
  int ws1;
  int dataIn1;
};

struct SirithDualI2SConfig {
  uint32_t sampleRateHz = 16000;
  size_t frameSamples = 1024;
  int32_t sampleShiftBits = 16;
  uint8_t dmaBufferCount = 8;
  uint16_t dmaBufferLength = 256;
  bool useApll = false;
};

// Reads 2x stereo I2S streams and exposes a single 4-channel interleaved frame.
class SirithDualI2SSource final : public IAudioSource {
 public:
  SirithDualI2SSource(const SirithDualI2SPins& pins, const SirithDualI2SConfig& config);
  ~SirithDualI2SSource() override;

  bool begin() override;
  uint32_t sampleRateHz() const override { return config_.sampleRateHz; }
  uint8_t channels() const override { return 4; }
  size_t frameSamples() const override { return config_.frameSamples; }
  bool readFrame(int16_t* interleavedOut, size_t samplesPerChannel) override;

 private:
  bool initPort(i2s_port_t port, int bclk, int ws, int dataIn);
  void stopAndUninstall(i2s_port_t port);
  void cleanup();
  int16_t toPcm16(int32_t raw) const;

  SirithDualI2SPins pins_;
  SirithDualI2SConfig config_;

  int32_t* raw0_ = nullptr;
  int32_t* raw1_ = nullptr;
  bool port0Installed_ = false;
  bool port1Installed_ = false;
};

}  // namespace mmpr
