#pragma once

#include <cstddef>
#include <cstdint>

#include "mmpr/IAudioSource.h"

namespace mmpr {

struct SirithPicoTdmPins {
  uint8_t dataIn;
  uint8_t bclk;
  uint8_t ws;
};

struct SirithPicoTdmConfig {
  uint32_t sampleRateHz = 16000;
  size_t frameSamples = 1024;
  int32_t sampleShiftBits = 16;

  uint8_t tdmSlots = 4;
  uint8_t slotBits = 32;

  // Output channel index -> TDM slot index (0-based).
  uint8_t outputChannelToSlot[4] = {0, 1, 3, 2};

  bool useSafeDriveStrength = true;
};

// RP2040/RP2350 master TDM receiver for 4-channel Sirith capture.
class SirithPicoTdmSource final : public IAudioSource {
 public:
  SirithPicoTdmSource(const SirithPicoTdmPins& pins, const SirithPicoTdmConfig& config);
  ~SirithPicoTdmSource() override;

  bool begin() override;
  uint32_t sampleRateHz() const override { return config_.sampleRateHz; }
  uint8_t channels() const override { return 4; }
  size_t frameSamples() const override { return config_.frameSamples; }
  bool readFrame(int16_t* interleavedOut, size_t samplesPerChannel) override;

 private:
  bool validateConfig() const;
  bool initPioStateMachine();
  void deinitPioStateMachine();
  int16_t toPcm16(int32_t raw) const;

  SirithPicoTdmPins pins_;
  SirithPicoTdmConfig config_;

  bool initialized_ = false;

  void* pio_ = nullptr;
  int sm_ = -1;
  uint32_t offset_ = 0;
  bool programInstalled_ = false;
};

}  // namespace mmpr
