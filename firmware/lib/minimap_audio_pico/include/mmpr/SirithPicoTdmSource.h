#pragma once

#include <cstddef>
#include <cstdint>

#include "mmpr/IAudioSource.h"
#include "mmpr/PicoSerialAudioTiming.h"

namespace mmpr {

struct SirithPicoTdmPins {
  uint8_t dataIn;
  uint8_t bclk;
  uint8_t ws;
};

struct SirithPicoTdmConfig {
  uint32_t sampleRateHz = 16000;
  size_t frameSamples = 1024;
  uint8_t tdmSlots = 4;
  uint8_t slotBits = 32;
  uint8_t validBits = 24;
  PicoSerialSampleEdge sampleEdge = PicoSerialSampleEdge::kRising;
  int8_t captureBitOffset = 0;
  PicoSerialDataPinBias dataPinBias = PicoSerialDataPinBias::kDisabled;
  bool enableWordDiagnostics = false;
  uint32_t ringFrames = 16;

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
  AudioSourceType sourceType() const override { return AudioSourceType::kTdm; }
  bool readFrame(
      int16_t* interleavedOut,
      size_t samplesPerChannel,
      AudioCaptureTimestamp* captureTimestamp = nullptr) override;
  uint32_t availableFrames() const override;
  uint32_t ringFramesCapacity() const override { return ringFramesCapacity_; }
  uint32_t ringFramesHighWater() const override { return ringFramesHighWater_; }
  bool readFrameNonblocking(
      int16_t* interleavedOut,
      size_t samplesPerChannel,
      AudioCaptureTimestamp* captureTimestamp = nullptr) override;
  bool snapshotProducerState(
      AudioProducerSnapshot& producerSnapshot,
      bool callerAlreadyInIrqContext = false) const override;

 private:
  // Leave about 1 s of capture slack so a single slow publish does not
  // immediately overrun DMA while the node is waiting on the network stack.
  static constexpr uint32_t kMaxBufferedFrames = 32;

  bool validateConfig() const;
  bool initPioStateMachine();
  bool initDmaCapture();
  void deinitDmaCapture();
  void deinitPioStateMachine();
  void onDmaIrq();
  static void sDmaIrq();
  int16_t toPcm16(int32_t raw) const;

  SirithPicoTdmPins pins_;
  SirithPicoTdmConfig config_;

  bool initialized_ = false;
  uint64_t frameDurationUs_ = 0;
  size_t wordsPerFrame_ = 0;

  void* pio_ = nullptr;
  int sm_ = -1;
  uint32_t offset_ = 0;
  bool programInstalled_ = false;
  int dmaChannel_ = -1;
  uint32_t* dmaFrameWords_ = nullptr;
  uint32_t ringFramesCapacity_ = 16;
  uint64_t blockStartSampleIndex_[kMaxBufferedFrames] = {};
  uint64_t blockEndMonotonicUs_[kMaxBufferedFrames] = {};
  uint64_t completedBlockCountBySlot_[kMaxBufferedFrames] = {};
  uint64_t nextProducedStartSampleIndex_ = 0;
  uint64_t nextCompletedBlockCount_ = 0;
  volatile uint32_t dmaWriteFrameIndex_ = 0;
  volatile uint32_t dmaReadFrameIndex_ = 0;
  volatile uint32_t completedFrameCount_ = 0;
  volatile uint32_t ringFramesHighWater_ = 0;
  volatile uint32_t droppedFrameCount_ = 0;
  uint32_t reportedDroppedFrameCount_ = 0;
};

}  // namespace mmpr
