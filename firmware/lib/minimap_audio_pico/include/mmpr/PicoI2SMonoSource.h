#pragma once

#include <cstddef>
#include <cstdint>

#include "mmpr/IAudioSource.h"
#include "mmpr/PicoSerialAudioTiming.h"

namespace mmpr {

enum class PicoI2SChannelSide : uint8_t {
  kLeft = 0,
  kRight = 1,
};

struct PicoI2SMonoPins {
  uint8_t dataIn;
  uint8_t bclk;
  uint8_t ws;
};

struct PicoI2SMonoConfig {
  uint32_t sampleRateHz = 16000;
  size_t frameSamples = 1024;
  uint8_t slotBits = 32;
  uint8_t validBits = 24;
  PicoI2SChannelSide channelSide = PicoI2SChannelSide::kLeft;
  PicoSerialSampleEdge sampleEdge = PicoSerialSampleEdge::kRising;
  int8_t captureBitOffset = 0;
  PicoSerialDataPinBias dataPinBias = PicoSerialDataPinBias::kDisabled;
  bool enableWordDiagnostics = false;
  bool useSafeDriveStrength = true;
};

// RP2040/RP2350 master I2S receiver that extracts one mono channel from a
// stereo 32-bit-slot I2S stream.
class PicoI2SMonoSource final : public IAudioSource {
 public:
  PicoI2SMonoSource(const PicoI2SMonoPins& pins, const PicoI2SMonoConfig& config);
  ~PicoI2SMonoSource() override;

  bool begin() override;
  uint32_t sampleRateHz() const override { return config_.sampleRateHz; }
  uint8_t channels() const override { return 1; }
  size_t frameSamples() const override { return config_.frameSamples; }
  bool readFrame(
      int16_t* interleavedOut,
      size_t samplesPerChannel,
      AudioCaptureTimestamp* captureTimestamp = nullptr) override;

 private:
  // Keep >1 s of capture slack in mono mode so short Wi-Fi stalls do not
  // immediately overflow the DMA ring while the main loop is publishing.
  static constexpr uint32_t kBufferedFrames = 16;

  bool validateConfig() const;
  bool initPioStateMachine();
  bool initDmaCapture();
  void deinitDmaCapture();
  void deinitPioStateMachine();
  void onDmaIrq();
  static void sDmaIrq();
  int16_t toPcm16(int32_t raw) const;

  PicoI2SMonoPins pins_;
  PicoI2SMonoConfig config_;

  bool initialized_ = false;
  uint64_t frameDurationUs_ = 0;
  size_t wordsPerFrame_ = 0;

  void* pio_ = nullptr;
  int sm_ = -1;
  uint32_t offset_ = 0;
  bool programInstalled_ = false;
  int dmaChannel_ = -1;
  uint32_t* dmaFrameWords_ = nullptr;
  uint64_t frameEndMonotonicUs_[kBufferedFrames] = {};
  volatile uint32_t dmaWriteFrameIndex_ = 0;
  volatile uint32_t dmaReadFrameIndex_ = 0;
  volatile uint32_t completedFrameCount_ = 0;
  volatile uint32_t droppedFrameCount_ = 0;
  uint32_t reportedDroppedFrameCount_ = 0;
};

}  // namespace mmpr
