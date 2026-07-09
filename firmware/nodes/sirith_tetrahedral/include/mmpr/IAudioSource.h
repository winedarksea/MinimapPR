#pragma once

#include <cstddef>
#include <cstdint>

namespace mmpr {

enum class AudioSourceType : uint8_t {
  kTdm = 0,
  kI2sMono = 1,
  kPdmDirect = 2,
  kSynthetic = 3,
  kSilence = 4,
};

struct AudioProducerSnapshot {
  bool valid = false;
  uint64_t capturedMonotonicUs = 0;
  double samplePosition = 0.0;
  uint64_t completedBlockCount = 0;
  uint32_t dmaRingSlotIndex = 0;
  uint32_t wordsTransferredInActiveBlock = 0;
  uint32_t wordsRemainingInActiveBlock = 0;
};

struct AudioCaptureTimestamp {
  uint64_t startSampleIndex = 0;
  uint64_t endSampleIndex = 0;
  uint64_t blockStartMonotonicUs = 0;
  uint64_t blockEndMonotonicUs = 0;
  uint64_t completedBlockCount = 0;
  uint32_t dmaRingSlotIndex = 0;
  uint32_t droppedBlocksBeforeCapture = 0;
};

class IAudioSource {
 public:
  virtual ~IAudioSource() = default;

  virtual bool begin() = 0;
  virtual uint32_t sampleRateHz() const = 0;
  virtual uint8_t channels() const = 0;
  virtual size_t frameSamples() const = 0;
  virtual AudioSourceType sourceType() const = 0;

  virtual bool readFrame(
      int16_t* interleavedOut,
      size_t samplesPerChannel,
      AudioCaptureTimestamp* captureTimestamp = nullptr) = 0;

  virtual uint32_t availableFrames() const { return 0; }
  virtual uint32_t ringFramesCapacity() const { return 0; }
  virtual uint32_t ringFramesHighWater() const { return 0; }

  virtual bool readFrameNonblocking(
      int16_t* interleavedOut,
      size_t samplesPerChannel,
      AudioCaptureTimestamp* captureTimestamp = nullptr) {
    if (availableFrames() == 0) {
      return false;
    }
    return readFrame(interleavedOut, samplesPerChannel, captureTimestamp);
  }

  virtual bool snapshotProducerState(
      AudioProducerSnapshot& producerSnapshot,
      bool callerAlreadyInIrqContext = false) const {
    (void)callerAlreadyInIrqContext;
    (void)producerSnapshot;
    return false;
  }
};

}  // namespace mmpr
