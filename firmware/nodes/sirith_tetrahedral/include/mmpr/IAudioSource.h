#pragma once

#include <cstddef>
#include <cstdint>

namespace mmpr {

struct AudioCaptureTimestamp {
  uint64_t frameStartMonotonicUs = 0;
  uint64_t frameEndMonotonicUs = 0;
  uint32_t droppedFramesBeforeCapture = 0;
};

class IAudioSource {
 public:
  virtual ~IAudioSource() = default;

  virtual bool begin() = 0;
  virtual uint32_t sampleRateHz() const = 0;
  virtual uint8_t channels() const = 0;
  virtual size_t frameSamples() const = 0;

  virtual bool readFrame(
      int16_t* interleavedOut,
      size_t samplesPerChannel,
      AudioCaptureTimestamp* captureTimestamp = nullptr) = 0;
};

}  // namespace mmpr
