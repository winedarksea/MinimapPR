#pragma once

#include <cstddef>
#include <cstdint>

namespace mmpr {

class IAudioSource {
 public:
  virtual ~IAudioSource() = default;

  virtual bool begin() = 0;
  virtual uint32_t sampleRateHz() const = 0;
  virtual uint8_t channels() const = 0;
  virtual size_t frameSamples() const = 0;

  virtual bool readFrame(int16_t* interleavedOut, size_t samplesPerChannel) = 0;
};

}  // namespace mmpr
