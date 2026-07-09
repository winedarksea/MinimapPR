#pragma once

#include <cstddef>
#include <cstdint>

namespace mmpr {

enum class AuxSensorType : uint8_t {
  kAccelerometer = 0,
  kGyroscope = 1,
  kBarometer = 2,
  kTemperature = 3,
  kUnknown = 255,
};

struct AuxSensorStreamView {
  AuxSensorType sensorType = AuxSensorType::kUnknown;
  uint8_t valuesPerSample = 0;
  uint16_t sampleCount = 0;
  uint64_t firstSampleUtcNs = 0;
  uint32_t sampleIntervalUs = 0;
  const float* values = nullptr;
};

class IAuxSensorSource {
 public:
  virtual ~IAuxSensorSource() = default;

  virtual size_t snapshotStreamsForWindow(
      uint64_t windowStartUtcNs,
      uint64_t windowEndUtcNs,
      AuxSensorStreamView* outStreams,
      size_t maxStreams) = 0;
};

}  // namespace mmpr
