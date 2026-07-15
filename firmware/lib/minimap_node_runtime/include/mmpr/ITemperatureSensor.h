#pragma once

namespace mmpr {

// Generic temperature capability that can be backed by any IMU/sensor chip.
class ITemperatureSensor {
 public:
  virtual ~ITemperatureSensor() = default;

  virtual bool begin() = 0;
  virtual bool readTemperatureC(float& outTemperatureC, const char*& outSourceName) = 0;
};

}  // namespace mmpr
