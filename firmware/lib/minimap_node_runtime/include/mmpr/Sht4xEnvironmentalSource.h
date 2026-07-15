#pragma once

#include <cstdint>

#include "mmpr/I2cBus.h"
#include "mmpr/IEnvironmentalSource.h"

namespace mmpr {

struct Sht4xEnvironmentalSourceConfig {
  uint8_t address7Bit = 0x44;
  uint32_t sampleIntervalMs = 2000;
};

class Sht4xEnvironmentalSource : public IEnvironmentalSource {
 public:
  Sht4xEnvironmentalSource(I2cBus& bus, const Sht4xEnvironmentalSourceConfig& config);

  bool begin() override;
  bool read(EnvironmentalSample& outSample) override;

  bool healthy() const { return healthy_; }
  bool enabled() const { return started_; }

 private:
  bool readMeasurement(float& outTemperatureC, float& outHumidityFraction);
  static bool verifyCrc(const uint8_t* data, uint8_t expectedCrc);

  I2cBus* bus_ = nullptr;
  Sht4xEnvironmentalSourceConfig config_ = {};

  bool started_ = false;
  bool healthy_ = false;
  bool hasReading_ = false;
  float lastTemperatureC_ = 0.0f;
  float lastHumidityFraction_ = 0.0f;
  uint32_t lastSampleMs_ = 0;
};

}  // namespace mmpr
