#pragma once

#include <cstdint>

#include "mmpr/I2cBus.h"
#include "mmpr/ITemperatureSensor.h"

namespace mmpr {

struct Lsm6TemperatureSensorConfig {
  uint8_t primaryAddress7Bit = 0x6A;
  uint8_t secondaryAddress7Bit = 0x6B;
  uint32_t sampleIntervalMs = 2000;
};

class Lsm6TemperatureSensor : public ITemperatureSensor {
 public:
  Lsm6TemperatureSensor(I2cBus& bus, const Lsm6TemperatureSensorConfig& config);

  bool begin() override;
  bool readTemperatureC(float& outTemperatureC, const char*& outSourceName) override;

  bool healthy() const { return healthy_; }
  bool enabled() const { return started_; }

 private:
  bool probeAddress(uint8_t address7Bit);
  bool readReg(uint8_t reg, uint8_t& value);
  bool writeReg(uint8_t reg, uint8_t value);
  bool readTemperatureRaw(int16_t& rawTemperature);

  I2cBus* bus_ = nullptr;
  Lsm6TemperatureSensorConfig config_ = {};

  bool started_ = false;
  bool healthy_ = false;
  bool hasReading_ = false;

  uint8_t address7Bit_ = 0;
  const char* sourceName_ = "lsm6";
  float lastTemperatureC_ = 0.0f;
  uint32_t lastSampleMs_ = 0;
};

}  // namespace mmpr
