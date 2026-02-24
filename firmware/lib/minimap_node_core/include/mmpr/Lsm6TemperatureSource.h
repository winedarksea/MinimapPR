#pragma once

#include <Arduino.h>
#include <Wire.h>

#include <cstdint>

#include "mmpr/IEnvironmentalSource.h"

namespace mmpr {

struct Lsm6TemperatureSourceConfig {
  uint8_t primaryAddress7Bit = 0x6A;
  uint8_t secondaryAddress7Bit = 0x6B;
  uint32_t sampleIntervalMs = 2000;
};

class Lsm6TemperatureSource : public IEnvironmentalSource {
 public:
  Lsm6TemperatureSource(TwoWire& wire, const Lsm6TemperatureSourceConfig& config);

  bool begin() override;
  bool read(EnvironmentalSample& outSample) override;

  bool healthy() const { return healthy_; }
  bool enabled() const { return started_; }

 private:
  bool probeAddress(uint8_t address7Bit);
  bool readReg(uint8_t reg, uint8_t& value);
  bool writeReg(uint8_t reg, uint8_t value);
  bool readTemperatureRaw(int16_t& rawTemperature);
  void emitLast(EnvironmentalSample& outSample) const;

  TwoWire* wire_ = nullptr;
  Lsm6TemperatureSourceConfig config_ = {};

  bool started_ = false;
  bool healthy_ = false;
  bool hasReading_ = false;

  uint8_t address7Bit_ = 0;
  const char* sourceName_ = "lsm6";
  float lastTemperatureC_ = 0.0f;
  uint32_t lastSampleMs_ = 0;
};

}  // namespace mmpr
