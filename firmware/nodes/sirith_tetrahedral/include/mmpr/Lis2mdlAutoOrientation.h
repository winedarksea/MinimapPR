#pragma once

#include <cstdint>

#include "mmpr/I2cBus.h"

namespace mmpr {

struct Lis2mdlAutoOrientationConfig {
  uint8_t i2cAddress7Bit = 0x1E;
  uint8_t outputDataRateBits = 0;  // 0:10Hz, 1:20Hz, 2:50Hz, 3:100Hz

  bool enableTempComp = true;
  bool lowPowerMode = false;

  uint32_t sampleIntervalMs = 500;
  float smoothingAlpha = 0.03f;
  float headingOffsetDeg = 0.0f;
  float minHorizontalFieldLsb = 50.0f;

  uint16_t stableSamplesRequired = 18;
};

class Lis2mdlAutoOrientation {
 public:
  bool begin(I2cBus& bus, const Lis2mdlAutoOrientationConfig& config, uint8_t initialRotationSteps);
  bool poll(uint8_t* changedRotationSteps = nullptr);

  bool healthy() const { return healthy_; }
  bool enabled() const { return started_; }
  uint8_t rotationSteps() const { return rotationSteps_; }
  float headingDeg() const { return headingDeg_; }

 private:
  bool readReg(uint8_t reg, uint8_t& value);
  bool writeReg(uint8_t reg, uint8_t value);
  bool readMagRaw(int16_t& x, int16_t& y, int16_t& z);

  static float wrap360(float deg);
  uint8_t headingToRotationSteps(float headingDeg) const;

  I2cBus* bus_ = nullptr;
  Lis2mdlAutoOrientationConfig config_ = {};

  bool started_ = false;
  bool healthy_ = false;

  uint8_t rotationSteps_ = 0;
  uint8_t candidateRotationSteps_ = 0;
  uint16_t stableSampleCount_ = 0;

  uint32_t lastSampleMs_ = 0;
  uint32_t sampleCount_ = 0;

  float filtX_ = 0.0f;
  float filtY_ = 0.0f;
  float headingDeg_ = 0.0f;
};

}  // namespace mmpr
