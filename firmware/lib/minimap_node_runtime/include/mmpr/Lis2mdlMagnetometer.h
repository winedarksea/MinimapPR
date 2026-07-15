#pragma once

#include <cstdint>

#include "mmpr/I2cBus.h"
#include "mmpr/IMagnetometer.h"

namespace mmpr {

/// Driver configuration for the LIS2MDLTR 3-axis magnetometer.
struct Lis2mdlMagConfig {
  uint8_t i2cAddress7Bit = 0x1E;

  /// Output data rate: 0 = 10 Hz, 1 = 20 Hz, 2 = 50 Hz, 3 = 100 Hz (LP only).
  /// 50 Hz (2) is recommended for Kalman-filtered heading at ≥ 5 Hz poll rate.
  uint8_t odrBits = 2;

  /// Enable built-in temperature compensation (recommended).
  bool enableTempComp = true;

  /// Enable hardware low-pass filter (bandwidth = ODR / 4).
  /// Significantly reduces wideband noise before software filtering.
  bool enableLpf = true;

  /// Enable built-in offset cancellation (SET/RESET pulse).
  /// Removes time-varying sensor offset drift.
  bool enableOffsetCancel = true;

  /// Hard-iron calibration offsets (subtracted from raw readings, in LSB).
  /// Compensate for static magnetic bias from the PCB and nearby components.
  /// Determined during calibration (e.g. figure-eight rotation, min/max averaging).
  float hardIronOffsetX = 0.0f;
  float hardIronOffsetY = 0.0f;
  float hardIronOffsetZ = 0.0f;
};

/// LIS2MDLTR magnetometer driver implementing IMagnetometer.
///
/// Sensitivity is fixed at 1.5 mGauss / LSB.  readField() returns values
/// in LSB (after hard-iron subtraction).  The orientation estimator's
/// field-magnitude threshold must be set in matching units.
class Lis2mdlMagnetometer final : public IMagnetometer {
 public:
  Lis2mdlMagnetometer(I2cBus& bus, const Lis2mdlMagConfig& config);

  bool begin() override;
  bool readField(float& x, float& y, float& z) override;
  bool healthy() const override { return healthy_; }
  const char* name() const override { return "lis2mdltr"; }

 private:
  bool readReg(uint8_t reg, uint8_t& value);
  bool writeReg(uint8_t reg, uint8_t value);

  I2cBus* bus_ = nullptr;
  Lis2mdlMagConfig config_ = {};
  bool started_ = false;
  bool healthy_ = false;
};

}  // namespace mmpr
