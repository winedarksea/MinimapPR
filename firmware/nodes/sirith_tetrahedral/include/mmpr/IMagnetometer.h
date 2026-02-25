#pragma once

namespace mmpr {

/// Abstract 3-axis magnetometer interface.
///
/// Implementations wrap a specific sensor IC (LIS2MDLTR, MMC5603, etc.)
/// and handle all register-level I/O.  The orientation estimator references
/// only this interface, so swapping to a different part requires only a new
/// driver — no changes to heading / rotation logic.
///
/// readField() returns values in sensor-native units (LSB, milliGauss, µT,
/// etc.) — the consumer must set its field-magnitude threshold accordingly.
///
/// Future: an IImu interface with readAccelGyro() would sit beside this for
/// 9-axis tilt-compensated heading.  The orientation class already reserves
/// a slot for it (currently unused).
class IMagnetometer {
 public:
  virtual ~IMagnetometer() = default;

  /// Initialise the sensor.  Returns true if the device was probed and
  /// configured successfully.
  virtual bool begin() = 0;

  /// Read the most recent 3-axis field measurement.
  /// @param x, y, z  output components (sensor-native units, after any
  ///                  hard-iron correction applied by the driver).
  /// @return true on successful read.
  virtual bool readField(float& x, float& y, float& z) = 0;

  /// True after a successful begin() and no subsequent fatal error.
  virtual bool healthy() const = 0;

  /// Human-readable driver / part name (e.g. "lis2mdltr").
  virtual const char* name() const = 0;
};

}  // namespace mmpr
