#pragma once

#include <cstdint>

#include "mmpr/IMagnetometer.h"

namespace mmpr {

/// Operating mode for auto-orientation.
enum class OrientationMode : uint8_t {
  /// Use magnetometer with Kalman-filtered heading to derive rotation step.
  kAuto = 0,

  /// Ignore magnetometer entirely; use the fixed rotation step from config.
  /// Use this when the magnetic environment is permanently hostile (strong
  /// hard-iron from nearby equipment, indoor steel structures, etc.).
  kManualFixed = 1,
};

/// Configuration for the magnetometer-based auto-orientation estimator.
///
/// The estimator uses a scalar Kalman filter on heading angle (circular,
/// in degrees) to heavily smooth transient magnetic noise while remaining
/// responsive to genuine orientation changes.  The rotation step only
/// updates after the heading stays in a new 120° sector for at least
/// `stableSamplesRequired` consecutive polls (hysteresis).
///
/// Tuning guide (steady-state Kalman gain ≈ √(Q/R)):
///   Q = 0.001, R = 4.0  →  gain ≈ 0.016, τ ≈ 12 s   (default, very stationary)
///   Q = 0.01,  R = 4.0  →  gain ≈ 0.050, τ ≈  4 s   (faster convergence)
///   Q = 0.0001,R = 4.0  →  gain ≈ 0.005, τ ≈ 40 s   (extremely heavy damping)
struct MagAutoOrientationConfig {
  OrientationMode mode = OrientationMode::kAuto;

  /// Minimum interval between magnetometer reads (ms).
  /// 200 ms (5 Hz) is a good balance for a stationary node at 50 Hz ODR.
  uint32_t sampleIntervalMs = 200;

  /// Heading offset (degrees).  Subtracted from the raw computed heading
  /// before mapping to a rotation step.  Used to align the compass frame
  /// with the physical installation orientation.
  ///
  /// For this PCB the base-mic bearings from the array centroid in the
  /// LIS2MDLTR sensor frame are approximately:
  ///   MK2 ≈  0°,  MK1 ≈ 120°,  MK3 ≈ 240°.
  float headingOffsetDeg = 0.0f;

  /// Minimum horizontal field magnitude (sensor-native units, e.g. LSB).
  /// Readings weaker than this are rejected as unreliable (tilt, soft-iron
  /// distortion, or very weak ambient field).
  float minFieldMagnitude = 50.0f;

  /// Number of consecutive polls in the same rotation sector before the
  /// step actually changes.  Prevents flicker at sector boundaries.
  uint16_t stableSamplesRequired = 18;

  // ---- Kalman filter tuning (heading in degrees) ----

  /// Process noise variance (deg² per poll step).
  /// Controls how quickly past measurements are forgotten.
  /// Very low values (0.001) suit a fixed-location node.
  float kalmanQ = 0.001f;

  /// Measurement noise variance (deg²).
  /// Represents expected single-sample heading noise from the magnetometer
  /// plus environmental interference.  Higher = more filtering.
  float kalmanR = 4.0f;

  /// Initial covariance (deg²).  Set large so the filter converges from
  /// the first measurement rather than from zero.
  float kalmanInitialP = 400.0f;
};

/// Magnetometer-based auto-orientation with circular Kalman filtering.
///
/// Estimates the compass heading of the PCB and maps it to one of three
/// 120° rotation steps for a tetrahedral base-plane mic array.  The
/// magnetometer implementation is injected via the IMagnetometer interface,
/// so the driver can be swapped without modifying this class.
///
/// Future: a setAccelGyro() method could fuse accelerometer / gyroscope
/// data for tilt-compensated heading (9-axis AHRS).  The interface is
/// designed to accommodate this without structural changes — add an
/// optional IImu* parameter alongside IMagnetometer.
class MagAutoOrientation {
 public:
  /// Initialise the estimator.
  /// @param mag               Magnetometer driver (must outlive this object).
  /// @param config            Filter and hysteresis parameters.
  /// @param initialRotation   Fallback rotation step (0–2) used until enough
  ///                          stable readings have been collected.
  /// @return true if the magnetometer is healthy and ready.
  bool begin(IMagnetometer& mag,
             const MagAutoOrientationConfig& config,
             uint8_t initialRotation);

  /// Poll the magnetometer and update the Kalman estimate.
  /// @param changedRotation  If non-null and the rotation step changed,
  ///                         receives the new step value (0, 1, or 2).
  /// @return true if and only if the rotation step changed on this call.
  bool poll(uint8_t* changedRotation = nullptr);

  bool healthy() const { return healthy_; }
  bool enabled() const { return started_; }

  uint8_t rotationSteps() const { return rotationSteps_; }

  /// Most recent Kalman-filtered heading (degrees, 0–360).
  float headingDeg() const { return headingDeg_; }

  /// Heading after applying the installation offset, suitable for rotating
  /// sensor-frame microphone coordinates into the local world XY frame.
  float worldHeadingDeg() const;

  bool hasHeadingEstimate() const { return hasEstimate_; }
  uint32_t estimateRevision() const { return estimateRevision_; }

  /// Current Kalman gain (useful for diagnostics / tuning).
  float kalmanGain() const { return lastGain_; }

  /// Current covariance P (degrees²).
  float covarianceP() const { return covP_; }

 private:
  static float wrap360(float deg);
  static float wrapPM180(float deg);
  uint8_t headingToRotationSteps(float headingDeg) const;

  IMagnetometer* mag_ = nullptr;
  MagAutoOrientationConfig config_ = {};

  bool started_ = false;
  bool healthy_ = false;

  // Kalman state.
  float headingDeg_ = 0.0f;   // filtered heading (degrees, 0–360)
  float covP_ = 400.0f;       // covariance
  float lastGain_ = 0.0f;
  bool hasEstimate_ = false;
  uint32_t estimateRevision_ = 0;

  // Rotation step hysteresis.
  uint8_t rotationSteps_ = 0;
  uint8_t candidateRotation_ = 0;
  uint16_t stableSampleCount_ = 0;

  uint32_t lastSampleMs_ = 0;
};

}  // namespace mmpr
