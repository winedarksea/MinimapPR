#include "mmpr/MagAutoOrientation.h"

#include <cmath>
#include <cstdio>

#include "pico/time.h"

namespace mmpr {
namespace {

constexpr float kRadToDeg = 57.29577951308232f;

uint32_t millis32() {
  return to_ms_since_boot(get_absolute_time());
}

}  // namespace

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

bool MagAutoOrientation::begin(
    IMagnetometer& mag,
    const MagAutoOrientationConfig& config,
    uint8_t initialRotation) {
  mag_ = &mag;
  config_ = config;

  // Clamp / fixup config.
  if (config_.sampleIntervalMs == 0) {
    config_.sampleIntervalMs = 200;
  }
  if (config_.stableSamplesRequired == 0) {
    config_.stableSamplesRequired = 18;
  }
  if (!(config_.kalmanQ > 0.0f)) {
    config_.kalmanQ = 0.001f;
  }
  if (!(config_.kalmanR > 0.0f)) {
    config_.kalmanR = 4.0f;
  }
  if (!(config_.kalmanInitialP > 0.0f)) {
    config_.kalmanInitialP = 400.0f;
  }

  rotationSteps_ = static_cast<uint8_t>(initialRotation % 3u);
  candidateRotation_ = rotationSteps_;
  stableSampleCount_ = 0;

  // Reset Kalman state.
  hasEstimate_ = false;
  estimateRevision_ = 0;
  headingDeg_ = 0.0f;
  covP_ = config_.kalmanInitialP;
  lastGain_ = 0.0f;

  // In manual-fixed mode we skip the magnetometer entirely.
  if (config_.mode == OrientationMode::kManualFixed) {
    started_ = true;
    healthy_ = true;
    std::printf("[mag-orient] manual-fixed mode, rotation step = %u\n",
                static_cast<unsigned>(rotationSteps_));
    return true;
  }

  // Auto mode — probe magnetometer.
  if (!mag_->begin()) {
    healthy_ = false;
    started_ = false;
    return false;
  }

  healthy_ = true;
  started_ = true;
  lastSampleMs_ = millis32();
  return true;
}

// ---------------------------------------------------------------------------
// Poll — Kalman-filtered heading with rotation-step hysteresis
// ---------------------------------------------------------------------------

bool MagAutoOrientation::poll(uint8_t* changedRotation) {
  if (!started_ || !healthy_) {
    return false;
  }

  // Manual mode never changes the step.
  if (config_.mode == OrientationMode::kManualFixed) {
    return false;
  }

  // Rate-limit reads.
  const uint32_t nowMs = millis32();
  if ((nowMs - lastSampleMs_) < config_.sampleIntervalMs) {
    return false;
  }
  lastSampleMs_ = nowMs;

  // ---- Read magnetometer ----
  float fx = 0.0f, fy = 0.0f, fz = 0.0f;
  (void)fz;
  if (!mag_->readField(fx, fy, fz)) {
    healthy_ = false;
    return false;
  }

  const float mag = std::sqrt((fx * fx) + (fy * fy));
  if (!(mag >= config_.minFieldMagnitude)) {
    // Field too weak — skip this sample (don't touch Kalman state).
    return false;
  }

  // ---- Raw heading from horizontal components ----
  const float rawHeadingDeg = wrap360(std::atan2(fy, fx) * kRadToDeg);

  // ---- Scalar circular Kalman filter ----
  if (!hasEstimate_) {
    // Seed the filter with the first valid measurement.
    headingDeg_ = rawHeadingDeg;
    covP_ = config_.kalmanInitialP;
    hasEstimate_ = true;
  } else {
    // Predict (stationary model: heading unchanged).
    const float pPred = covP_ + config_.kalmanQ;

    // Innovation (circular difference, ±180°).
    const float innovation = wrapPM180(rawHeadingDeg - headingDeg_);

    // Update.
    const float s = pPred + config_.kalmanR;
    const float k = pPred / s;
    headingDeg_ = wrap360(headingDeg_ + k * innovation);
    covP_ = (1.0f - k) * pPred;
    lastGain_ = k;
  }
  ++estimateRevision_;

  // ---- Map heading to rotation step with hysteresis ----
  const uint8_t candidate = headingToRotationSteps(headingDeg_);

  if (candidate != candidateRotation_) {
    candidateRotation_ = candidate;
    stableSampleCount_ = 1;
    return false;
  }

  if (stableSampleCount_ < 0xFFFFu) {
    ++stableSampleCount_;
  }

  if (candidateRotation_ != rotationSteps_ &&
      stableSampleCount_ >= config_.stableSamplesRequired) {
    rotationSteps_ = candidateRotation_;
    if (changedRotation != nullptr) {
      *changedRotation = rotationSteps_;
    }
    stableSampleCount_ = 0;
    return true;
  }

  return false;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

float MagAutoOrientation::wrap360(float deg) {
  float out = std::fmod(deg, 360.0f);
  if (out < 0.0f) {
    out += 360.0f;
  }
  return out;
}

float MagAutoOrientation::wrapPM180(float deg) {
  float out = std::fmod(deg + 180.0f, 360.0f);
  if (out < 0.0f) {
    out += 360.0f;
  }
  return out - 180.0f;
}

float MagAutoOrientation::worldHeadingDeg() const {
  return wrap360(headingDeg_ - config_.headingOffsetDeg);
}

uint8_t MagAutoOrientation::headingToRotationSteps(float heading) const {
  const float adjusted = wrap360(heading - config_.headingOffsetDeg);
  const int sector = static_cast<int>(std::lround(adjusted / 120.0f));
  const int mod = sector % 3;
  return static_cast<uint8_t>((mod < 0) ? (mod + 3) : mod);
}

}  // namespace mmpr
