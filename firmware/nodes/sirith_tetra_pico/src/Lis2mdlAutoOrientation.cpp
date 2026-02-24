#include "mmpr/Lis2mdlAutoOrientation.h"

#include <math.h>

namespace mmpr {
namespace {

constexpr uint8_t kRegWhoAmI = 0x4F;
constexpr uint8_t kRegCfgA = 0x60;
constexpr uint8_t kRegCfgC = 0x62;
constexpr uint8_t kRegOutXL = 0x68;

constexpr uint8_t kWhoAmIValue = 0x40;
constexpr float kRadToDeg = 57.29577951308232f;

}  // namespace

bool Lis2mdlAutoOrientation::begin(TwoWire& wire, const Lis2mdlAutoOrientationConfig& config, uint8_t initialRotationSteps) {
  wire_ = &wire;
  config_ = config;

  if (config_.sampleIntervalMs == 0) {
    config_.sampleIntervalMs = 500;
  }
  if (!(config_.smoothingAlpha > 0.0f && config_.smoothingAlpha <= 1.0f)) {
    config_.smoothingAlpha = 0.03f;
  }
  if (config_.stableSamplesRequired == 0) {
    config_.stableSamplesRequired = 18;
  }

  rotationSteps_ = static_cast<uint8_t>(initialRotationSteps % 3);
  candidateRotationSteps_ = rotationSteps_;

  uint8_t whoAmI = 0;
  if (!readReg(kRegWhoAmI, whoAmI) || whoAmI != kWhoAmIValue) {
    healthy_ = false;
    started_ = false;
    return false;
  }

  uint8_t cfgA = 0;
  cfgA |= static_cast<uint8_t>((config_.outputDataRateBits & 0x03u) << 2u);
  if (config_.lowPowerMode) {
    cfgA |= 0x10u;
  }
  if (config_.enableTempComp) {
    cfgA |= 0x80u;
  }

  // BDU enabled, all other fields default.
  const uint8_t cfgC = 0x10u;

  if (!writeReg(kRegCfgA, cfgA) || !writeReg(kRegCfgC, cfgC)) {
    healthy_ = false;
    started_ = false;
    return false;
  }

  healthy_ = true;
  started_ = true;
  sampleCount_ = 0;
  stableSampleCount_ = 0;
  lastSampleMs_ = millis();

  return true;
}

bool Lis2mdlAutoOrientation::poll(uint8_t* changedRotationSteps) {
  if (!started_ || !healthy_) {
    return false;
  }

  const uint32_t nowMs = millis();
  if ((nowMs - lastSampleMs_) < config_.sampleIntervalMs) {
    return false;
  }
  lastSampleMs_ = nowMs;

  int16_t x = 0;
  int16_t y = 0;
  int16_t z = 0;
  (void)z;

  if (!readMagRaw(x, y, z)) {
    healthy_ = false;
    return false;
  }

  const float fx = static_cast<float>(x);
  const float fy = static_cast<float>(y);
  const float mag = sqrtf((fx * fx) + (fy * fy));
  if (!(mag >= config_.minHorizontalFieldLsb)) {
    return false;
  }

  const float nx = fx / mag;
  const float ny = fy / mag;

  if (sampleCount_ == 0) {
    filtX_ = nx;
    filtY_ = ny;
  } else {
    const float alpha = config_.smoothingAlpha;
    filtX_ = ((1.0f - alpha) * filtX_) + (alpha * nx);
    filtY_ = ((1.0f - alpha) * filtY_) + (alpha * ny);

    const float n = sqrtf((filtX_ * filtX_) + (filtY_ * filtY_));
    if (n > 1e-6f) {
      filtX_ /= n;
      filtY_ /= n;
    }
  }

  ++sampleCount_;

  headingDeg_ = wrap360(atan2f(filtY_, filtX_) * kRadToDeg);
  const uint8_t candidate = headingToRotationSteps(headingDeg_);

  if (candidate != candidateRotationSteps_) {
    candidateRotationSteps_ = candidate;
    stableSampleCount_ = 1;
    return false;
  }

  if (stableSampleCount_ < 0xFFFFu) {
    ++stableSampleCount_;
  }

  if (candidateRotationSteps_ != rotationSteps_ && stableSampleCount_ >= config_.stableSamplesRequired) {
    rotationSteps_ = candidateRotationSteps_;
    if (changedRotationSteps != nullptr) {
      *changedRotationSteps = rotationSteps_;
    }
    stableSampleCount_ = 0;
    return true;
  }

  return false;
}

bool Lis2mdlAutoOrientation::readReg(uint8_t reg, uint8_t& value) {
  if (wire_ == nullptr) {
    return false;
  }

  wire_->beginTransmission(config_.i2cAddress7Bit);
  wire_->write(reg);
  if (wire_->endTransmission(false) != 0) {
    return false;
  }

  if (wire_->requestFrom(static_cast<int>(config_.i2cAddress7Bit), 1) != 1) {
    return false;
  }

  value = static_cast<uint8_t>(wire_->read());
  return true;
}

bool Lis2mdlAutoOrientation::writeReg(uint8_t reg, uint8_t value) {
  if (wire_ == nullptr) {
    return false;
  }

  wire_->beginTransmission(config_.i2cAddress7Bit);
  wire_->write(reg);
  wire_->write(value);
  return wire_->endTransmission(true) == 0;
}

bool Lis2mdlAutoOrientation::readMagRaw(int16_t& x, int16_t& y, int16_t& z) {
  if (wire_ == nullptr) {
    return false;
  }

  wire_->beginTransmission(config_.i2cAddress7Bit);
  wire_->write(kRegOutXL);
  if (wire_->endTransmission(false) != 0) {
    return false;
  }

  constexpr uint8_t kReadLen = 6;
  if (wire_->requestFrom(static_cast<int>(config_.i2cAddress7Bit), static_cast<int>(kReadLen)) != kReadLen) {
    return false;
  }

  const uint8_t xL = static_cast<uint8_t>(wire_->read());
  const uint8_t xH = static_cast<uint8_t>(wire_->read());
  const uint8_t yL = static_cast<uint8_t>(wire_->read());
  const uint8_t yH = static_cast<uint8_t>(wire_->read());
  const uint8_t zL = static_cast<uint8_t>(wire_->read());
  const uint8_t zH = static_cast<uint8_t>(wire_->read());

  x = static_cast<int16_t>((static_cast<uint16_t>(xH) << 8u) | xL);
  y = static_cast<int16_t>((static_cast<uint16_t>(yH) << 8u) | yL);
  z = static_cast<int16_t>((static_cast<uint16_t>(zH) << 8u) | zL);
  return true;
}

float Lis2mdlAutoOrientation::wrap360(float deg) {
  float out = fmodf(deg, 360.0f);
  if (out < 0.0f) {
    out += 360.0f;
  }
  return out;
}

uint8_t Lis2mdlAutoOrientation::headingToRotationSteps(float headingDeg) const {
  const float adjusted = wrap360(headingDeg - config_.headingOffsetDeg);
  const int sector = static_cast<int>(lroundf(adjusted / 120.0f));
  const int mod = sector % 3;
  return static_cast<uint8_t>((mod < 0) ? (mod + 3) : mod);
}

}  // namespace mmpr
