#include "mmpr/Lis2mdlAutoOrientation.h"

#include <cmath>

#include "pico/time.h"

namespace mmpr {
namespace {

constexpr uint8_t kRegWhoAmI = 0x4F;
constexpr uint8_t kRegCfgA = 0x60;
constexpr uint8_t kRegCfgC = 0x62;
constexpr uint8_t kRegOutXL = 0x68;

constexpr uint8_t kWhoAmIValue = 0x40;
constexpr float kRadToDeg = 57.29577951308232f;

uint32_t millis32() {
  return to_ms_since_boot(get_absolute_time());
}

}  // namespace

bool Lis2mdlAutoOrientation::begin(I2cBus& bus, const Lis2mdlAutoOrientationConfig& config, uint8_t initialRotationSteps) {
  bus_ = &bus;
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

  rotationSteps_ = static_cast<uint8_t>(initialRotationSteps % 3u);
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
  lastSampleMs_ = millis32();
  return true;
}

bool Lis2mdlAutoOrientation::poll(uint8_t* changedRotationSteps) {
  if (!started_ || !healthy_) {
    return false;
  }

  const uint32_t nowMs = millis32();
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
  const float mag = std::sqrt((fx * fx) + (fy * fy));
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

    const float n = std::sqrt((filtX_ * filtX_) + (filtY_ * filtY_));
    if (n > 1e-6f) {
      filtX_ /= n;
      filtY_ /= n;
    }
  }

  ++sampleCount_;

  headingDeg_ = wrap360(std::atan2(filtY_, filtX_) * kRadToDeg);
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
  if (bus_ == nullptr) {
    return false;
  }

  uint8_t out = 0;
  if (!bus_->readReg(config_.i2cAddress7Bit, reg, &out, 1)) {
    return false;
  }
  value = out;
  return true;
}

bool Lis2mdlAutoOrientation::writeReg(uint8_t reg, uint8_t value) {
  if (bus_ == nullptr) {
    return false;
  }
  return bus_->writeReg(config_.i2cAddress7Bit, reg, value);
}

bool Lis2mdlAutoOrientation::readMagRaw(int16_t& x, int16_t& y, int16_t& z) {
  if (bus_ == nullptr) {
    return false;
  }

  uint8_t raw[6] = {0};
  if (!bus_->readReg(config_.i2cAddress7Bit, kRegOutXL, raw, sizeof(raw))) {
    return false;
  }

  x = static_cast<int16_t>((static_cast<uint16_t>(raw[1]) << 8u) | raw[0]);
  y = static_cast<int16_t>((static_cast<uint16_t>(raw[3]) << 8u) | raw[2]);
  z = static_cast<int16_t>((static_cast<uint16_t>(raw[5]) << 8u) | raw[4]);
  return true;
}

float Lis2mdlAutoOrientation::wrap360(float deg) {
  float out = std::fmod(deg, 360.0f);
  if (out < 0.0f) {
    out += 360.0f;
  }
  return out;
}

uint8_t Lis2mdlAutoOrientation::headingToRotationSteps(float headingDeg) const {
  const float adjusted = wrap360(headingDeg - config_.headingOffsetDeg);
  const int sector = static_cast<int>(std::lround(adjusted / 120.0f));
  const int mod = sector % 3;
  return static_cast<uint8_t>((mod < 0) ? (mod + 3) : mod);
}

}  // namespace mmpr
