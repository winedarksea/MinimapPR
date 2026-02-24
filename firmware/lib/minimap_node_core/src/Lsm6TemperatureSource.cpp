#include "mmpr/Lsm6TemperatureSource.h"

#include <math.h>

namespace mmpr {
namespace {

constexpr uint8_t kRegWhoAmI = 0x0F;
constexpr uint8_t kRegCtrl1Xl = 0x10;
constexpr uint8_t kRegOutTempL = 0x20;

constexpr uint8_t kWhoAmILsm6dsox = 0x6C;
constexpr uint8_t kWhoAmILsm6dsv16x = 0x70;

constexpr float kTemperatureScaleLsbPerC = 256.0f;
constexpr float kTemperatureOffsetC = 25.0f;

}  // namespace

Lsm6TemperatureSource::Lsm6TemperatureSource(TwoWire& wire, const Lsm6TemperatureSourceConfig& config)
    : wire_(&wire), config_(config) {}

bool Lsm6TemperatureSource::begin() {
  started_ = false;
  healthy_ = false;
  hasReading_ = false;
  address7Bit_ = 0;
  sourceName_ = "lsm6";

  if (wire_ == nullptr) {
    return false;
  }
  if (config_.sampleIntervalMs == 0) {
    config_.sampleIntervalMs = 2000;
  }

  if (probeAddress(config_.primaryAddress7Bit)) {
    started_ = true;
    healthy_ = true;
    return true;
  }

  if (config_.secondaryAddress7Bit != config_.primaryAddress7Bit &&
      probeAddress(config_.secondaryAddress7Bit)) {
    started_ = true;
    healthy_ = true;
    return true;
  }

  return false;
}

bool Lsm6TemperatureSource::read(EnvironmentalSample& outSample) {
  if (!started_) {
    return false;
  }

  const uint32_t nowMs = millis();
  if (hasReading_ && (nowMs - lastSampleMs_) < config_.sampleIntervalMs) {
    emitLast(outSample);
    return true;
  }

  int16_t rawTemperature = 0;
  if (!readTemperatureRaw(rawTemperature)) {
    healthy_ = false;
    if (hasReading_) {
      emitLast(outSample);
      return true;
    }
    return false;
  }

  const float temperatureC = (static_cast<float>(rawTemperature) / kTemperatureScaleLsbPerC) + kTemperatureOffsetC;
  if (!(isfinite(temperatureC) && temperatureC >= -55.0f && temperatureC <= 125.0f)) {
    if (hasReading_) {
      emitLast(outSample);
      return true;
    }
    return false;
  }

  lastTemperatureC_ = temperatureC;
  lastSampleMs_ = nowMs;
  hasReading_ = true;
  healthy_ = true;

  emitLast(outSample);
  return true;
}

bool Lsm6TemperatureSource::probeAddress(uint8_t address7Bit) {
  if (address7Bit == 0) {
    return false;
  }

  address7Bit_ = address7Bit;

  uint8_t whoAmI = 0;
  if (!readReg(kRegWhoAmI, whoAmI)) {
    return false;
  }

  if (whoAmI == kWhoAmILsm6dsox) {
    sourceName_ = "lsm6dsox";
  } else if (whoAmI == kWhoAmILsm6dsv16x) {
    sourceName_ = "lsm6dsv16x";
  } else {
    return false;
  }

  // Ensure accel ODR is non-zero so temperature output updates.
  uint8_t ctrl1Xl = 0;
  if (readReg(kRegCtrl1Xl, ctrl1Xl)) {
    const uint8_t odrMask = 0xF0u;
    if ((ctrl1Xl & odrMask) == 0) {
      const uint8_t odr12_5Hz = 0x10u;
      (void)writeReg(kRegCtrl1Xl, static_cast<uint8_t>((ctrl1Xl & 0x0Fu) | odr12_5Hz));
    }
  }

  return true;
}

bool Lsm6TemperatureSource::readReg(uint8_t reg, uint8_t& value) {
  if (wire_ == nullptr || address7Bit_ == 0) {
    return false;
  }

  wire_->beginTransmission(address7Bit_);
  wire_->write(reg);
  if (wire_->endTransmission(false) != 0) {
    return false;
  }

  if (wire_->requestFrom(static_cast<int>(address7Bit_), 1) != 1) {
    return false;
  }

  value = static_cast<uint8_t>(wire_->read());
  return true;
}

bool Lsm6TemperatureSource::writeReg(uint8_t reg, uint8_t value) {
  if (wire_ == nullptr || address7Bit_ == 0) {
    return false;
  }

  wire_->beginTransmission(address7Bit_);
  wire_->write(reg);
  wire_->write(value);
  return wire_->endTransmission(true) == 0;
}

bool Lsm6TemperatureSource::readTemperatureRaw(int16_t& rawTemperature) {
  if (wire_ == nullptr || address7Bit_ == 0) {
    return false;
  }

  wire_->beginTransmission(address7Bit_);
  wire_->write(kRegOutTempL);
  if (wire_->endTransmission(false) != 0) {
    return false;
  }

  constexpr uint8_t kReadLen = 2;
  if (wire_->requestFrom(static_cast<int>(address7Bit_), static_cast<int>(kReadLen)) != kReadLen) {
    return false;
  }

  const uint8_t tempL = static_cast<uint8_t>(wire_->read());
  const uint8_t tempH = static_cast<uint8_t>(wire_->read());
  rawTemperature = static_cast<int16_t>((static_cast<uint16_t>(tempH) << 8u) | tempL);
  return true;
}

void Lsm6TemperatureSource::emitLast(EnvironmentalSample& outSample) const {
  outSample = EnvironmentalSample();
  outSample.hasTemperatureC = hasReading_;
  outSample.temperatureC = lastTemperatureC_;
  outSample.temperatureSource = sourceName_;
}

}  // namespace mmpr
