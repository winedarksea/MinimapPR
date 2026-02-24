#include "mmpr/Lsm6TemperatureSensor.h"

#include <cmath>

#include "pico/time.h"

namespace mmpr {
namespace {

constexpr uint8_t kRegWhoAmI = 0x0F;
constexpr uint8_t kRegCtrl1Xl = 0x10;
constexpr uint8_t kRegOutTempL = 0x20;

constexpr uint8_t kWhoAmILsm6dsox = 0x6C;
constexpr uint8_t kWhoAmILsm6dsv16x = 0x70;

constexpr float kTemperatureScaleLsbPerC = 256.0f;
constexpr float kTemperatureOffsetC = 25.0f;

uint32_t millis32() {
  return to_ms_since_boot(get_absolute_time());
}

}  // namespace

Lsm6TemperatureSensor::Lsm6TemperatureSensor(I2cBus& bus, const Lsm6TemperatureSensorConfig& config)
    : bus_(&bus), config_(config) {}

bool Lsm6TemperatureSensor::begin() {
  started_ = false;
  healthy_ = false;
  hasReading_ = false;
  address7Bit_ = 0;
  sourceName_ = "lsm6";

  if (bus_ == nullptr || !bus_->initialized()) {
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

bool Lsm6TemperatureSensor::readTemperatureC(float& outTemperatureC, const char*& outSourceName) {
  if (!started_) {
    return false;
  }

  const uint32_t nowMs = millis32();
  if (hasReading_ && (nowMs - lastSampleMs_) < config_.sampleIntervalMs) {
    outTemperatureC = lastTemperatureC_;
    outSourceName = sourceName_;
    return true;
  }

  int16_t rawTemperature = 0;
  if (!readTemperatureRaw(rawTemperature)) {
    healthy_ = false;
    if (!hasReading_) {
      return false;
    }
    outTemperatureC = lastTemperatureC_;
    outSourceName = sourceName_;
    return true;
  }

  const float temperatureC = (static_cast<float>(rawTemperature) / kTemperatureScaleLsbPerC) + kTemperatureOffsetC;
  if (!(std::isfinite(temperatureC) && temperatureC >= -55.0f && temperatureC <= 125.0f)) {
    if (!hasReading_) {
      return false;
    }
    outTemperatureC = lastTemperatureC_;
    outSourceName = sourceName_;
    return true;
  }

  lastTemperatureC_ = temperatureC;
  lastSampleMs_ = nowMs;
  hasReading_ = true;
  healthy_ = true;

  outTemperatureC = lastTemperatureC_;
  outSourceName = sourceName_;
  return true;
}

bool Lsm6TemperatureSensor::probeAddress(uint8_t address7Bit) {
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

bool Lsm6TemperatureSensor::readReg(uint8_t reg, uint8_t& value) {
  if (bus_ == nullptr || address7Bit_ == 0) {
    return false;
  }

  uint8_t out = 0;
  if (!bus_->readReg(address7Bit_, reg, &out, 1)) {
    return false;
  }

  value = out;
  return true;
}

bool Lsm6TemperatureSensor::writeReg(uint8_t reg, uint8_t value) {
  if (bus_ == nullptr || address7Bit_ == 0) {
    return false;
  }
  return bus_->writeReg(address7Bit_, reg, value);
}

bool Lsm6TemperatureSensor::readTemperatureRaw(int16_t& rawTemperature) {
  if (bus_ == nullptr || address7Bit_ == 0) {
    return false;
  }

  uint8_t raw[2] = {0};
  if (!bus_->readReg(address7Bit_, kRegOutTempL, raw, sizeof(raw))) {
    return false;
  }

  rawTemperature = static_cast<int16_t>((static_cast<uint16_t>(raw[1]) << 8u) | raw[0]);
  return true;
}

}  // namespace mmpr
