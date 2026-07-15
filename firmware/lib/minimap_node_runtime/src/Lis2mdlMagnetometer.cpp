#include "mmpr/Lis2mdlMagnetometer.h"

namespace mmpr {
namespace {

// LIS2MDLTR register addresses.
constexpr uint8_t kRegWhoAmI = 0x4F;
constexpr uint8_t kRegCfgA   = 0x60;
constexpr uint8_t kRegCfgB   = 0x61;
constexpr uint8_t kRegCfgC   = 0x62;
constexpr uint8_t kRegOutXL  = 0x68;

constexpr uint8_t kWhoAmIExpected = 0x40;

}  // namespace

Lis2mdlMagnetometer::Lis2mdlMagnetometer(I2cBus& bus, const Lis2mdlMagConfig& config)
    : bus_(&bus), config_(config) {}

bool Lis2mdlMagnetometer::begin() {
  started_ = false;
  healthy_ = false;

  if (bus_ == nullptr || !bus_->initialized()) {
    return false;
  }

  // Probe WHO_AM_I.
  uint8_t whoAmI = 0;
  if (!readReg(kRegWhoAmI, whoAmI) || whoAmI != kWhoAmIExpected) {
    return false;
  }

  // CFG_REG_A: continuous mode (bits 1:0 = 00), ODR, temp comp.
  // High-resolution mode is the default (LP bit = 0).
  uint8_t cfgA = 0x00u;
  cfgA |= static_cast<uint8_t>((config_.odrBits & 0x03u) << 2u);
  if (config_.enableTempComp) {
    cfgA |= 0x80u;  // COMP_TEMP_EN
  }

  // CFG_REG_B: hardware LPF, offset cancellation.
  uint8_t cfgB = 0x00u;
  if (config_.enableLpf) {
    cfgB |= 0x01u;  // LPF bit — bandwidth = ODR / 4
  }
  if (config_.enableOffsetCancel) {
    cfgB |= 0x02u;  // OFF_CANC — automatic offset cancellation via SET/RESET
  }

  // CFG_REG_C: BDU (block data update) for consistent multi-byte reads.
  constexpr uint8_t cfgC = 0x10u;

  if (!writeReg(kRegCfgA, cfgA) ||
      !writeReg(kRegCfgB, cfgB) ||
      !writeReg(kRegCfgC, cfgC)) {
    return false;
  }

  started_ = true;
  healthy_ = true;
  return true;
}

bool Lis2mdlMagnetometer::readField(float& x, float& y, float& z) {
  if (!started_ || bus_ == nullptr) {
    return false;
  }

  uint8_t raw[6] = {0};
  if (!bus_->readReg(config_.i2cAddress7Bit, kRegOutXL, raw, sizeof(raw))) {
    healthy_ = false;
    return false;
  }

  const auto rawX = static_cast<int16_t>((static_cast<uint16_t>(raw[1]) << 8u) | raw[0]);
  const auto rawY = static_cast<int16_t>((static_cast<uint16_t>(raw[3]) << 8u) | raw[2]);
  const auto rawZ = static_cast<int16_t>((static_cast<uint16_t>(raw[5]) << 8u) | raw[4]);

  // Apply hard-iron calibration offsets.
  x = static_cast<float>(rawX) - config_.hardIronOffsetX;
  y = static_cast<float>(rawY) - config_.hardIronOffsetY;
  z = static_cast<float>(rawZ) - config_.hardIronOffsetZ;

  healthy_ = true;
  return true;
}

bool Lis2mdlMagnetometer::readReg(uint8_t reg, uint8_t& value) {
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

bool Lis2mdlMagnetometer::writeReg(uint8_t reg, uint8_t value) {
  if (bus_ == nullptr) {
    return false;
  }
  return bus_->writeReg(config_.i2cAddress7Bit, reg, value);
}

}  // namespace mmpr
