#pragma once

#include <cstddef>
#include <cstdint>

#include "hardware/gpio.h"
#include "hardware/i2c.h"

namespace mmpr {

class I2cBus {
 public:
  bool begin(i2c_inst_t* inst, uint sdaPin, uint sclPin, uint32_t baudHz);

  bool initialized() const { return initialized_; }

  // Optional callback invoked after each successful bus read.  Intended for
  // lightweight notification (e.g. an activity LED pulse); must not block.
  void setReadCallback(void (*cb)()) { readCallback_ = cb; }

  bool write(uint8_t address7Bit, const uint8_t* data, size_t length, bool noStop);
  bool read(uint8_t address7Bit, uint8_t* data, size_t length, bool noStop);

  bool readReg(uint8_t address7Bit, uint8_t reg, uint8_t* outData, size_t outLen);
  bool writeReg(uint8_t address7Bit, uint8_t reg, uint8_t value);

 private:
  // Bound each transfer so an optional peripheral cannot wedge the node if the
  // bus is held or a device stretches forever during bring-up.
  static constexpr uint32_t kTransferTimeoutUs = 20000;

  i2c_inst_t* inst_ = nullptr;
  bool initialized_ = false;
  void (*readCallback_)() = nullptr;
};

inline bool I2cBus::begin(i2c_inst_t* inst, uint sdaPin, uint sclPin, uint32_t baudHz) {
  if (inst == nullptr) {
    return false;
  }

  inst_ = inst;
  i2c_init(inst_, baudHz);
  gpio_set_function(sdaPin, GPIO_FUNC_I2C);
  gpio_set_function(sclPin, GPIO_FUNC_I2C);
  gpio_pull_up(sdaPin);
  gpio_pull_up(sclPin);
  initialized_ = true;
  return true;
}

inline bool I2cBus::write(uint8_t address7Bit, const uint8_t* data, size_t length, bool noStop) {
  if (!initialized_ || data == nullptr || length == 0) {
    return false;
  }

  const int written = i2c_write_timeout_us(inst_, address7Bit, data, length, noStop, kTransferTimeoutUs);
  return written == static_cast<int>(length);
}

inline bool I2cBus::read(uint8_t address7Bit, uint8_t* data, size_t length, bool noStop) {
  if (!initialized_ || data == nullptr || length == 0) {
    return false;
  }

  const int readCount = i2c_read_timeout_us(inst_, address7Bit, data, length, noStop, kTransferTimeoutUs);
  if (readCount == static_cast<int>(length)) {
    if (readCallback_ != nullptr) {
      readCallback_();
    }
    return true;
  }
  return false;
}

inline bool I2cBus::readReg(uint8_t address7Bit, uint8_t reg, uint8_t* outData, size_t outLen) {
  if (!write(address7Bit, &reg, 1, true)) {
    return false;
  }
  return read(address7Bit, outData, outLen, false);
}

inline bool I2cBus::writeReg(uint8_t address7Bit, uint8_t reg, uint8_t value) {
  uint8_t payload[2] = {reg, value};
  return write(address7Bit, payload, 2, false);
}

}  // namespace mmpr
