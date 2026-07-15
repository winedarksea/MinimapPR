#include "mmpr/SpiHostLink.h"

#include "hardware/gpio.h"

namespace mmpr {

namespace {
// SPI dummy byte clocked out while reading a response (the C5 ignores it;
// 0x00 rather than 0xFF so a bus fault/floating MISO reads as "no data"
// rather than a plausible-looking sync byte).
constexpr uint8_t kSpiFillByte = 0x00;
}  // namespace

SpiHostLink::SpiHostLink(const SpiHostLinkConfig& config) : config_(config) {}

void SpiHostLink::begin() {
  if (began_) {
    return;
  }

  spi_init(config_.spi, config_.baudHz);
  gpio_set_function(config_.sckPin, GPIO_FUNC_SPI);
  gpio_set_function(config_.mosiPin, GPIO_FUNC_SPI);
  gpio_set_function(config_.misoPin, GPIO_FUNC_SPI);

  // CS is driven manually rather than left to the SPI peripheral's hardware
  // CS: the pico-sdk SPI block does not auto-assert/deassert CS per transfer
  // in master mode, so every transaction brackets itself with gpio_put.
  gpio_init(config_.csPin);
  gpio_set_dir(config_.csPin, GPIO_OUT);
  gpio_put(config_.csPin, 1);  // idle high (deasserted)

  // Host-wake: the C5 drives this to tell the RP2350 a response frame is
  // ready. Input with pull-down so an unconnected/unpowered C5 reads as "not
  // ready" rather than floating high.
  gpio_init(config_.wakePin);
  gpio_set_dir(config_.wakePin, GPIO_IN);
  gpio_pull_down(config_.wakePin);

  began_ = true;
}

void SpiHostLink::sendFrame(const std::vector<uint8_t>& encodedFrame) {
  if (!began_ || encodedFrame.empty()) {
    return;
  }

  gpio_put(config_.csPin, 0);
  spi_write_blocking(config_.spi, encodedFrame.data(), encodedFrame.size());
  gpio_put(config_.csPin, 1);
}

bool SpiHostLink::responseReady() const {
  if (!began_) {
    return false;
  }
  return gpio_get(config_.wakePin) != 0;
}

size_t SpiHostLink::readResponse(std::vector<uint8_t>& outBytes, size_t maxBytes) {
  if (!began_ || maxBytes == 0) {
    return 0;
  }

  const size_t offset = outBytes.size();
  outBytes.resize(offset + maxBytes);

  gpio_put(config_.csPin, 0);
  const int read = spi_read_blocking(config_.spi, kSpiFillByte, outBytes.data() + offset, maxBytes);
  gpio_put(config_.csPin, 1);

  const size_t readBytes = read > 0 ? static_cast<size_t>(read) : 0;
  outBytes.resize(offset + readBytes);
  return readBytes;
}

}  // namespace mmpr
