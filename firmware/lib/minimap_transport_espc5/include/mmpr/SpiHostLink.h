#pragma once

#include <cstdint>
#include <vector>

#include "hardware/spi.h"

namespace mmpr {

struct SpiHostLinkConfig {
  spi_inst_t* spi = spi0;
  uint32_t baudHz = 4 * 1000 * 1000;  // 4 MHz to start; bench-tune against the real C5.
  uint8_t sckPin = 22;                // GP22 (kEspSpiSckPin)
  uint8_t mosiPin = 23;               // GP23 (kEspSpiTxPin, host TX -> C5 RX)
  uint8_t misoPin = 20;               // GP20 (kEspSpiRxPin, host RX <- C5 TX)
  uint8_t csPin = 21;                 // GP21 (kEspSpiCsPin)
  uint8_t wakePin = 11;               // GP11 (kEspHostWakePin) -- C5 asserts to signal a pending response
};

// Thin SPI-master HAL for the RP2350 <-> ESP32-C5 SPI link (see node_config.h
// kEspSpi*/kEspHostWakePin for the board's pin assignment). This class owns
// bus init and raw frame TX/RX only; it knows nothing about DATA_POST/
// POST_STATUS semantics. EspC5Publisher (EspC5Publisher.h) owns the
// request/response protocol state machine and calls into this for bytes on
// the wire.
//
// HARDWARE-DEPENDENT / UNVERIFIED WITHOUT A BENCH: actual SPI bus timing
// (clock phase/polarity the C5-side firmware expects, CS setup/hold timing,
// how promptly the C5 asserts wakePin once a response frame is ready) cannot
// be validated without real hardware on both ends of the link. The chip
// selects and manual GPIO CS toggling below follow the standard pico-sdk SPI
// master pattern (hardware/spi.h does not auto-manage CS on this family), but
// the specific mode (CPOL/CPHA) and baud rate are placeholders pending a
// bring-up session against the C5 bridge firmware.
class SpiHostLink {
 public:
  explicit SpiHostLink(const SpiHostLinkConfig& config = SpiHostLinkConfig{});

  // Configures the SPI peripheral and GPIO pin functions. Must be called once
  // before sendFrame/readResponse/responseReady.
  void begin();

  // Sends a fully-encoded frame (see EspC5Frame.h::encodeEspC5Frame) to the
  // C5. Blocking for the duration of the SPI transfer.
  void sendFrame(const std::vector<uint8_t>& encodedFrame);

  // True if the C5 has asserted its host-wake line, i.e. it believes a
  // response frame is ready to be clocked in. Non-blocking.
  bool responseReady() const;

  // Reads up to `maxBytes` bytes of a pending response, appending them to
  // `outBytes`. Returns the number of bytes actually read. Blocking for the
  // duration of the transfer only -- callers should gate on responseReady()
  // (or a timeout) rather than calling this in a tight loop.
  size_t readResponse(std::vector<uint8_t>& outBytes, size_t maxBytes);

  const SpiHostLinkConfig& config() const { return config_; }

 private:
  SpiHostLinkConfig config_;
  bool began_ = false;
};

}  // namespace mmpr
