#include "mmpr/SirithPicoTdmSource.h"

#if !defined(ARDUINO_ARCH_RP2040) && !defined(PICO_RP2040) && !defined(PICO_RP2350)
#error "SirithPicoTdmSource requires RP2040 or RP2350"
#endif

#if defined(ARDUINO)
#include <Arduino.h>
#define MMPR_PICO_LOG(...) Serial.printf(__VA_ARGS__)
#define MMPR_PICO_LOG_LINE(msg) Serial.println(msg)
#else
#include <cstdio>
#define MMPR_PICO_LOG(...) std::printf(__VA_ARGS__)
#define MMPR_PICO_LOG_LINE(msg) std::puts(msg)
#endif

#include <hardware/clocks.h>
#include <hardware/gpio.h>
#include <hardware/pio.h>
#include <hardware/pio_instructions.h>

namespace mmpr {
namespace {

// Master TDM receiver.
// - Generates BCLK/FSYNC on side-set pins (BCLK = side bit 0, FSYNC = side bit 1)
// - Samples one SDATA bit per BCLK rising edge
// - Uses a one-bit positive FSYNC pulse at frame start (ADAU7112 TDM requirement)
// - Auto-pushes every 32 bits -> one slot per FIFO word
static const uint16_t kSirithTdmMasterRxInstructions[] = {
    static_cast<uint16_t>(pio_encode_set(pio_x, 127) | pio_encode_sideset(2, 0x2)),      // FSYNC high, BCLK low
    static_cast<uint16_t>(pio_encode_in(pio_pins, 1) | pio_encode_sideset(2, 0x3)),       // First rising edge sample
    static_cast<uint16_t>(pio_encode_jmp_x_dec(3) | pio_encode_sideset(2, 0x0)),          // FSYNC pulse ends, jump into steady loop
    static_cast<uint16_t>(pio_encode_in(pio_pins, 1) | pio_encode_sideset(2, 0x1)),       // BCLK high, sample SDATA
    static_cast<uint16_t>(pio_encode_jmp_x_dec(3) | pio_encode_sideset(2, 0x0)),          // BCLK low
};

static const pio_program kSirithTdmMasterRxProgram = {
    kSirithTdmMasterRxInstructions,
    static_cast<uint>(sizeof(kSirithTdmMasterRxInstructions) / sizeof(kSirithTdmMasterRxInstructions[0])),
    -1,
};

constexpr uint32_t kPicoMaxGpio = 47;

bool pinIsValid(uint8_t pin) {
  return pin <= kPicoMaxGpio;
}

}  // namespace

SirithPicoTdmSource::SirithPicoTdmSource(const SirithPicoTdmPins& pins, const SirithPicoTdmConfig& config)
    : pins_(pins), config_(config) {}

SirithPicoTdmSource::~SirithPicoTdmSource() {
  deinitPioStateMachine();
}

bool SirithPicoTdmSource::validateConfig() const {
  if (config_.sampleRateHz == 0 || config_.frameSamples == 0) {
    return false;
  }

  if (config_.tdmSlots != 4 || config_.slotBits != 32) {
    return false;
  }

  if (!pinIsValid(pins_.dataIn) || !pinIsValid(pins_.bclk) || !pinIsValid(pins_.ws)) {
    return false;
  }

  if (pins_.dataIn == pins_.bclk || pins_.dataIn == pins_.ws || pins_.bclk == pins_.ws) {
    return false;
  }

  // Side-set requires consecutive output pins.
  if (pins_.ws != (pins_.bclk + 1)) {
    return false;
  }

  for (uint8_t i = 0; i < 4; ++i) {
    if (config_.outputChannelToSlot[i] >= config_.tdmSlots) {
      return false;
    }
  }

  return true;
}

bool SirithPicoTdmSource::initPioStateMachine() {
  PIO selectedPio = pio0;
  int selectedSm = pio_claim_unused_sm(selectedPio, false);
  if (selectedSm < 0) {
      selectedPio = pio1;
      selectedSm = pio_claim_unused_sm(selectedPio, false);
      if (selectedSm < 0) {
      MMPR_PICO_LOG_LINE("[sirith-pico] no free PIO state machine");
      return false;
    }
  }

  if (!pio_can_add_program(selectedPio, &kSirithTdmMasterRxProgram)) {
    pio_sm_unclaim(selectedPio, selectedSm);
    MMPR_PICO_LOG_LINE("[sirith-pico] no room for PIO TDM program");
    return false;
  }

  const uint32_t bitsPerFrame = static_cast<uint32_t>(config_.slotBits) * static_cast<uint32_t>(config_.tdmSlots);
  const uint32_t bitRateHz = config_.sampleRateHz * bitsPerFrame;
  // This program uses 2 cycles/bit plus one setup cycle at each frame boundary.
  const uint32_t smCyclesPerFrame = (bitsPerFrame * 2u) + 1u;
  const float smClockHz = static_cast<float>(config_.sampleRateHz) * static_cast<float>(smCyclesPerFrame);
  const float clkSysHz = static_cast<float>(clock_get_hz(clk_sys));
  const float clkDiv = clkSysHz / smClockHz;

  if (!(clkDiv >= 1.0f && clkDiv <= 65535.0f)) {
    pio_sm_unclaim(selectedPio, selectedSm);
    MMPR_PICO_LOG_LINE("[sirith-pico] unsupported sample rate for PIO clock divider");
    return false;
  }

  const uint programOffset = pio_add_program(selectedPio, &kSirithTdmMasterRxProgram);

  // The jmp-x-- instructions encode an absolute PIO instruction-memory address.
  // Patch instructions 2 and 4 to jump to programOffset+3 (the inner sample
  // instruction) now that the actual load address is known.
  const uint jmpTarget = programOffset + 3u;
  selectedPio->instr_mem[programOffset + 2u] =
      (kSirithTdmMasterRxInstructions[2] & ~0x1Fu) | (jmpTarget & 0x1Fu);
  selectedPio->instr_mem[programOffset + 4u] =
      (kSirithTdmMasterRxInstructions[4] & ~0x1Fu) | (jmpTarget & 0x1Fu);

  pio_gpio_init(selectedPio, pins_.dataIn);
  pio_gpio_init(selectedPio, pins_.bclk);
  pio_gpio_init(selectedPio, pins_.ws);

  gpio_set_dir(pins_.dataIn, GPIO_IN);
  gpio_pull_down(pins_.dataIn);

  gpio_set_dir(pins_.bclk, GPIO_OUT);
  gpio_set_dir(pins_.ws, GPIO_OUT);
  gpio_put(pins_.bclk, 0);
  gpio_put(pins_.ws, 0);

  if (config_.useSafeDriveStrength) {
#if defined(GPIO_DRIVE_STRENGTH_2MA)
    gpio_set_drive_strength(pins_.bclk, GPIO_DRIVE_STRENGTH_2MA);
    gpio_set_drive_strength(pins_.ws, GPIO_DRIVE_STRENGTH_2MA);
#endif
#if defined(GPIO_SLEW_RATE_SLOW)
    gpio_set_slew_rate(pins_.bclk, GPIO_SLEW_RATE_SLOW);
    gpio_set_slew_rate(pins_.ws, GPIO_SLEW_RATE_SLOW);
#endif
  }

  pio_sm_config smCfg = pio_get_default_sm_config();
  sm_config_set_wrap(&smCfg, programOffset + 0u, programOffset + 4u);
  sm_config_set_sideset(&smCfg, 2, false, false);
  sm_config_set_sideset_pins(&smCfg, pins_.bclk);
  sm_config_set_in_pins(&smCfg, pins_.dataIn);
  sm_config_set_in_shift(&smCfg, false, true, config_.slotBits);
  sm_config_set_fifo_join(&smCfg, PIO_FIFO_JOIN_RX);
  sm_config_set_clkdiv(&smCfg, clkDiv);

  pio_sm_set_consecutive_pindirs(selectedPio, selectedSm, pins_.dataIn, 1, false);
  pio_sm_set_consecutive_pindirs(selectedPio, selectedSm, pins_.bclk, 2, true);

  pio_sm_init(selectedPio, selectedSm, programOffset, &smCfg);
  pio_sm_clear_fifos(selectedPio, selectedSm);
  pio_sm_restart(selectedPio, selectedSm);
  pio_sm_set_enabled(selectedPio, selectedSm, true);

  pio_ = selectedPio;
  sm_ = selectedSm;
  offset_ = programOffset;
  programInstalled_ = true;

  MMPR_PICO_LOG(
      "[sirith-pico] TDM started pio=%d sm=%d offset=%u sr=%lu bclk=%luHz ws=%luHz\n",
      (selectedPio == pio0) ? 0 : 1,
      selectedSm,
      programOffset,
      static_cast<unsigned long>(config_.sampleRateHz),
      static_cast<unsigned long>(bitRateHz),
      static_cast<unsigned long>(config_.sampleRateHz));

  return true;
}

void SirithPicoTdmSource::deinitPioStateMachine() {
  if (!initialized_ && pio_ == nullptr) {
    return;
  }

  PIO pio = reinterpret_cast<PIO>(pio_);
  if (pio != nullptr && sm_ >= 0) {
    pio_sm_set_enabled(pio, sm_, false);
    pio_sm_unclaim(pio, sm_);
  }

  if (pio != nullptr && programInstalled_) {
    pio_remove_program(pio, &kSirithTdmMasterRxProgram, offset_);
  }

  gpio_put(pins_.bclk, 0);
  gpio_put(pins_.ws, 0);
  gpio_set_dir(pins_.bclk, GPIO_IN);
  gpio_set_dir(pins_.ws, GPIO_IN);

  pio_ = nullptr;
  sm_ = -1;
  offset_ = 0;
  programInstalled_ = false;
  initialized_ = false;
}

bool SirithPicoTdmSource::begin() {
  deinitPioStateMachine();

  if (!validateConfig()) {
    MMPR_PICO_LOG_LINE("[sirith-pico] invalid TDM config");
    return false;
  }

  if (!initPioStateMachine()) {
    return false;
  }

  initialized_ = true;
  return true;
}

int16_t SirithPicoTdmSource::toPcm16(int32_t raw) const {
  int32_t shifted = raw >> config_.sampleShiftBits;
  if (shifted > 32767) {
    shifted = 32767;
  } else if (shifted < -32768) {
    shifted = -32768;
  }
  return static_cast<int16_t>(shifted);
}

bool SirithPicoTdmSource::readFrame(int16_t* interleavedOut, size_t samplesPerChannel) {
  if (!initialized_ || interleavedOut == nullptr || samplesPerChannel != config_.frameSamples || pio_ == nullptr || sm_ < 0) {
    return false;
  }

  PIO pio = reinterpret_cast<PIO>(pio_);
  for (size_t i = 0; i < config_.frameSamples; ++i) {
    int32_t slotWords[4] = {0, 0, 0, 0};
    for (uint8_t slot = 0; slot < 4; ++slot) {
      slotWords[slot] = static_cast<int32_t>(pio_sm_get_blocking(pio, sm_));
    }

    for (uint8_t channel = 0; channel < 4; ++channel) {
      const uint8_t slot = config_.outputChannelToSlot[channel];
      interleavedOut[(i * 4) + channel] = toPcm16(slotWords[slot]);
    }
  }

  return true;
}

}  // namespace mmpr
