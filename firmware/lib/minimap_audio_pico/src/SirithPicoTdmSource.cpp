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
#include <hardware/dma.h>
#include <hardware/gpio.h>
#include <hardware/irq.h>
#include <hardware/pio.h>
#include <hardware/pio_instructions.h>
#include <hardware/sync.h>
#include <pico/time.h>

#include <new>

#include "mmpr_audio_rx.pio.h"

namespace mmpr {
namespace {

constexpr uint32_t kPicoMaxGpio = 47;
constexpr uint32_t kDiagnosticLogEveryFrames = 200;

bool pinIsValid(uint8_t pin) {
  return pin <= kPicoMaxGpio;
}

const char* sampleEdgeName(PicoSerialSampleEdge sampleEdge) {
  return sampleEdge == PicoSerialSampleEdge::kFalling ? "falling" : "rising";
}

const char* dataPinBiasName(PicoSerialDataPinBias dataPinBias) {
  return dataPinBias == PicoSerialDataPinBias::kPullDown ? "pull_down" : "disabled";
}

void applyDataPinBias(uint8_t pin, PicoSerialDataPinBias dataPinBias) {
  gpio_disable_pulls(pin);
  if (dataPinBias == PicoSerialDataPinBias::kPullDown) {
    gpio_pull_down(pin);
  }
}

uint32_t applyCaptureBitOffset(uint32_t rawWord, int8_t captureBitOffset) {
  if (captureBitOffset > 0) {
    return rawWord << static_cast<uint8_t>(captureBitOffset);
  }
  if (captureBitOffset < 0) {
    return rawWord >> static_cast<uint8_t>(-captureBitOffset);
  }
  return rawWord;
}

int16_t decodeLeftJustifiedSlotToPcm16(uint32_t rawWord, uint8_t slotBits, uint8_t validBits, int8_t captureBitOffset) {
  rawWord = applyCaptureBitOffset(rawWord, captureBitOffset);
  const uint8_t paddingBits = static_cast<uint8_t>(slotBits - validBits);
  // ADAU7112 drives 24 valid PCM bits within each 32-bit TDM slot, then SDATA
  // is no longer valid for the remaining clocks. Clamp the pad region low so
  // decode does not depend on bus float or external bias between active slots.
  const uint32_t maskedWord =
      paddingBits == 0 ? rawWord : (rawWord & (~((1u << paddingBits) - 1u)));
  int32_t signedPayload = static_cast<int32_t>(maskedWord) >> paddingBits;

  const int shiftToPcm16 = static_cast<int>(validBits) - 16;
  int32_t pcm16 = shiftToPcm16 > 0 ? (signedPayload >> shiftToPcm16) : (signedPayload << (-shiftToPcm16));
  if (pcm16 > 32767) {
    pcm16 = 32767;
  } else if (pcm16 < -32768) {
    pcm16 = -32768;
  }
  return static_cast<int16_t>(pcm16);
}

void logTdmWordDiagnostics(const uint32_t* frameWords, const SirithPicoTdmConfig& config) {
  static uint32_t sDiagFrameCount = 0;
  if (!config.enableWordDiagnostics || (++sDiagFrameCount % kDiagnosticLogEveryFrames) != 0u) {
    return;
  }

  MMPR_PICO_LOG(
      "[tdm-diag] raw[0..3]=%08lX %08lX %08lX %08lX edge=%s offset=%d\n",
      static_cast<unsigned long>(frameWords[0]),
      static_cast<unsigned long>(frameWords[1]),
      static_cast<unsigned long>(frameWords[2]),
      static_cast<unsigned long>(frameWords[3]),
      sampleEdgeName(config.sampleEdge),
      static_cast<int>(config.captureBitOffset));

  for (uint8_t slot = 0; slot < config.tdmSlots; ++slot) {
    const uint32_t rawWord = frameWords[slot];
    const int16_t pcmMinusOne = decodeLeftJustifiedSlotToPcm16(rawWord, config.slotBits, config.validBits, -1);
    const int16_t pcmNominal =
        decodeLeftJustifiedSlotToPcm16(rawWord, config.slotBits, config.validBits, config.captureBitOffset);
    const int16_t pcmPlusOne = decodeLeftJustifiedSlotToPcm16(rawWord, config.slotBits, config.validBits, 1);
    MMPR_PICO_LOG(
        "[tdm-diag] slot=%u pcm[-1]=%d pcm[cur]=%d pcm[+1]=%d\n",
        static_cast<unsigned>(slot),
        static_cast<int>(pcmMinusOne),
        static_cast<int>(pcmNominal),
        static_cast<int>(pcmPlusOne));
  }
}

SirithPicoTdmSource* gActiveDmaSource = nullptr;

}  // namespace

SirithPicoTdmSource::SirithPicoTdmSource(const SirithPicoTdmPins& pins, const SirithPicoTdmConfig& config)
    : pins_(pins), config_(config) {}

SirithPicoTdmSource::~SirithPicoTdmSource() {
  deinitDmaCapture();
  deinitPioStateMachine();
}

bool SirithPicoTdmSource::validateConfig() const {
  if (config_.sampleRateHz == 0 || config_.frameSamples == 0) {
    return false;
  }

  if (config_.tdmSlots != 4 || config_.slotBits != 32 || config_.validBits == 0 ||
      config_.validBits > config_.slotBits) {
    return false;
  }

  const int captureBitOffset = static_cast<int>(config_.captureBitOffset);
  if (captureBitOffset <= -static_cast<int>(config_.slotBits) ||
      captureBitOffset >= static_cast<int>(config_.slotBits)) {
    return false;
  }

  if (!pinIsValid(pins_.dataIn) || !pinIsValid(pins_.bclk) || !pinIsValid(pins_.ws)) {
    return false;
  }

  if (pins_.dataIn == pins_.bclk || pins_.dataIn == pins_.ws || pins_.bclk == pins_.ws) {
    return false;
  }

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

  const pio_program* selectedProgram =
      config_.sampleEdge == PicoSerialSampleEdge::kFalling
          ? &mmpr_sirith_tdm_in_falling_program
          : &mmpr_sirith_tdm_in_program;

  if (!pio_can_add_program(selectedPio, selectedProgram)) {
    pio_sm_unclaim(selectedPio, selectedSm);
    MMPR_PICO_LOG_LINE("[sirith-pico] no room for PIO TDM program");
    return false;
  }

  const uint32_t bitsPerFrame =
      static_cast<uint32_t>(config_.slotBits) * static_cast<uint32_t>(config_.tdmSlots);
  const uint32_t bitRateHz = config_.sampleRateHz * bitsPerFrame;
  // Input-only TDM capture uses one setup instruction plus two instructions
  // per bit. That preserves ADAU7112's one-bit FSYNC pulse without needing an
  // unused data-out phase.
  const uint32_t smCyclesPerFrame = (bitsPerFrame * 2u) + 1u;
  const float smClockHz = static_cast<float>(config_.sampleRateHz) * static_cast<float>(smCyclesPerFrame);
  const float clkSysHz = static_cast<float>(clock_get_hz(clk_sys));
  const float clkDiv = clkSysHz / smClockHz;

  if (!(clkDiv >= 1.0f && clkDiv <= 65535.0f)) {
    pio_sm_unclaim(selectedPio, selectedSm);
    MMPR_PICO_LOG_LINE("[sirith-pico] unsupported sample rate for PIO clock divider");
    return false;
  }

  const uint programOffset = pio_add_program(selectedPio, selectedProgram);

  pio_gpio_init(selectedPio, pins_.dataIn);
  pio_gpio_init(selectedPio, pins_.bclk);
  pio_gpio_init(selectedPio, pins_.ws);

  gpio_set_dir(pins_.dataIn, GPIO_IN);
  // ADAU7112 only drives SDATA during the active portion of each TDM slot, so
  // the node config controls whether the idle bus is weakly biased or left
  // floating.
  applyDataPinBias(pins_.dataIn, config_.dataPinBias);

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

  pio_sm_config smCfg = config_.sampleEdge == PicoSerialSampleEdge::kFalling
      ? mmpr_sirith_tdm_in_falling_program_get_default_config(programOffset)
      : mmpr_sirith_tdm_in_program_get_default_config(programOffset);
  sm_config_set_sideset_pins(&smCfg, pins_.bclk);
  sm_config_set_in_pins(&smCfg, pins_.dataIn);
  sm_config_set_in_shift(&smCfg, false, true, config_.slotBits);
  // Leave TX enabled so the host can push the frame-bit count into Y before
  // capture starts. DMA still services RX only.
  sm_config_set_fifo_join(&smCfg, PIO_FIFO_JOIN_NONE);
  sm_config_set_clkdiv(&smCfg, clkDiv);

  pio_sm_set_consecutive_pindirs(selectedPio, selectedSm, pins_.dataIn, 1, false);
  pio_sm_set_consecutive_pindirs(selectedPio, selectedSm, pins_.bclk, 2, true);

  pio_sm_init(selectedPio, selectedSm, programOffset, &smCfg);
  pio_sm_clear_fifos(selectedPio, selectedSm);
  pio_sm_restart(selectedPio, selectedSm);

  pio_sm_put_blocking(selectedPio, selectedSm, bitsPerFrame - 1u);
  pio_sm_exec(selectedPio, selectedSm, pio_encode_pull(false, false));
  pio_sm_exec(selectedPio, selectedSm, pio_encode_mov(pio_y, pio_osr));

  pio_ = selectedPio;
  sm_ = selectedSm;
  offset_ = programOffset;
  programInstalled_ = true;

  MMPR_PICO_LOG(
      "[sirith-pico] TDM started pio=%d sm=%d offset=%u sr=%lu bclk=%luHz ws=%luHz valid_bits=%u edge=%s bit_offset=%d bias=%s diag=%u\n",
      (selectedPio == pio0) ? 0 : 1,
      selectedSm,
      programOffset,
      static_cast<unsigned long>(config_.sampleRateHz),
      static_cast<unsigned long>(bitRateHz),
      static_cast<unsigned long>(config_.sampleRateHz),
      static_cast<unsigned>(config_.validBits),
      sampleEdgeName(config_.sampleEdge),
      static_cast<int>(config_.captureBitOffset),
      dataPinBiasName(config_.dataPinBias),
      static_cast<unsigned>(config_.enableWordDiagnostics ? 1u : 0u));

  return true;
}

bool SirithPicoTdmSource::initDmaCapture() {
  if (pio_ == nullptr || sm_ < 0 || wordsPerFrame_ == 0) {
    return false;
  }

  dmaFrameWords_ = new (std::nothrow) uint32_t[wordsPerFrame_ * kBufferedFrames];
  if (dmaFrameWords_ == nullptr) {
    MMPR_PICO_LOG_LINE("[sirith-pico] unable to allocate DMA frame ring");
    return false;
  }

  dmaChannel_ = dma_claim_unused_channel(false);
  if (dmaChannel_ < 0) {
    MMPR_PICO_LOG_LINE("[sirith-pico] no free DMA channel");
    delete[] dmaFrameWords_;
    dmaFrameWords_ = nullptr;
    return false;
  }

  dma_channel_config dmaCfg = dma_channel_get_default_config(static_cast<uint>(dmaChannel_));
  channel_config_set_transfer_data_size(&dmaCfg, DMA_SIZE_32);
  channel_config_set_read_increment(&dmaCfg, false);
  channel_config_set_write_increment(&dmaCfg, true);
  channel_config_set_dreq(&dmaCfg, PIO_DREQ_NUM(reinterpret_cast<PIO>(pio_), sm_, false));

  dmaWriteFrameIndex_ = 0;
  dmaReadFrameIndex_ = 0;
  completedFrameCount_ = 0;
  droppedFrameCount_ = 0;
  reportedDroppedFrameCount_ = 0;

  dma_channel_configure(
      static_cast<uint>(dmaChannel_),
      &dmaCfg,
      dmaFrameWords_,
      &reinterpret_cast<PIO>(pio_)->rxf[sm_],
      static_cast<uint32_t>(wordsPerFrame_),
      false);

  gActiveDmaSource = this;
  irq_set_exclusive_handler(DMA_IRQ_0, sDmaIrq);
  dma_channel_acknowledge_irq0(static_cast<uint>(dmaChannel_));
  dma_channel_set_irq0_enabled(static_cast<uint>(dmaChannel_), true);
  irq_set_enabled(DMA_IRQ_0, true);
  dma_channel_start(static_cast<uint>(dmaChannel_));
  pio_sm_set_enabled(reinterpret_cast<PIO>(pio_), sm_, true);
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
    if (config_.sampleEdge == PicoSerialSampleEdge::kFalling) {
      pio_remove_program(pio, &mmpr_sirith_tdm_in_falling_program, offset_);
    } else {
      pio_remove_program(pio, &mmpr_sirith_tdm_in_program, offset_);
    }
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

void SirithPicoTdmSource::deinitDmaCapture() {
  if (dmaChannel_ >= 0) {
    irq_set_enabled(DMA_IRQ_0, false);
    dma_channel_set_irq0_enabled(static_cast<uint>(dmaChannel_), false);
    dma_channel_acknowledge_irq0(static_cast<uint>(dmaChannel_));
    dma_channel_abort(static_cast<uint>(dmaChannel_));
    dma_channel_unclaim(static_cast<uint>(dmaChannel_));
    dmaChannel_ = -1;
  }

  if (gActiveDmaSource == this) {
    gActiveDmaSource = nullptr;
  }

  delete[] dmaFrameWords_;
  dmaFrameWords_ = nullptr;
  dmaWriteFrameIndex_ = 0;
  dmaReadFrameIndex_ = 0;
  completedFrameCount_ = 0;
  droppedFrameCount_ = 0;
  reportedDroppedFrameCount_ = 0;
}

bool SirithPicoTdmSource::begin() {
  deinitDmaCapture();
  deinitPioStateMachine();

  if (!validateConfig()) {
    MMPR_PICO_LOG_LINE("[sirith-pico] invalid TDM config");
    return false;
  }

  if (!initPioStateMachine()) {
    return false;
  }

  wordsPerFrame_ = config_.frameSamples * config_.tdmSlots;
  frameDurationUs_ =
      static_cast<uint64_t>((static_cast<double>(config_.frameSamples) * 1000000.0) /
                            static_cast<double>(config_.sampleRateHz));
  if (!initDmaCapture()) {
    deinitPioStateMachine();
    return false;
  }
  initialized_ = true;
  return true;
}

int16_t SirithPicoTdmSource::toPcm16(int32_t raw) const {
  return decodeLeftJustifiedSlotToPcm16(
      static_cast<uint32_t>(raw), config_.slotBits, config_.validBits, config_.captureBitOffset);
}

bool SirithPicoTdmSource::readFrame(
    int16_t* interleavedOut,
    size_t samplesPerChannel,
    AudioCaptureTimestamp* captureTimestamp) {
  if (!initialized_ || interleavedOut == nullptr || samplesPerChannel != config_.frameSamples ||
      dmaFrameWords_ == nullptr) {
    return false;
  }

  uint32_t readFrameIndex = 0;
  uint64_t frameEndUs = 0;
  uint32_t droppedFramesBeforeCapture = 0;
  while (true) {
    {
      const uint32_t irqState = save_and_disable_interrupts();
      if (completedFrameCount_ == 0) {
        restore_interrupts(irqState);
        tight_loop_contents();
        continue;
      }
      readFrameIndex = dmaReadFrameIndex_;
      frameEndUs = frameEndMonotonicUs_[readFrameIndex];
      droppedFramesBeforeCapture = droppedFrameCount_ - reportedDroppedFrameCount_;
      reportedDroppedFrameCount_ = droppedFrameCount_;
      restore_interrupts(irqState);
    }

    const uint32_t* frameWords = dmaFrameWords_ + (readFrameIndex * wordsPerFrame_);
    logTdmWordDiagnostics(frameWords, config_);

    for (size_t i = 0; i < config_.frameSamples; ++i) {
      const int32_t slotWords[4] = {
          static_cast<int32_t>(frameWords[(i * 4u) + 0u]),
          static_cast<int32_t>(frameWords[(i * 4u) + 1u]),
          static_cast<int32_t>(frameWords[(i * 4u) + 2u]),
          static_cast<int32_t>(frameWords[(i * 4u) + 3u]),
      };

      for (uint8_t channel = 0; channel < 4; ++channel) {
        const uint8_t slot = config_.outputChannelToSlot[channel];
        interleavedOut[(i * 4u) + channel] = toPcm16(slotWords[slot]);
      }
    }

    {
      const uint32_t irqState = save_and_disable_interrupts();
      dmaReadFrameIndex_ = (readFrameIndex + 1u) % kBufferedFrames;
      --completedFrameCount_;
      restore_interrupts(irqState);
    }
    break;
  }

  if (captureTimestamp != nullptr) {
    captureTimestamp->frameEndMonotonicUs = frameEndUs;
    captureTimestamp->frameStartMonotonicUs =
        (frameEndUs >= frameDurationUs_) ? (frameEndUs - frameDurationUs_) : 0;
    captureTimestamp->droppedFramesBeforeCapture = droppedFramesBeforeCapture;
  }
  return true;
}

void SirithPicoTdmSource::onDmaIrq() {
  if (dmaChannel_ < 0 || !dma_channel_get_irq0_status(static_cast<uint>(dmaChannel_))) {
    return;
  }

  dma_channel_acknowledge_irq0(static_cast<uint>(dmaChannel_));
  frameEndMonotonicUs_[dmaWriteFrameIndex_] = time_us_64();
  if (completedFrameCount_ == kBufferedFrames) {
    dmaReadFrameIndex_ = (dmaReadFrameIndex_ + 1u) % kBufferedFrames;
    ++droppedFrameCount_;
  } else {
    ++completedFrameCount_;
  }

  dmaWriteFrameIndex_ = (dmaWriteFrameIndex_ + 1u) % kBufferedFrames;
  dma_channel_set_write_addr(
      static_cast<uint>(dmaChannel_),
      dmaFrameWords_ + (dmaWriteFrameIndex_ * wordsPerFrame_),
      false);
  dma_channel_set_trans_count(static_cast<uint>(dmaChannel_), static_cast<uint32_t>(wordsPerFrame_), true);
}

void SirithPicoTdmSource::sDmaIrq() {
  if (gActiveDmaSource != nullptr) {
    gActiveDmaSource->onDmaIrq();
  }
}

}  // namespace mmpr
