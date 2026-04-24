#include "mmpr/PicoI2SMonoSource.h"

#if !defined(ARDUINO_ARCH_RP2040) && !defined(PICO_RP2040) && !defined(PICO_RP2350)
#error "PicoI2SMonoSource requires RP2040 or RP2350"
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
#include <hardware/structs/dma.h>
#include <hardware/sync.h>
#include <pico/time.h>

#include <algorithm>
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
  // ICS-43434 drives 24 valid I2S bits then releases SD for the remaining pad
  // clocks in the 32-bit slot. Force the pad region low so decode does not
  // depend on board-level pull state or line float during the tri-stated tail.
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

void logI2sWordDiagnostics(const uint32_t* frameWords, const PicoI2SMonoConfig& config) {
  static uint32_t sDiagFrameCount = 0;
  if (!config.enableWordDiagnostics || (++sDiagFrameCount % kDiagnosticLogEveryFrames) != 0u) {
    return;
  }

  const size_t channelWordIndex = (config.channelSide == PicoI2SChannelSide::kRight) ? 1u : 0u;
  const uint32_t rawWord = frameWords[channelWordIndex];
  const int16_t pcmMinusOne = decodeLeftJustifiedSlotToPcm16(rawWord, config.slotBits, config.validBits, -1);
  const int16_t pcmNominal =
      decodeLeftJustifiedSlotToPcm16(rawWord, config.slotBits, config.validBits, config.captureBitOffset);
  const int16_t pcmPlusOne = decodeLeftJustifiedSlotToPcm16(rawWord, config.slotBits, config.validBits, 1);

  MMPR_PICO_LOG(
      "[pico-i2s-diag] raw=%08lX edge=%s offset=%d pcm[-1]=%d pcm[cur]=%d pcm[+1]=%d\n",
      static_cast<unsigned long>(rawWord),
      sampleEdgeName(config.sampleEdge),
      static_cast<int>(config.captureBitOffset),
      static_cast<int>(pcmMinusOne),
      static_cast<int>(pcmNominal),
      static_cast<int>(pcmPlusOne));
}

PicoI2SMonoSource* gActiveDmaSource = nullptr;

}  // namespace

PicoI2SMonoSource::PicoI2SMonoSource(const PicoI2SMonoPins& pins, const PicoI2SMonoConfig& config)
    : pins_(pins), config_(config) {}

PicoI2SMonoSource::~PicoI2SMonoSource() {
  deinitDmaCapture();
  deinitPioStateMachine();
}

bool PicoI2SMonoSource::validateConfig() const {
  if (config_.sampleRateHz == 0 || config_.frameSamples == 0) {
    return false;
  }

  if (config_.slotBits != 32 || config_.validBits == 0 || config_.validBits > config_.slotBits) {
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

  return true;
}

bool PicoI2SMonoSource::initPioStateMachine() {
  PIO selectedPio = pio0;
  int selectedSm = pio_claim_unused_sm(selectedPio, false);
  if (selectedSm < 0) {
    selectedPio = pio1;
    selectedSm = pio_claim_unused_sm(selectedPio, false);
    if (selectedSm < 0) {
      MMPR_PICO_LOG_LINE("[pico-i2s] no free PIO state machine");
      return false;
    }
  }

  const pio_program* selectedProgram =
      config_.sampleEdge == PicoSerialSampleEdge::kFalling
          ? &mmpr_pico_i2s_in_falling_program
          : &mmpr_pico_i2s_in_program;

  if (!pio_can_add_program(selectedPio, selectedProgram)) {
    pio_sm_unclaim(selectedPio, selectedSm);
    MMPR_PICO_LOG_LINE("[pico-i2s] no room for PIO I2S program");
    return false;
  }

  const uint32_t bitsPerFrame = static_cast<uint32_t>(config_.slotBits) * 2u;
  const uint32_t bitRateHz = config_.sampleRateHz * bitsPerFrame;
  const uint32_t smCyclesPerFrame = (bitsPerFrame * 2u) + 2u;
  const float smClockHz = static_cast<float>(config_.sampleRateHz) * static_cast<float>(smCyclesPerFrame);
  const float clkSysHz = static_cast<float>(clock_get_hz(clk_sys));
  const float clkDiv = clkSysHz / smClockHz;

  if (!(clkDiv >= 1.0f && clkDiv <= 65535.0f)) {
    pio_sm_unclaim(selectedPio, selectedSm);
    MMPR_PICO_LOG_LINE("[pico-i2s] unsupported sample rate for PIO clock divider");
    return false;
  }

  const uint programOffset = pio_add_program(selectedPio, selectedProgram);

  pio_gpio_init(selectedPio, pins_.dataIn);
  pio_gpio_init(selectedPio, pins_.bclk);
  pio_gpio_init(selectedPio, pins_.ws);

  gpio_set_dir(pins_.dataIn, GPIO_IN);
  // ICS-43434 tri-states SD outside its active channel word, so allow the node
  // config to choose whether the inactive half-frame is weakly pulled down or
  // left unbiased.
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
      ? mmpr_pico_i2s_in_falling_program_get_default_config(programOffset)
      : mmpr_pico_i2s_in_program_get_default_config(programOffset);
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

  // Match the reference driver: preload the loop counter and compensate for
  // I2S's LRCLK-to-MSB one-bit delay before DMA starts consuming words.
  pio_sm_exec(selectedPio, selectedSm, pio_encode_set(pio_y, config_.slotBits - 2));
  pio_sm_exec(selectedPio, selectedSm, pio_encode_in(pio_pins, config_.slotBits));
  pio_sm_exec(selectedPio, selectedSm, pio_encode_in(pio_pins, config_.slotBits - 1));

  pio_ = selectedPio;
  sm_ = selectedSm;
  offset_ = programOffset;
  programInstalled_ = true;

  MMPR_PICO_LOG(
      "[pico-i2s] started pio=%d sm=%d offset=%u sr=%lu bclk=%luHz ws=%luHz channel=%s valid_bits=%u edge=%s bit_offset=%d bias=%s diag=%u\n",
      (selectedPio == pio0) ? 0 : 1,
      selectedSm,
      programOffset,
      static_cast<unsigned long>(config_.sampleRateHz),
      static_cast<unsigned long>(bitRateHz),
      static_cast<unsigned long>(config_.sampleRateHz),
      (config_.channelSide == PicoI2SChannelSide::kRight) ? "right" : "left",
      static_cast<unsigned>(config_.validBits),
      sampleEdgeName(config_.sampleEdge),
      static_cast<int>(config_.captureBitOffset),
      dataPinBiasName(config_.dataPinBias),
      static_cast<unsigned>(config_.enableWordDiagnostics ? 1u : 0u));

  return true;
}

bool PicoI2SMonoSource::initDmaCapture() {
  if (pio_ == nullptr || sm_ < 0 || wordsPerFrame_ == 0) {
    return false;
  }

  dmaFrameWords_ = new (std::nothrow) uint32_t[wordsPerFrame_ * kBufferedFrames];
  if (dmaFrameWords_ == nullptr) {
    MMPR_PICO_LOG_LINE("[pico-i2s] unable to allocate DMA frame ring");
    return false;
  }

  dmaChannel_ = dma_claim_unused_channel(false);
  if (dmaChannel_ < 0) {
    MMPR_PICO_LOG_LINE("[pico-i2s] no free DMA channel");
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

void PicoI2SMonoSource::deinitPioStateMachine() {
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
      pio_remove_program(pio, &mmpr_pico_i2s_in_falling_program, offset_);
    } else {
      pio_remove_program(pio, &mmpr_pico_i2s_in_program, offset_);
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

void PicoI2SMonoSource::deinitDmaCapture() {
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
  nextProducedStartSampleIndex_ = 0;
  nextCompletedBlockCount_ = 0;
}

bool PicoI2SMonoSource::begin() {
  deinitDmaCapture();
  deinitPioStateMachine();

  if (!validateConfig()) {
    MMPR_PICO_LOG_LINE("[pico-i2s] invalid mono I2S config");
    return false;
  }

  if (!initPioStateMachine()) {
    return false;
  }

  wordsPerFrame_ = config_.frameSamples * 2u;
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

int16_t PicoI2SMonoSource::toPcm16(int32_t raw) const {
  return decodeLeftJustifiedSlotToPcm16(
      static_cast<uint32_t>(raw), config_.slotBits, config_.validBits, config_.captureBitOffset);
}

bool PicoI2SMonoSource::readFrame(
    int16_t* interleavedOut,
    size_t samplesPerChannel,
    AudioCaptureTimestamp* captureTimestamp) {
  if (!initialized_ || interleavedOut == nullptr || samplesPerChannel != config_.frameSamples ||
      dmaFrameWords_ == nullptr) {
    return false;
  }

  uint32_t readFrameIndex = 0;
  uint64_t blockEndUs = 0;
  uint64_t blockStartSampleIndex = 0;
  uint64_t completedBlockCount = 0;
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
      blockStartSampleIndex = blockStartSampleIndex_[readFrameIndex];
      blockEndUs = blockEndMonotonicUs_[readFrameIndex];
      completedBlockCount = completedBlockCountBySlot_[readFrameIndex];
      droppedFramesBeforeCapture = droppedFrameCount_ - reportedDroppedFrameCount_;
      reportedDroppedFrameCount_ = droppedFrameCount_;
      restore_interrupts(irqState);
    }

    const uint32_t* frameWords = dmaFrameWords_ + (readFrameIndex * wordsPerFrame_);
    logI2sWordDiagnostics(frameWords, config_);
    const size_t channelWordIndex = (config_.channelSide == PicoI2SChannelSide::kRight) ? 1u : 0u;
    for (size_t i = 0; i < config_.frameSamples; ++i) {
      const int32_t raw = static_cast<int32_t>(frameWords[(i * 2u) + channelWordIndex]);
      interleavedOut[i] = toPcm16(raw);
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
    captureTimestamp->startSampleIndex = blockStartSampleIndex;
    captureTimestamp->endSampleIndex = blockStartSampleIndex + config_.frameSamples;
    captureTimestamp->blockEndMonotonicUs = blockEndUs;
    captureTimestamp->blockStartMonotonicUs =
        (blockEndUs >= frameDurationUs_) ? (blockEndUs - frameDurationUs_) : 0;
    captureTimestamp->completedBlockCount = completedBlockCount;
    captureTimestamp->dmaRingSlotIndex = readFrameIndex;
    captureTimestamp->droppedBlocksBeforeCapture = droppedFramesBeforeCapture;
  }
  return true;
}

bool PicoI2SMonoSource::snapshotProducerState(
    AudioProducerSnapshot& producerSnapshot,
    bool callerAlreadyInIrqContext) const {
  producerSnapshot = {};
  if (!initialized_ || dmaChannel_ < 0 || wordsPerFrame_ == 0) {
    return false;
  }

  const uint32_t irqState = callerAlreadyInIrqContext ? 0u : save_and_disable_interrupts();
  const uint64_t activeBlockStartSampleIndex = nextProducedStartSampleIndex_;
  const uint64_t completedBlockCount = nextCompletedBlockCount_;
  const uint32_t dmaRingSlotIndex = dmaWriteFrameIndex_;
  dma_channel_hw_t* channelHw = dma_channel_hw_addr(static_cast<uint>(dmaChannel_));
  const uint32_t wordsRemaining = channelHw->transfer_count;
  if (!callerAlreadyInIrqContext) {
    restore_interrupts(irqState);
  }

  const uint32_t clampedWordsRemaining = std::min<uint32_t>(wordsRemaining, static_cast<uint32_t>(wordsPerFrame_));
  const uint32_t wordsTransferred = static_cast<uint32_t>(wordsPerFrame_) - clampedWordsRemaining;
  const double sampleOffset = static_cast<double>(wordsTransferred) / 2.0;

  producerSnapshot.valid = true;
  producerSnapshot.capturedMonotonicUs = time_us_64();
  producerSnapshot.samplePosition = static_cast<double>(activeBlockStartSampleIndex) + sampleOffset;
  producerSnapshot.completedBlockCount = completedBlockCount;
  producerSnapshot.dmaRingSlotIndex = dmaRingSlotIndex;
  producerSnapshot.wordsTransferredInActiveBlock = wordsTransferred;
  producerSnapshot.wordsRemainingInActiveBlock = clampedWordsRemaining;
  return true;
}

void PicoI2SMonoSource::onDmaIrq() {
  if (dmaChannel_ < 0 || !dma_channel_get_irq0_status(static_cast<uint>(dmaChannel_))) {
    return;
  }

  dma_channel_acknowledge_irq0(static_cast<uint>(dmaChannel_));
  blockStartSampleIndex_[dmaWriteFrameIndex_] = nextProducedStartSampleIndex_;
  nextProducedStartSampleIndex_ += config_.frameSamples;
  blockEndMonotonicUs_[dmaWriteFrameIndex_] = time_us_64();
  completedBlockCountBySlot_[dmaWriteFrameIndex_] = ++nextCompletedBlockCount_;
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

void PicoI2SMonoSource::sDmaIrq() {
  if (gActiveDmaSource != nullptr) {
    gActiveDmaSource->onDmaIrq();
  }
}

}  // namespace mmpr
