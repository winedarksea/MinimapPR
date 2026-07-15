#include "mmpr/PdmPlanarSource.h"

#if !defined(PICO_RP2040) && !defined(PICO_RP2350)
#error "PdmPlanarSource requires RP2040 or RP2350"
#endif

#include <cstdio>

#include <hardware/clocks.h>
#include <hardware/dma.h>
#include <hardware/gpio.h>
#include <hardware/irq.h>
#include <hardware/pio.h>
#include <hardware/pio_instructions.h>
#include <hardware/structs/dma.h>
#include <hardware/sync.h>
#include <pico/multicore.h>
#include <pico/time.h>

#include <algorithm>
#include <new>

#include "mmpr_pdm_rx.pio.h"

namespace mmpr {
namespace {

constexpr uint32_t kPicoMaxGpio = 47;
constexpr double kChipRateHz = 3072000.0;

bool pinIsValid(uint8_t pin) { return pin <= kPicoMaxGpio; }

PdmPlanarSource* gActiveDmaSource = nullptr;

}  // namespace

PdmPlanarSource::PdmPlanarSource(const PdmPlanarPins& pins, const PdmPlanarConfig& config)
    : pins_(pins), config_(config), decimator_(PdmCicDecimator::Config{true, config.enableDither, 0x9E3779B9u}) {}

PdmPlanarSource::~PdmPlanarSource() {
  deinitDmaCapture();
  deinitPioStateMachine();
  delete[] rawWordRing_;
  delete[] frameRing_;
  delete[] frameStartSampleIndex_;
}

bool PdmPlanarSource::validateConfig() const {
  if (config_.sampleRateHz == 0 || config_.frameSamples == 0) return false;
  if (config_.ringFrames == 0 || config_.ringFrames > 64) return false;
  if (pins_.dataPinCount != 3 || !pinIsValid(pins_.dataPinBase) || !pinIsValid(pins_.clockPin)) return false;
  return true;
}

bool PdmPlanarSource::initPioStateMachine() {
  PIO selectedPio = pio0;
  int selectedSm = pio_claim_unused_sm(selectedPio, false);
  if (selectedSm < 0) {
    selectedPio = pio1;
    selectedSm = pio_claim_unused_sm(selectedPio, false);
    if (selectedSm < 0) {
      std::puts("[pdm-planar] no free PIO state machine");
      return false;
    }
  }

  if (!pio_can_add_program(selectedPio, &mmpr_pdm_rx_program)) {
    pio_sm_unclaim(selectedPio, selectedSm);
    std::puts("[pdm-planar] no room for PDM PIO program");
    return false;
  }

  // D2: clk_sys / (2 * chip_rate) must be a whole number of SM cycles per
  // half PDM period (50 for the 153.6 MHz bench overclock, 40 for the final
  // 122.88 MHz TCXO board). Y = cyclesPerHalf - 3 (see mmpr_pdm_rx.pio).
  const double clkSysHz = static_cast<double>(clock_get_hz(clk_sys));
  const double cyclesPerHalfExact = clkSysHz / (2.0 * kChipRateHz);
  const uint32_t cyclesPerHalf = static_cast<uint32_t>(cyclesPerHalfExact + 0.5);
  const double residual = cyclesPerHalfExact - static_cast<double>(cyclesPerHalf);
  if (std::abs(residual) > 1e-6) {
    std::printf("[pdm-planar] WARNING: clk_sys=%.0f Hz is not an exact multiple of 2x chip rate "
                "(cyclesPerHalf=%.6f) -- PDM clock will not be jitter-free (D2)\n",
                clkSysHz, cyclesPerHalfExact);
  }
  if (cyclesPerHalf < 3) {
    pio_sm_unclaim(selectedPio, selectedSm);
    std::puts("[pdm-planar] clk_sys too low for the PDM chip rate");
    return false;
  }
  const uint32_t y = cyclesPerHalf - 3;

  const uint programOffset = pio_add_program(selectedPio, &mmpr_pdm_rx_program);

  for (uint8_t i = 0; i < pins_.dataPinCount; ++i) {
    pio_gpio_init(selectedPio, pins_.dataPinBase + i);
    gpio_set_dir(pins_.dataPinBase + i, GPIO_IN);
    // Erratum RP2350-E9 (HARDWARE_REVIEW.md): never rely on the internal
    // pull-down here -- external ~10k pull-downs on GP1-3 are required.
    gpio_disable_pulls(pins_.dataPinBase + i);
  }
  pio_gpio_init(selectedPio, pins_.clockPin);
  gpio_set_dir(pins_.clockPin, GPIO_OUT);
  gpio_put(pins_.clockPin, 0);

  pio_sm_config smCfg = mmpr_pdm_rx_program_get_default_config(programOffset);
  sm_config_set_sideset_pins(&smCfg, pins_.clockPin);
  sm_config_set_in_pins(&smCfg, pins_.dataPinBase);
  sm_config_set_in_shift(&smCfg, /*shift_right=*/false, /*autopush=*/true, /*push_threshold=*/24);
  sm_config_set_fifo_join(&smCfg, PIO_FIFO_JOIN_RX);
  sm_config_set_clkdiv(&smCfg, 1.0f);

  pio_sm_set_consecutive_pindirs(selectedPio, selectedSm, pins_.dataPinBase, pins_.dataPinCount, false);
  pio_sm_set_consecutive_pindirs(selectedPio, selectedSm, pins_.clockPin, 1, true);

  pio_sm_init(selectedPio, selectedSm, programOffset, &smCfg);
  pio_sm_clear_fifos(selectedPio, selectedSm);
  pio_sm_restart(selectedPio, selectedSm);

  // Preload Y with the settle-cycle count (mirrors mmpr_sirith_tdm_in's
  // Y-preload idiom in mmpr_audio_rx.pio).
  pio_sm_put_blocking(selectedPio, selectedSm, y);
  pio_sm_exec(selectedPio, selectedSm, pio_encode_pull(false, false));
  pio_sm_exec(selectedPio, selectedSm, pio_encode_mov(pio_y, pio_osr));

  pio_ = selectedPio;
  sm_ = selectedSm;
  offset_ = programOffset;
  programInstalled_ = true;

  std::printf("[pdm-planar] PIO started pio=%d sm=%d offset=%u clk_sys=%.0fHz cyclesPerHalf=%u\n",
              (selectedPio == pio0) ? 0 : 1, selectedSm, programOffset, clkSysHz, cyclesPerHalf);
  return true;
}

void PdmPlanarSource::deinitPioStateMachine() {
  if (pio_ == nullptr) return;
  PIO pio = reinterpret_cast<PIO>(pio_);
  if (sm_ >= 0) {
    pio_sm_set_enabled(pio, sm_, false);
    pio_sm_unclaim(pio, sm_);
  }
  if (programInstalled_) {
    pio_remove_program(pio, &mmpr_pdm_rx_program, offset_);
  }
  gpio_put(pins_.clockPin, 0);
  gpio_set_dir(pins_.clockPin, GPIO_IN);
  pio_ = nullptr;
  sm_ = -1;
  offset_ = 0;
  programInstalled_ = false;
}

bool PdmPlanarSource::initDmaCapture() {
  if (pio_ == nullptr || sm_ < 0) return false;

  rawWordRing_ = new (std::nothrow) uint32_t[kPdmRawBlockRingSize * kPdmRawWordsPerBlock];
  if (rawWordRing_ == nullptr) {
    std::puts("[pdm-planar] unable to allocate raw word ring");
    return false;
  }

  dmaChannel_ = dma_claim_unused_channel(false);
  if (dmaChannel_ < 0) {
    std::puts("[pdm-planar] no free DMA channel");
    return false;
  }

  dma_channel_config dmaCfg = dma_channel_get_default_config(static_cast<uint>(dmaChannel_));
  channel_config_set_transfer_data_size(&dmaCfg, DMA_SIZE_32);
  channel_config_set_read_increment(&dmaCfg, false);
  channel_config_set_write_increment(&dmaCfg, true);
  channel_config_set_dreq(&dmaCfg, PIO_DREQ_NUM(reinterpret_cast<PIO>(pio_), sm_, false));

  rawBlockWriteIndex_ = 0;
  rawBlockReadIndex_ = 0;
  rawBlockDroppedCount_ = 0;
  nextRawWordIndex_ = 0;
  completedBlockCount_ = 0;

  dma_channel_configure(
      static_cast<uint>(dmaChannel_),
      &dmaCfg,
      rawWordRing_,
      &reinterpret_cast<PIO>(pio_)->rxf[sm_],
      static_cast<uint32_t>(kPdmRawWordsPerBlock),
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

void PdmPlanarSource::deinitDmaCapture() {
  if (dmaChannel_ >= 0) {
    irq_set_enabled(DMA_IRQ_0, false);
    dma_channel_set_irq0_enabled(static_cast<uint>(dmaChannel_), false);
    dma_channel_acknowledge_irq0(static_cast<uint>(dmaChannel_));
    dma_channel_abort(static_cast<uint>(dmaChannel_));
    dma_channel_unclaim(static_cast<uint>(dmaChannel_));
    dmaChannel_ = -1;
  }
  if (gActiveDmaSource == this) gActiveDmaSource = nullptr;
}

bool PdmPlanarSource::begin() {
  deinitDmaCapture();
  deinitPioStateMachine();

  if (!validateConfig()) {
    std::puts("[pdm-planar] invalid config");
    return false;
  }

  frameRing_ = new (std::nothrow)
      int16_t[static_cast<size_t>(config_.ringFrames) * config_.frameSamples * PdmCicDecimator::maxChannels()];
  frameStartSampleIndex_ = new (std::nothrow) uint64_t[config_.ringFrames];
  if (frameRing_ == nullptr || frameStartSampleIndex_ == nullptr) {
    std::puts("[pdm-planar] unable to allocate frame ring");
    return false;
  }
  frameWriteIndex_ = 0;
  frameReadIndex_ = 0;
  frameAvailableCount_ = 0;
  ringFramesHighWater_ = 0;

  if (!initPioStateMachine()) return false;
  if (!initDmaCapture()) {
    deinitPioStateMachine();
    return false;
  }

  // Core 1 is a pure DSP consumer (D5): it only ever touches rawWordRing_
  // (read-only, produced by the core-0 IRQ), decimator_, and frameRing_
  // (write-only from core 1's side). It never touches DMA/PIO/IRQ state.
  multicore_launch_core1([]() {
    if (gActiveDmaSource != nullptr) {
      gActiveDmaSource->core1DecodeLoop();
    }
  });

  initialized_ = true;
  return true;
}

void PdmPlanarSource::onDmaIrq() {
  if (dmaChannel_ < 0 || !dma_channel_get_irq0_status(static_cast<uint>(dmaChannel_))) return;
  dma_channel_acknowledge_irq0(static_cast<uint>(dmaChannel_));

  const uint32_t completedBlock = rawBlockWriteIndex_;
  blockStartRawWordIndex_[completedBlock] = nextRawWordIndex_;
  blockEndMonotonicUs_[completedBlock] = time_us_64();
  nextRawWordIndex_ += kPdmRawWordsPerBlock;
  ++completedBlockCount_;

  const uint32_t nextWrite = (rawBlockWriteIndex_ + 1u) % kPdmRawBlockRingSize;
  // If core 1 hasn't kept up, drop the oldest unread block rather than
  // overrunning it (same policy as SirithPicoTdmSource's frame ring).
  if (nextWrite == rawBlockReadIndex_) {
    rawBlockReadIndex_ = (rawBlockReadIndex_ + 1u) % kPdmRawBlockRingSize;
    ++rawBlockDroppedCount_;
  }
  rawBlockWriteIndex_ = nextWrite;

  dma_channel_set_write_addr(
      static_cast<uint>(dmaChannel_),
      rawWordRing_ + (static_cast<size_t>(rawBlockWriteIndex_) * kPdmRawWordsPerBlock),
      false);
  dma_channel_set_trans_count(static_cast<uint>(dmaChannel_), static_cast<uint32_t>(kPdmRawWordsPerBlock), true);

  // SIO doorbell: tell core 1 a new block is ready. Non-blocking -- core 1
  // is expected to keep up; if the inter-core FIFO is momentarily full we
  // drop the doorbell rather than stalling this IRQ (core 1 will simply
  // notice fewer signals than blocks and can poll rawBlockReadIndex_ vs
  // rawBlockWriteIndex_ as a backstop -- see core1DecodeLoop()).
  multicore_fifo_push_timeout_us(completedBlock, 0);
}

void PdmPlanarSource::sDmaIrq() {
  if (gActiveDmaSource != nullptr) gActiveDmaSource->onDmaIrq();
}

void PdmPlanarSource::core1DecodeLoop() {
  // Pure DSP consumer (D5): no DMA/PIO/IRQ access from here, ever.
  const size_t maxOutPerBlock = kPdmRawWordsPerBlock;  // >= 1 decimated sample per 16 words
  int16_t* stagingInterleaved = new int16_t[maxOutPerBlock * PdmCicDecimator::maxChannels()];
  int16_t* frameStaging = new int16_t[config_.frameSamples * PdmCicDecimator::maxChannels()];
  size_t stagingFill = 0;
  uint64_t totalDecimatedSamples = 0;
  uint64_t frameStartIndex = 0;

  while (true) {
    uint32_t blockIndex = 0;
    // The doorbell is the primary signal; if it was dropped (FIFO full),
    // fall back to polling the read/write indices so core 1 never stalls
    // permanently on a lost doorbell.
    if (multicore_fifo_pop_timeout_us(2000, &blockIndex)) {
      // blockIndex from the doorbell is authoritative.
    } else if (rawBlockReadIndex_ == rawBlockWriteIndex_) {
      continue;  // nothing pending
    } else {
      blockIndex = rawBlockReadIndex_;
    }

    const uint32_t* block = rawWordRing_ + (static_cast<size_t>(blockIndex) * kPdmRawWordsPerBlock);
    const size_t produced =
        decimator_.processRawWords(block, kPdmRawWordsPerBlock, stagingInterleaved, maxOutPerBlock);
    rawBlockReadIndex_ = (blockIndex + 1u) % kPdmRawBlockRingSize;

    for (size_t s = 0; s < produced; ++s) {
      for (size_t ch = 0; ch < PdmCicDecimator::maxChannels(); ++ch) {
        frameStaging[(stagingFill * PdmCicDecimator::maxChannels()) + ch] =
            stagingInterleaved[(s * PdmCicDecimator::maxChannels()) + ch];
      }
      if (stagingFill == 0) frameStartIndex = totalDecimatedSamples;
      ++stagingFill;
      ++totalDecimatedSamples;

      if (stagingFill == config_.frameSamples) {
        const uint32_t writeSlot = frameWriteIndex_;
        int16_t* dst = frameRing_ + (static_cast<size_t>(writeSlot) * config_.frameSamples *
                                      PdmCicDecimator::maxChannels());
        for (size_t i = 0; i < config_.frameSamples * PdmCicDecimator::maxChannels(); ++i) {
          dst[i] = frameStaging[i];
        }
        frameStartSampleIndex_[writeSlot] = frameStartIndex;

        const uint32_t nextWrite = (writeSlot + 1u) % config_.ringFrames;
        const uint32_t irqState = save_and_disable_interrupts();
        if (frameAvailableCount_ == config_.ringFrames) {
          frameReadIndex_ = (frameReadIndex_ + 1u) % config_.ringFrames;
        } else {
          ++frameAvailableCount_;
          if (frameAvailableCount_ > ringFramesHighWater_) ringFramesHighWater_ = frameAvailableCount_;
        }
        restore_interrupts(irqState);
        frameWriteIndex_ = nextWrite;
        stagingFill = 0;
      }
    }
  }
}

bool PdmPlanarSource::readFrame(
    int16_t* interleavedOut,
    size_t samplesPerChannel,
    AudioCaptureTimestamp* captureTimestamp) {
  if (!initialized_ || interleavedOut == nullptr || samplesPerChannel != config_.frameSamples) return false;

  uint32_t readSlot = 0;
  while (true) {
    const uint32_t irqState = save_and_disable_interrupts();
    if (frameAvailableCount_ == 0) {
      restore_interrupts(irqState);
      tight_loop_contents();
      continue;
    }
    readSlot = frameReadIndex_;
    frameReadIndex_ = (frameReadIndex_ + 1u) % config_.ringFrames;
    --frameAvailableCount_;
    restore_interrupts(irqState);
    break;
  }

  const int16_t* src =
      frameRing_ + (static_cast<size_t>(readSlot) * config_.frameSamples * PdmCicDecimator::maxChannels());
  std::copy(src, src + (config_.frameSamples * PdmCicDecimator::maxChannels()), interleavedOut);

  if (captureTimestamp != nullptr) {
    const uint64_t startIdx = frameStartSampleIndex_[readSlot];
    captureTimestamp->startSampleIndex = startIdx;
    captureTimestamp->endSampleIndex = startIdx + config_.frameSamples;
    captureTimestamp->blockEndMonotonicUs = time_us_64();
    const uint64_t frameDurationUs =
        (static_cast<uint64_t>(config_.frameSamples) * 1000000ull) / config_.sampleRateHz;
    captureTimestamp->blockStartMonotonicUs =
        captureTimestamp->blockEndMonotonicUs >= frameDurationUs
            ? captureTimestamp->blockEndMonotonicUs - frameDurationUs
            : 0;
    captureTimestamp->completedBlockCount = startIdx / config_.frameSamples;
    captureTimestamp->dmaRingSlotIndex = readSlot;
    captureTimestamp->droppedBlocksBeforeCapture = rawBlockDroppedCount_;
  }
  return true;
}

uint32_t PdmPlanarSource::availableFrames() const {
  if (!initialized_) return 0;
  const uint32_t irqState = save_and_disable_interrupts();
  const uint32_t n = frameAvailableCount_;
  restore_interrupts(irqState);
  return n;
}

bool PdmPlanarSource::readFrameNonblocking(
    int16_t* interleavedOut,
    size_t samplesPerChannel,
    AudioCaptureTimestamp* captureTimestamp) {
  if (availableFrames() == 0) return false;
  return readFrame(interleavedOut, samplesPerChannel, captureTimestamp);
}

bool PdmPlanarSource::snapshotProducerState(
    AudioProducerSnapshot& producerSnapshot,
    bool callerAlreadyInIrqContext) const {
  producerSnapshot = {};
  if (!initialized_ || dmaChannel_ < 0) return false;

  // D5: everything read here is core-0-only state (DMA hardware registers
  // and the nextRawWordIndex_/blockStart bookkeeping onDmaIrq() maintains).
  // Safe to call from a core-0 IRQ context (GpsPpsTimerCapture::onIrq()).
  const uint32_t irqState = callerAlreadyInIrqContext ? 0u : save_and_disable_interrupts();
  const uint64_t blockStartWordIndex = nextRawWordIndex_;
  const uint64_t completedBlocks = completedBlockCount_;
  dma_channel_hw_t* channelHw = dma_channel_hw_addr(static_cast<uint>(dmaChannel_));
  const uint32_t wordsRemaining = channelHw->transfer_count;
  if (!callerAlreadyInIrqContext) restore_interrupts(irqState);

  const uint32_t clampedRemaining = std::min<uint32_t>(wordsRemaining, static_cast<uint32_t>(kPdmRawWordsPerBlock));
  const uint32_t wordsTransferred = static_cast<uint32_t>(kPdmRawWordsPerBlock) - clampedRemaining;

  // D4: raw word index -> per-channel chip index (x4, one chip/channel/word
  // -period... each word is 4 periods) -> decimated sample index is an
  // exact /64 (32 CIC x 2 halfband). Then subtract the constant group delay
  // so the reported position is acoustic-arrival-time, not capture-time.
  const double totalWords = static_cast<double>(blockStartWordIndex) + static_cast<double>(wordsTransferred);
  const double chipIndex = totalWords * 4.0;
  const double decimatedSamplePosition = chipIndex / static_cast<double>(PdmCicDecimator::totalDecimation());
  const double groupDelaySamples =
      PdmCicDecimator::groupDelayMicroseconds() * 1e-6 * static_cast<double>(config_.sampleRateHz);

  producerSnapshot.valid = true;
  producerSnapshot.capturedMonotonicUs = time_us_64();
  producerSnapshot.samplePosition = decimatedSamplePosition - groupDelaySamples;
  producerSnapshot.completedBlockCount = completedBlocks;
  producerSnapshot.dmaRingSlotIndex = rawBlockWriteIndex_;
  producerSnapshot.wordsTransferredInActiveBlock = wordsTransferred;
  producerSnapshot.wordsRemainingInActiveBlock = clampedRemaining;
  return true;
}

}  // namespace mmpr
