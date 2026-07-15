#pragma once

// PdmPlanarSource -- IAudioSource implementation for the sirith_planar 5-mic
// PDM array (Phase 2 in the firmware plan). Pattern-matched to
// SirithPicoTdmSource: PIO+DMA raw-word capture, core-0 IRQ producer
// accounting, snapshotProducerState() mapping the live DMA position to a
// decimated sample index.
//
// D5 (core split): the DMA IRQ and ALL timestamp/producer-state bookkeeping
// stay on core 0 -- GpsPpsTimerCapture::onIrq() (minimap_node_runtime) calls
// snapshotProducerState() directly from a PIO IRQ context on core 0, so that
// state must never depend on anything core 1 touches. Core 1 is a pure DSP
// consumer: it pops raw PDM word-block descriptors from an SPSC ring (core 0
// producer, core 1 consumer) signaled by the RP2350 inter-core SIO FIFO
// (pico_multicore), runs PdmCicDecimator on them, and writes decimated int16
// frames into a second SPSC ring (core 1 producer, core 0 consumer) that
// readFrame()/readFrameNonblocking() drain. Core 1 never touches DMA
// hardware, IRQs, or the producer-state fields core 0 uses for timestamps.
//
// D4 (timestamps): snapshotProducerState() converts the live raw-word DMA
// position to a decimated (48 kHz) sample index via an exact division by 64
// (32 CIC decimation x 2 halfband decimation), then subtracts
// PdmCicDecimator::groupDelayMicroseconds() so the reported sample position
// represents acoustic-arrival-time, not raw-capture-time.
//
// Hardware-dependent parts of this file (PIO program timing, DMA IRQ
// behavior, actual multicore hand-off latency) have not been bench-verified
// -- see PDM_DESIGN.md and HARDWARE_REVIEW.md.

#include <cstddef>
#include <cstdint>

#include "mmpr/IAudioSource.h"
#include "mmpr/PdmCicDecimator.h"

namespace mmpr {

struct PdmPlanarPins {
  uint8_t dataPinBase;  // GP1 (3 contiguous data pins: GP1..GP3)
  uint8_t dataPinCount;  // 3
  uint8_t clockPin;      // GP4
};

struct PdmPlanarConfig {
  uint32_t sampleRateHz = 48000;   // final (post-halfband) output rate
  size_t frameSamples = 512;       // samples per channel per readFrame()
  uint32_t ringFrames = 16;        // output (48kHz) frame ring depth
  bool enableDither = true;
};

// Raw PDM capture words per DMA block handed from core 0 to core 1. Kept
// small and a single named constant so it is easy to bump if core-1
// scheduling latency ever needs more slack (Phase 6 note in node_config.h
// about PSRAM headroom applies here too).
static constexpr size_t kPdmRawWordsPerBlock = 256;   // 256 words = 1024 PDM periods = 2048 chips/channel
static constexpr size_t kPdmRawBlockRingSize = 8;     // SPSC ring depth, core0->core1

class PdmPlanarSource final : public IAudioSource {
 public:
  PdmPlanarSource(const PdmPlanarPins& pins, const PdmPlanarConfig& config);
  ~PdmPlanarSource() override;

  bool begin() override;
  uint32_t sampleRateHz() const override { return config_.sampleRateHz; }
  uint8_t channels() const override { return static_cast<uint8_t>(PdmCicDecimator::maxChannels()); }
  size_t frameSamples() const override { return config_.frameSamples; }
  AudioSourceType sourceType() const override { return AudioSourceType::kPdmDirect; }

  bool readFrame(
      int16_t* interleavedOut,
      size_t samplesPerChannel,
      AudioCaptureTimestamp* captureTimestamp = nullptr) override;
  uint32_t availableFrames() const override;
  uint32_t ringFramesCapacity() const override { return config_.ringFrames; }
  uint32_t ringFramesHighWater() const override { return ringFramesHighWater_; }
  bool readFrameNonblocking(
      int16_t* interleavedOut,
      size_t samplesPerChannel,
      AudioCaptureTimestamp* captureTimestamp = nullptr) override;
  bool snapshotProducerState(
      AudioProducerSnapshot& producerSnapshot,
      bool callerAlreadyInIrqContext = false) const override;

  // Entry point for the core-1 launch (pico_multicore's multicore_launch_core1).
  // Not part of IAudioSource; called once from sirith_planar.cpp's core-1
  // trampoline. Loops forever consuming raw blocks and producing decimated
  // frames until core 1 is halted (which this firmware never does).
  void core1DecodeLoop();

 private:
  bool validateConfig() const;
  bool initPioStateMachine();
  bool initDmaCapture();
  void deinitDmaCapture();
  void deinitPioStateMachine();
  void onDmaIrq();
  static void sDmaIrq();

  PdmPlanarPins pins_;
  PdmPlanarConfig config_;

  bool initialized_ = false;

  void* pio_ = nullptr;
  int sm_ = -1;
  uint32_t offset_ = 0;
  bool programInstalled_ = false;
  int dmaChannel_ = -1;

  // Raw-word capture ring (core 0 produces via DMA IRQ, core 1 consumes).
  // Double-buffered DMA target: the DMA channel free-runs into
  // rawWordRing_[block], re-armed at IRQ time to the next block.
  uint32_t* rawWordRing_ = nullptr;  // [kPdmRawBlockRingSize][kPdmRawWordsPerBlock]
  volatile uint32_t rawBlockWriteIndex_ = 0;  // core 0 only
  volatile uint32_t rawBlockReadIndex_ = 0;   // core 1 only (core 0 reads for fill level)
  volatile uint32_t rawBlockDroppedCount_ = 0;

  // D4/D5: core-0-only producer/timestamp bookkeeping, updated in
  // onDmaIrq(). raw word index -> decimated sample index is an exact /64.
  volatile uint64_t nextRawWordIndex_ = 0;  // total words DMA'd so far
  volatile uint64_t blockStartRawWordIndex_[kPdmRawBlockRingSize] = {};
  volatile uint64_t blockEndMonotonicUs_[kPdmRawBlockRingSize] = {};
  volatile uint64_t completedBlockCount_ = 0;

  // Decimated (48 kHz) output frame ring: core 1 produces (via
  // PdmCicDecimator), core 0 consumes in readFrame(). Sized in whole
  // frameSamples-sized slots.
  int16_t* frameRing_ = nullptr;  // [ringFrames][frameSamples][channels]
  volatile uint32_t frameWriteIndex_ = 0;  // core 1 only
  volatile uint32_t frameReadIndex_ = 0;   // core 0 only
  volatile uint32_t frameAvailableCount_ = 0;
  volatile uint32_t ringFramesHighWater_ = 0;
  uint64_t* frameStartSampleIndex_ = nullptr;  // [ringFrames], allocated in begin()

  PdmCicDecimator decimator_;  // core-1-only after begin()
};

}  // namespace mmpr
