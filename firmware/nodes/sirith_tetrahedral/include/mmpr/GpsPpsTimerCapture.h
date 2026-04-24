#pragma once

#include <cstdint>

#include "mmpr/IAudioSource.h"

namespace mmpr {

struct GpsPpsCaptureEvent {
  uint32_t edgeCount = 0;
  uint64_t monotonicUs = 0;
  uint64_t tickCycles = 0;
  AudioProducerSnapshot audioProducerSnapshot = {};
};

class GpsPpsTimerCapture {
 public:
  ~GpsPpsTimerCapture();

  bool begin(int gpioPin);
  bool configured() const { return configured_; }
  void bindAudioSource(IAudioSource* audioSource);
  bool consumeNext(GpsPpsCaptureEvent& outEvent);

 private:
  static constexpr uint32_t kQueuedEvents = 8;

  bool initPioStateMachine();
  void deinitPioStateMachine();
  void onIrq();
  static uint64_t ticksToUs(uint64_t tickCycles, uint32_t clkSysHz);
  static void pio0IrqHandler();
  static void pio1IrqHandler();

  static constexpr uint32_t kCounterReload = 0xffffffffu;
  static constexpr uint32_t kCyclesPerLoop = 2u;
  // The PIO edge path spends a fixed 3 extra cycles in mov/push/mov after the
  // rising edge before the loop resumes. We do not compensate for that here
  // because it is a constant offset across nodes, so it cancels in PPS interval
  // estimation and inter-node TDOA comparisons.
  static constexpr uint32_t kFixedEdgePipelineCycles = 3u;

  int gpioPin_ = -1;
  bool configured_ = false;
  void* pio_ = nullptr;
  int sm_ = -1;
  uint32_t offset_ = 0;
  bool programInstalled_ = false;
  uint32_t clkSysHz_ = 0;
  uint64_t monotonicBaseUs_ = 0;
  uint64_t accumulatedTickCycles_ = 0;
  uint32_t observedEdgeCount_ = 0;
  IAudioSource* audioSource_ = nullptr;
  GpsPpsCaptureEvent queuedEvents_[kQueuedEvents] = {};
  volatile uint32_t queueHead_ = 0;
  volatile uint32_t queueTail_ = 0;
  int irqIndex_ = -1;
  int irqNumber_ = -1;
};

}  // namespace mmpr
