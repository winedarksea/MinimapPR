#include "mmpr/GpsPpsTimerCapture.h"

#include <hardware/irq.h>
#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "hardware/pio.h"
#include "hardware/sync.h"
#include "pico/time.h"

#include "gps_pps_capture.pio.h"

namespace mmpr {
namespace {

GpsPpsTimerCapture* gPio0Capture = nullptr;
GpsPpsTimerCapture* gPio1Capture = nullptr;

bool runningInExceptionContext() {
  uint32_t ipsr = 0;
  __asm volatile("mrs %0, ipsr" : "=r"(ipsr));
  return ipsr != 0;
}

}  // namespace

GpsPpsTimerCapture::~GpsPpsTimerCapture() {
  deinitPioStateMachine();
}

bool GpsPpsTimerCapture::begin(int gpioPin) {
  deinitPioStateMachine();

  if (gpioPin < 0) {
    configured_ = false;
    return false;
  }

  gpioPin_ = gpioPin;
  clkSysHz_ = clock_get_hz(clk_sys);
  monotonicBaseUs_ = time_us_64();
  accumulatedTickCycles_ = 0;
  observedEdgeCount_ = 0;
  queueHead_ = 0;
  queueTail_ = 0;
  configured_ = initPioStateMachine();
  return configured_;
}

void GpsPpsTimerCapture::bindAudioSource(IAudioSource* audioSource) {
  const uint32_t irqState = save_and_disable_interrupts();
  audioSource_ = audioSource;
  restore_interrupts(irqState);
}

bool GpsPpsTimerCapture::consumeNext(GpsPpsCaptureEvent& outEvent) {
  if (!configured_) {
    return false;
  }

  const uint32_t irqState = save_and_disable_interrupts();
  if (queueHead_ == queueTail_) {
    restore_interrupts(irqState);
    return false;
  }
  outEvent = queuedEvents_[queueTail_];
  queueTail_ = (queueTail_ + 1u) % kQueuedEvents;
  restore_interrupts(irqState);
  return true;
}

bool GpsPpsTimerCapture::initPioStateMachine() {
  PIO selectedPio = pio1;
  int selectedSm = pio_claim_unused_sm(selectedPio, false);
  if (selectedSm < 0) {
    selectedPio = pio0;
    selectedSm = pio_claim_unused_sm(selectedPio, false);
    if (selectedSm < 0) {
      return false;
    }
  }

  if (!pio_can_add_program(selectedPio, &gps_pps_capture_program)) {
    pio_sm_unclaim(selectedPio, static_cast<uint>(selectedSm));
    return false;
  }

  const uint32_t programOffset = pio_add_program(selectedPio, &gps_pps_capture_program);
  pio_gpio_init(selectedPio, static_cast<uint>(gpioPin_));
  gpio_set_dir(static_cast<uint>(gpioPin_), GPIO_IN);
  gpio_pull_down(static_cast<uint>(gpioPin_));

  pio_sm_config smConfig = gps_pps_capture_program_get_default_config(programOffset);
  sm_config_set_jmp_pin(&smConfig, static_cast<uint>(gpioPin_));
  sm_config_set_in_pins(&smConfig, static_cast<uint>(gpioPin_));
  sm_config_set_clkdiv(&smConfig, 1.0f);
  sm_config_set_fifo_join(&smConfig, PIO_FIFO_JOIN_RX);

  pio_sm_set_consecutive_pindirs(selectedPio, static_cast<uint>(selectedSm), static_cast<uint>(gpioPin_), 1, false);
  pio_sm_init(selectedPio, static_cast<uint>(selectedSm), programOffset, &smConfig);
  pio_sm_clear_fifos(selectedPio, static_cast<uint>(selectedSm));
  pio_sm_restart(selectedPio, static_cast<uint>(selectedSm));
  pio_sm_put_blocking(selectedPio, static_cast<uint>(selectedSm), kCounterReload);
  pio_sm_exec(selectedPio, static_cast<uint>(selectedSm), pio_encode_pull(false, false));
  pio_sm_exec(selectedPio, static_cast<uint>(selectedSm), pio_encode_mov(pio_y, pio_osr));
  pio_sm_exec(selectedPio, static_cast<uint>(selectedSm), pio_encode_mov(pio_x, pio_y));
  pio_sm_set_enabled(selectedPio, static_cast<uint>(selectedSm), true);

  irqIndex_ = 0;
  irqNumber_ = (selectedPio == pio0) ? PIO0_IRQ_0 : PIO1_IRQ_0;
  if (selectedPio == pio0) {
    gPio0Capture = this;
    irq_set_exclusive_handler(PIO0_IRQ_0, pio0IrqHandler);
  } else {
    gPio1Capture = this;
    irq_set_exclusive_handler(PIO1_IRQ_0, pio1IrqHandler);
  }
  pio_set_irq0_source_enabled(
      selectedPio,
      static_cast<enum pio_interrupt_source>(pis_sm0_rx_fifo_not_empty + selectedSm),
      true);
  irq_set_enabled(static_cast<irq_num_t>(irqNumber_), true);

  pio_ = selectedPio;
  sm_ = selectedSm;
  offset_ = programOffset;
  programInstalled_ = true;
  return true;
}

void GpsPpsTimerCapture::deinitPioStateMachine() {
  if (pio_ == nullptr || sm_ < 0) {
    configured_ = false;
    return;
  }

  PIO pio = reinterpret_cast<PIO>(pio_);
  if (irqNumber_ >= 0) {
    irq_set_enabled(static_cast<irq_num_t>(irqNumber_), false);
  }
  if (irqIndex_ == 0) {
    pio_set_irq0_source_enabled(
        pio,
        static_cast<enum pio_interrupt_source>(pis_sm0_rx_fifo_not_empty + sm_),
        false);
  }
  if (pio == pio0 && gPio0Capture == this) {
    irq_remove_handler(PIO0_IRQ_0, pio0IrqHandler);
    gPio0Capture = nullptr;
  } else if (pio == pio1 && gPio1Capture == this) {
    irq_remove_handler(PIO1_IRQ_0, pio1IrqHandler);
    gPio1Capture = nullptr;
  }
  pio_sm_set_enabled(pio, static_cast<uint>(sm_), false);
  pio_sm_unclaim(pio, static_cast<uint>(sm_));
  if (programInstalled_) {
    pio_remove_program(pio, &gps_pps_capture_program, offset_);
  }

  pio_ = nullptr;
  sm_ = -1;
  offset_ = 0;
  programInstalled_ = false;
  configured_ = false;
  irqIndex_ = -1;
  irqNumber_ = -1;
}

uint64_t GpsPpsTimerCapture::ticksToUs(uint64_t tickCycles, uint32_t clkSysHz) {
  if (clkSysHz == 0) {
    return 0;
  }
  return (tickCycles * 1000000ULL) / static_cast<uint64_t>(clkSysHz);
}

void GpsPpsTimerCapture::onIrq() {
  if (!configured_ || pio_ == nullptr || sm_ < 0) {
    return;
  }

  PIO pio = reinterpret_cast<PIO>(pio_);
  while (!pio_sm_is_rx_fifo_empty(pio, static_cast<uint>(sm_))) {
    const uint32_t remainingTicks = pio_sm_get(pio, static_cast<uint>(sm_));
    const uint64_t elapsedLoops = static_cast<uint64_t>(kCounterReload - remainingTicks);
    accumulatedTickCycles_ += elapsedLoops * static_cast<uint64_t>(kCyclesPerLoop);
    ++observedEdgeCount_;

    GpsPpsCaptureEvent nextEvent = {};
    nextEvent.edgeCount = observedEdgeCount_;
    nextEvent.monotonicUs = monotonicBaseUs_ + ticksToUs(accumulatedTickCycles_ + kFixedEdgePipelineCycles, clkSysHz_);
    nextEvent.tickCycles = accumulatedTickCycles_;
    if (audioSource_ != nullptr) {
      nextEvent.audioProducerSnapshot.valid =
          audioSource_->snapshotProducerState(nextEvent.audioProducerSnapshot, runningInExceptionContext());
    }

    const uint32_t nextHead = (queueHead_ + 1u) % kQueuedEvents;
    if (nextHead == queueTail_) {
      queueTail_ = (queueTail_ + 1u) % kQueuedEvents;
    }
    queuedEvents_[queueHead_] = nextEvent;
    queueHead_ = nextHead;
  }
}

void GpsPpsTimerCapture::pio0IrqHandler() {
  if (gPio0Capture != nullptr) {
    gPio0Capture->onIrq();
  }
}

void GpsPpsTimerCapture::pio1IrqHandler() {
  if (gPio1Capture != nullptr) {
    gPio1Capture->onIrq();
  }
}

}  // namespace mmpr
