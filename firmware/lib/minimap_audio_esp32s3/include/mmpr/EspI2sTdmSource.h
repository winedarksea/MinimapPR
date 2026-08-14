#pragma once

// ESP32-S3 4-channel TDM capture (IAudioSource), the S3 counterpart of
// SirithPicoTdmSource.
//
// The S3's I2S peripheral declares SOC_I2S_SUPPORTS_TDM, so the PIO program the
// RP2350 needs (mmpr_audio_rx.pio) is replaced by esp_driver_i2s in TDM RX
// master mode: the S3 drives BCLK and FSYNC and samples SDATA, exactly like the
// PIO did.
//
// PCM PARITY: slots are read as raw 32-bit words (data_bit_width = 32) and
// converted by the shared decodeLeftJustifiedSlotToPcm16() in
// mmpr/PcmSlotDecode.h -- the same function the Pico path uses -- rather than by
// asking the driver for 24-bit samples. That is what makes this node's PCM
// byte-identical to the Pico's on the same acoustic input, which is the whole
// premise of the A/B frame-loss comparison.
//
// FRAMING: the ADAU7112 waveform the PIO produced is one 1-BCLK-wide positive
// FSYNC pulse per frame, then low (mmpr_audio_rx.pio sideset bit0 = BCLK,
// bit1 = FSYNC) -- that is I2S_TDM_PCM_SHORT: ws_width = 1, ws_pol = true,
// bit_shift = true. Those four knobs are config fields rather than constants
// because framing is the single most likely thing to be off by one bit against
// real hardware, mirroring how the Pico exposes kTdmCaptureBitOffset /
// kTdmSampleEdge.
//
// TASK MODEL: begin() starts a capture task that blocks in i2s_channel_read,
// decodes slots, and pushes complete frames into a lock-free-ish ring guarded by
// a spinlock. That task is the S3's equivalent of the Pico's DMA IRQ, which is
// what keeps availableFrames() / ringFramesHighWater() -- the counters the whole
// measurement rests on -- meaning the same thing on both nodes. NodeRunner stays
// single-threaded on the consumer side.
//
// NOT IMPLEMENTED: snapshotProducerState() returns false. The IDF I2S driver
// does not expose the live in-descriptor DMA transfer count that the Pico reads
// for sub-sample PPS alignment, so GPS/PPS discipline is out of scope for this
// node (see the plan's "GPS PPS discipline -- deferred" section for the three
// ways to fill that gap). Block-boundary accuracy comes from stamping
// esp_timer_get_time() in the driver's on_recv callback rather than at read()
// return.

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "driver/i2s_tdm.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "mmpr/IAudioSource.h"

namespace mmpr {

struct EspI2sTdmPins {
  int bclk = -1;    // BCLK output (S3 is master)
  int ws = -1;      // FSYNC/WS output
  int din = -1;     // SDATA input
  // Loopback transmit pin. When >= 0 a TX channel is allocated on the same I2S
  // port, which the IDF driver pairs into full duplex with the RX channel (same
  // BCLK/WS). Jumper this pin to `din` and the board can self-test its own TDM
  // framing and channel ordering with no ADAU7112 present -- see
  // EspI2sTdmLoopback.h. Set to -1 for normal capture.
  int loopbackDout = -1;
};

struct EspI2sTdmConfig {
  uint32_t sampleRateHz = 32000;
  size_t frameSamples = 512;
  uint8_t tdmSlots = 4;
  uint8_t slotBits = 32;
  uint8_t validBits = 24;
  // The ESP32-S3 I2S peripheral absorbs the ADAU7112's 1-BCLK I2S-style delay in
  // hardware when bit_shift is set, so unlike the Pico (which needs +1) this is
  // 0 by default. Kept as a knob for bench work.
  int8_t captureBitOffset = 0;

  // --- Framing (see the header comment) ---
  uint32_t wsWidth = 1;
  bool wsPolarityHigh = true;
  bool bitShift = true;
  bool leftAlign = false;

  // --- DMA ---
  // One descriptor is capped at 4092 bytes and a frame is tdmSlots * 4 B, so at
  // 4 slots the largest legal power-of-two is 128 frames (2048 B). dmaDescNum
  // descriptors is dmaDescNum * dmaFrameNum / sampleRate of hardware buffering
  // ahead of the capture task -- 64 ms at the defaults.
  uint32_t dmaFrameNum = 128;
  uint32_t dmaDescNum = 16;

  // Software ring, in whole audio frames. Sized from kTargetRingMs by the node
  // config, the same derivation the Pico uses.
  uint32_t ringFrames = 13;

  // Output channel index -> TDM slot index, from the ADAU7112 strap wiring.
  uint8_t outputChannelToSlot[4] = {1, 0, 3, 2};

  int i2sPort = 0;
  int captureTaskCoreId = 1;
  uint32_t captureTaskPriority = 20;
  uint32_t captureTaskStackBytes = 4096;
};

class EspI2sTdmSource final : public IAudioSource {
 public:
  EspI2sTdmSource(const EspI2sTdmPins& pins, const EspI2sTdmConfig& config);
  ~EspI2sTdmSource() override;

  EspI2sTdmSource(const EspI2sTdmSource&) = delete;
  EspI2sTdmSource& operator=(const EspI2sTdmSource&) = delete;

  bool begin() override;
  uint32_t sampleRateHz() const override { return config_.sampleRateHz; }
  uint8_t channels() const override { return config_.tdmSlots; }
  size_t frameSamples() const override { return config_.frameSamples; }
  AudioSourceType sourceType() const override { return AudioSourceType::kTdm; }

  bool readFrame(
      int16_t* interleavedOut,
      size_t samplesPerChannel,
      AudioCaptureTimestamp* captureTimestamp = nullptr) override;
  bool readFrameNonblocking(
      int16_t* interleavedOut,
      size_t samplesPerChannel,
      AudioCaptureTimestamp* captureTimestamp = nullptr) override;
  uint32_t availableFrames() const override;
  uint32_t ringFramesCapacity() const override { return config_.ringFrames; }
  uint32_t ringFramesHighWater() const override { return ringFramesHighWater_; }

  // Live configuration, so the loopback self-test derives its expected PCM
  // from the same slotBits/validBits/captureBitOffset the decoder uses
  // instead of hardcoding values that could drift out of agreement.
  const EspI2sTdmConfig& config() const { return config_; }

  // Full-duplex TX handle, non-null only when pins.loopbackDout >= 0.
  i2s_chan_handle_t loopbackTxHandle() const { return txHandle_; }
  // Bytes per TDM frame on the wire (tdmSlots * slotBits/8), for the loopback
  // feeder to size its pattern buffer.
  size_t rawFrameBytes() const { return static_cast<size_t>(config_.tdmSlots) * (config_.slotBits / 8u); }

  // Blocks that the DMA/driver dropped because the ring was full, cumulative.
  // Reported to NodeRunner via AudioCaptureTimestamp::droppedBlocksBeforeCapture.
  uint32_t droppedFrameCount() const { return droppedFrameCount_; }
  // Count of on_recv_q_ovf events -- the driver's own overflow signal, which
  // fires before the software ring can even see the data.
  uint32_t driverOverflowCount() const { return driverOverflowCount_.load(std::memory_order_relaxed); }

 private:
  struct RingSlotMeta {
    uint64_t startSampleIndex = 0;
    uint64_t blockEndMonotonicUs = 0;
    uint64_t completedBlockCount = 0;
  };

  static void captureTaskEntry(void* context);
  static bool onRecvIsr(i2s_chan_handle_t handle, i2s_event_data_t* event, void* context);
  static bool onRecvOverflowIsr(i2s_chan_handle_t handle, i2s_event_data_t* event, void* context);

  void captureLoop();
  bool configureChannels();
  bool takeFrame(
      int16_t* interleavedOut,
      size_t samplesPerChannel,
      AudioCaptureTimestamp* captureTimestamp);

  EspI2sTdmPins pins_;
  EspI2sTdmConfig config_;

  i2s_chan_handle_t rxHandle_ = nullptr;
  i2s_chan_handle_t txHandle_ = nullptr;
  TaskHandle_t captureTask_ = nullptr;
  SemaphoreHandle_t frameReady_ = nullptr;

  bool initialized_ = false;
  size_t samplesPerFrameAllChannels_ = 0;
  size_t chunkFrames_ = 0;
  size_t chunkBytes_ = 0;
  uint64_t frameDurationUs_ = 0;

  uint32_t* rawChunk_ = nullptr;
  int16_t* ring_ = nullptr;
  RingSlotMeta* ringMeta_ = nullptr;

  // Producer-side (capture task) state.
  size_t stagedSamples_ = 0;
  uint64_t nextProducedStartSampleIndex_ = 0;
  uint64_t nextCompletedBlockCount_ = 0;
  uint32_t consumedDescriptors_ = 0;

  // Descriptor-EOF timestamps written from the driver ISR. Sized to dmaDescNum
  // at begin(); each entry carries the descriptor counter it belongs to so a
  // lost descriptor is detected instead of silently mis-stamping a frame. The
  // counter is the release/acquire handle for the pair of plain array writes.
  uint64_t* eofMonotonicUs_ = nullptr;
  uint32_t* eofSequence_ = nullptr;
  std::atomic<uint32_t> eofCount_{0};

  // Ring indices, guarded by ringLock_.
  uint32_t writeIndex_ = 0;
  uint32_t readIndex_ = 0;
  uint32_t filledFrames_ = 0;
  uint32_t ringFramesHighWater_ = 0;
  uint32_t droppedFrameCount_ = 0;
  uint32_t reportedDroppedFrameCount_ = 0;
  std::atomic<uint32_t> driverOverflowCount_{0};
  portMUX_TYPE ringLock_ = portMUX_INITIALIZER_UNLOCKED;
};

}  // namespace mmpr
