#include "mmpr/EspI2sTdmSource.h"

#include <atomic>
#include <cstring>
#include <new>

#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

#include "mmpr/PcmSlotDecode.h"
#include "mmpr/PlatformTime.h"

namespace mmpr {
namespace {

constexpr const char* kTag = "mmpr-i2s";

// One DMA descriptor may not exceed 4092 bytes on the S3's GDMA.
constexpr size_t kMaxDmaDescriptorBytes = 4092;

}  // namespace

EspI2sTdmSource::EspI2sTdmSource(const EspI2sTdmPins& pins, const EspI2sTdmConfig& config)
    : pins_(pins), config_(config) {}

EspI2sTdmSource::~EspI2sTdmSource() {
  if (captureTask_ != nullptr) {
    vTaskDelete(captureTask_);
    captureTask_ = nullptr;
  }
  if (rxHandle_ != nullptr) {
    i2s_channel_disable(rxHandle_);
    i2s_del_channel(rxHandle_);
    rxHandle_ = nullptr;
  }
  if (txHandle_ != nullptr) {
    i2s_channel_disable(txHandle_);
    i2s_del_channel(txHandle_);
    txHandle_ = nullptr;
  }
  heap_caps_free(rawChunk_);
  heap_caps_free(ring_);
  delete[] ringMeta_;
  delete[] eofMonotonicUs_;
  delete[] eofSequence_;
  if (frameReady_ != nullptr) {
    vSemaphoreDelete(frameReady_);
  }
}

bool EspI2sTdmSource::configureChannels() {
  i2s_chan_config_t channelConfig = I2S_CHANNEL_DEFAULT_CONFIG(config_.i2sPort, I2S_ROLE_MASTER);
  channelConfig.dma_desc_num = config_.dmaDescNum;
  channelConfig.dma_frame_num = config_.dmaFrameNum;
  channelConfig.auto_clear = true;

  const bool wantLoopback = pins_.loopbackDout >= 0;
  esp_err_t err = i2s_new_channel(
      &channelConfig, wantLoopback ? &txHandle_ : nullptr, &rxHandle_);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "i2s_new_channel failed: %s", esp_err_to_name(err));
    return false;
  }

  i2s_tdm_config_t tdmConfig = {};
  tdmConfig.clk_cfg = {
      .sample_rate_hz = config_.sampleRateHz,
      .clk_src = I2S_CLK_SRC_DEFAULT,
      .ext_clk_freq_hz = 0,
      // 256x gives mclk = 8.192 MHz and bclk = sampleRate * slots * slotBits =
      // 4.096 MHz at 32 kHz/4ch/32-bit -- an integer ratio, so no sample-rate
      // rounding error.
      .mclk_multiple = I2S_MCLK_MULTIPLE_256,
      .bclk_div = 8,
  };
  tdmConfig.slot_cfg = {
      // Read raw 32-bit slots and do the 24 -> 16 bit conversion in software via
      // the shared decoder, for byte-parity with the Pico node.
      .data_bit_width = I2S_DATA_BIT_WIDTH_32BIT,
      .slot_bit_width = I2S_SLOT_BIT_WIDTH_32BIT,
      .slot_mode = I2S_SLOT_MODE_STEREO,
      .slot_mask = static_cast<i2s_tdm_slot_mask_t>(
          I2S_TDM_SLOT0 | I2S_TDM_SLOT1 | I2S_TDM_SLOT2 | I2S_TDM_SLOT3),
      .ws_width = config_.wsWidth,
      .ws_pol = config_.wsPolarityHigh,
      .bit_shift = config_.bitShift,
      .left_align = config_.leftAlign,
      .big_endian = false,
      .bit_order_lsb = false,
      .skip_mask = false,
      .total_slot = config_.tdmSlots,
  };
  tdmConfig.gpio_cfg = {
      .mclk = I2S_GPIO_UNUSED,
      .bclk = static_cast<gpio_num_t>(pins_.bclk),
      .ws = static_cast<gpio_num_t>(pins_.ws),
      .dout = wantLoopback ? static_cast<gpio_num_t>(pins_.loopbackDout) : I2S_GPIO_UNUSED,
      .din = static_cast<gpio_num_t>(pins_.din),
      .invert_flags = {
          .mclk_inv = 0,
          .bclk_inv = 0,
          .ws_inv = 0,
      },
  };

  err = i2s_channel_init_tdm_mode(rxHandle_, &tdmConfig);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "rx init_tdm_mode failed: %s", esp_err_to_name(err));
    return false;
  }
  if (txHandle_ != nullptr) {
    // Same port, same config -> the driver pairs the two into full duplex, so TX
    // rides the RX-generated BCLK/FSYNC and the loopback is inherently aligned.
    err = i2s_channel_init_tdm_mode(txHandle_, &tdmConfig);
    if (err != ESP_OK) {
      ESP_LOGE(kTag, "tx init_tdm_mode failed: %s", esp_err_to_name(err));
      return false;
    }
  }

  i2s_event_callbacks_t callbacks = {};
  callbacks.on_recv = &EspI2sTdmSource::onRecvIsr;
  callbacks.on_recv_q_ovf = &EspI2sTdmSource::onRecvOverflowIsr;
  err = i2s_channel_register_event_callback(rxHandle_, &callbacks, this);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "register_event_callback failed: %s", esp_err_to_name(err));
    return false;
  }
  return true;
}

bool EspI2sTdmSource::begin() {
  if (initialized_) {
    return true;
  }
  if (config_.sampleRateHz == 0 || config_.frameSamples == 0 || config_.tdmSlots != 4 ||
      config_.ringFrames == 0 || config_.dmaFrameNum == 0 || config_.dmaDescNum == 0) {
    ESP_LOGE(kTag, "invalid TDM config");
    return false;
  }
  if (pins_.bclk < 0 || pins_.ws < 0 || pins_.din < 0) {
    ESP_LOGE(kTag, "TDM pins not assigned");
    return false;
  }

  chunkFrames_ = config_.dmaFrameNum;
  chunkBytes_ = chunkFrames_ * rawFrameBytes();
  if (chunkBytes_ > kMaxDmaDescriptorBytes) {
    ESP_LOGE(kTag,
             "dma_frame_num=%u yields %u B/descriptor, over the %u B GDMA limit",
             static_cast<unsigned>(config_.dmaFrameNum),
             static_cast<unsigned>(chunkBytes_),
             static_cast<unsigned>(kMaxDmaDescriptorBytes));
    return false;
  }
  if ((config_.frameSamples % chunkFrames_) != 0) {
    // The descriptor-EOF timestamp bookkeeping assumes a whole number of DMA
    // descriptors per audio frame; otherwise a frame's end time would land
    // mid-descriptor and drift.
    ESP_LOGE(kTag,
             "frameSamples=%u must be a multiple of dma_frame_num=%u",
             static_cast<unsigned>(config_.frameSamples),
             static_cast<unsigned>(chunkFrames_));
    return false;
  }

  samplesPerFrameAllChannels_ = config_.frameSamples * config_.tdmSlots;
  frameDurationUs_ = static_cast<uint64_t>(
      (static_cast<double>(config_.frameSamples) * 1000000.0) /
      static_cast<double>(config_.sampleRateHz));

  // Audio buffers live in internal SRAM. PSRAM on this board is QSPI; putting
  // the hot capture path behind it would trade the very headroom this node
  // exists to measure.
  constexpr uint32_t kInternalDma = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
  rawChunk_ = static_cast<uint32_t*>(heap_caps_malloc(chunkBytes_, kInternalDma));
  ring_ = static_cast<int16_t*>(heap_caps_malloc(
      static_cast<size_t>(config_.ringFrames) * samplesPerFrameAllChannels_ * sizeof(int16_t),
      kInternalDma));
  ringMeta_ = new (std::nothrow) RingSlotMeta[config_.ringFrames];
  eofMonotonicUs_ = new (std::nothrow) uint64_t[config_.dmaDescNum]();
  eofSequence_ = new (std::nothrow) uint32_t[config_.dmaDescNum]();
  frameReady_ = xSemaphoreCreateBinary();
  if (rawChunk_ == nullptr || ring_ == nullptr || ringMeta_ == nullptr ||
      eofMonotonicUs_ == nullptr || eofSequence_ == nullptr || frameReady_ == nullptr) {
    ESP_LOGE(kTag, "audio buffer allocation failed");
    return false;
  }

  if (!configureChannels()) {
    return false;
  }
  esp_err_t err = i2s_channel_enable(rxHandle_);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "rx enable failed: %s", esp_err_to_name(err));
    return false;
  }
  if (txHandle_ != nullptr) {
    err = i2s_channel_enable(txHandle_);
    if (err != ESP_OK) {
      ESP_LOGE(kTag, "tx enable failed: %s", esp_err_to_name(err));
      return false;
    }
  }

  const BaseType_t created = xTaskCreatePinnedToCore(
      &EspI2sTdmSource::captureTaskEntry,
      "mmpr_cap",
      config_.captureTaskStackBytes,
      this,
      config_.captureTaskPriority,
      &captureTask_,
      config_.captureTaskCoreId);
  if (created != pdPASS) {
    ESP_LOGE(kTag, "capture task create failed");
    return false;
  }

  initialized_ = true;
  ESP_LOGI(kTag,
           "TDM master %u Hz x%u ch bclk=%d ws=%d din=%d dout=%d ring=%u frames dma=%ux%u",
           static_cast<unsigned>(config_.sampleRateHz),
           static_cast<unsigned>(config_.tdmSlots),
           pins_.bclk, pins_.ws, pins_.din, pins_.loopbackDout,
           static_cast<unsigned>(config_.ringFrames),
           static_cast<unsigned>(config_.dmaDescNum),
           static_cast<unsigned>(config_.dmaFrameNum));
  return true;
}

bool IRAM_ATTR EspI2sTdmSource::onRecvIsr(
    i2s_chan_handle_t /*handle*/, i2s_event_data_t* /*event*/, void* context) {
  EspI2sTdmSource* self = static_cast<EspI2sTdmSource*>(context);
  const uint32_t sequence = self->eofCount_.load(std::memory_order_relaxed) + 1u;
  const uint32_t slot = sequence % self->config_.dmaDescNum;
  self->eofMonotonicUs_[slot] = nowMonotonicUs();
  self->eofSequence_[slot] = sequence;
  self->eofCount_.store(sequence, std::memory_order_release);
  return false;  // no higher-priority task woken from here
}

bool IRAM_ATTR EspI2sTdmSource::onRecvOverflowIsr(
    i2s_chan_handle_t /*handle*/, i2s_event_data_t* /*event*/, void* context) {
  EspI2sTdmSource* self = static_cast<EspI2sTdmSource*>(context);
  self->driverOverflowCount_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

void EspI2sTdmSource::captureTaskEntry(void* context) {
  static_cast<EspI2sTdmSource*>(context)->captureLoop();
}

void EspI2sTdmSource::captureLoop() {
  const size_t chunkSamples = chunkFrames_;
  while (true) {
    size_t bytesRead = 0;
    const esp_err_t err =
        i2s_channel_read(rxHandle_, rawChunk_, chunkBytes_, &bytesRead, portMAX_DELAY);
    if (err != ESP_OK || bytesRead != chunkBytes_) {
      // A short or failed read means the driver dropped data; restart the frame
      // being assembled rather than splicing a gap into it silently. The lost
      // samples still surface as a continuity violation at the NodeRunner layer.
      stagedSamples_ = 0;
      continue;
    }

    ++consumedDescriptors_;

    // Decode this descriptor's slots into the ring slot currently being filled.
    int16_t* destination = ring_ + (static_cast<size_t>(writeIndex_) * samplesPerFrameAllChannels_) +
                           (stagedSamples_ * config_.tdmSlots);
    for (size_t sample = 0; sample < chunkSamples; ++sample) {
      const uint32_t* slotWords = rawChunk_ + (sample * config_.tdmSlots);
      for (uint8_t channel = 0; channel < 4; ++channel) {
        const uint8_t slot = config_.outputChannelToSlot[channel];
        destination[(sample * 4u) + channel] = decodeLeftJustifiedSlotToPcm16(
            slotWords[slot], config_.slotBits, config_.validBits, config_.captureBitOffset);
      }
    }
    stagedSamples_ += chunkSamples;
    if (stagedSamples_ < config_.frameSamples) {
      continue;
    }
    stagedSamples_ = 0;

    // Block end time: the EOF timestamp of the descriptor that just completed
    // this frame, taken in the driver ISR rather than here, so scheduler latency
    // on the capture task does not smear the timestamp.
    const uint32_t slot = consumedDescriptors_ % config_.dmaDescNum;
    const uint32_t latestEof = eofCount_.load(std::memory_order_acquire);
    uint64_t blockEndUs = eofMonotonicUs_[slot];
    if (eofSequence_[slot] != consumedDescriptors_ || blockEndUs == 0) {
      // Descriptor bookkeeping slipped (driver overflow, or a restarted frame).
      // Fall back to now(); slightly less accurate, never wrong by a frame.
      blockEndUs = nowMonotonicUs();
      consumedDescriptors_ = latestEof;
    }

    RingSlotMeta meta;
    meta.startSampleIndex = nextProducedStartSampleIndex_;
    meta.blockEndMonotonicUs = blockEndUs;
    meta.completedBlockCount = nextCompletedBlockCount_ + 1u;

    taskENTER_CRITICAL(&ringLock_);
    ringMeta_[writeIndex_] = meta;
    nextProducedStartSampleIndex_ += config_.frameSamples;
    ++nextCompletedBlockCount_;
    if (filledFrames_ == config_.ringFrames) {
      // Ring full: drop the oldest, exactly as the Pico's DMA IRQ does, so
      // droppedBlocksBeforeCapture keeps the same meaning on both nodes.
      readIndex_ = (readIndex_ + 1u) % config_.ringFrames;
      ++droppedFrameCount_;
    } else {
      ++filledFrames_;
      if (filledFrames_ > ringFramesHighWater_) {
        ringFramesHighWater_ = filledFrames_;
      }
    }
    writeIndex_ = (writeIndex_ + 1u) % config_.ringFrames;
    taskEXIT_CRITICAL(&ringLock_);

    xSemaphoreGive(frameReady_);
  }
}

bool EspI2sTdmSource::takeFrame(
    int16_t* interleavedOut,
    size_t samplesPerChannel,
    AudioCaptureTimestamp* captureTimestamp) {
  if (!initialized_ || interleavedOut == nullptr || samplesPerChannel != config_.frameSamples) {
    return false;
  }

  uint32_t slot = 0;
  RingSlotMeta meta;
  uint32_t droppedBeforeCapture = 0;

  taskENTER_CRITICAL(&ringLock_);
  if (filledFrames_ == 0) {
    taskEXIT_CRITICAL(&ringLock_);
    return false;
  }
  slot = readIndex_;
  meta = ringMeta_[slot];
  droppedBeforeCapture = droppedFrameCount_ - reportedDroppedFrameCount_;
  reportedDroppedFrameCount_ = droppedFrameCount_;
  // Reserve the slot before leaving the critical section, so the producer can
  // never hand the same frame out twice (the Pico source moves its read index up
  // front for the same reason).
  //
  // Residual hazard, identical to the Pico's: once the ring is full the producer
  // is filling the slot the consumer would read next, so a frame taken at that
  // exact moment can be half-overwritten. It only happens during an overflow --
  // which is already counted as loss in droppedFrameCount_ -- and matching the
  // Pico's behaviour here is deliberate, since the two nodes' loss numbers are
  // meant to be compared.
  readIndex_ = (slot + 1u) % config_.ringFrames;
  --filledFrames_;
  taskEXIT_CRITICAL(&ringLock_);

  std::memcpy(
      interleavedOut,
      ring_ + (static_cast<size_t>(slot) * samplesPerFrameAllChannels_),
      samplesPerFrameAllChannels_ * sizeof(int16_t));

  if (captureTimestamp != nullptr) {
    captureTimestamp->startSampleIndex = meta.startSampleIndex;
    captureTimestamp->endSampleIndex = meta.startSampleIndex + config_.frameSamples;
    captureTimestamp->blockEndMonotonicUs = meta.blockEndMonotonicUs;
    captureTimestamp->blockStartMonotonicUs = meta.blockEndMonotonicUs >= frameDurationUs_
        ? (meta.blockEndMonotonicUs - frameDurationUs_)
        : 0;
    captureTimestamp->completedBlockCount = meta.completedBlockCount;
    captureTimestamp->dmaRingSlotIndex = slot;
    captureTimestamp->droppedBlocksBeforeCapture = droppedBeforeCapture;
  }
  return true;
}

bool EspI2sTdmSource::readFrame(
    int16_t* interleavedOut,
    size_t samplesPerChannel,
    AudioCaptureTimestamp* captureTimestamp) {
  while (true) {
    if (takeFrame(interleavedOut, samplesPerChannel, captureTimestamp)) {
      return true;
    }
    if (!initialized_ || samplesPerChannel != config_.frameSamples || interleavedOut == nullptr) {
      return false;
    }
    // Bounded wait so a stalled capture path surfaces as a slow read rather than
    // a permanently blocked runner task.
    xSemaphoreTake(frameReady_, pdMS_TO_TICKS(100));
  }
}

bool EspI2sTdmSource::readFrameNonblocking(
    int16_t* interleavedOut,
    size_t samplesPerChannel,
    AudioCaptureTimestamp* captureTimestamp) {
  return takeFrame(interleavedOut, samplesPerChannel, captureTimestamp);
}

uint32_t EspI2sTdmSource::availableFrames() const {
  if (!initialized_) {
    return 0;
  }
  portMUX_TYPE* lock = const_cast<portMUX_TYPE*>(&ringLock_);
  taskENTER_CRITICAL(lock);
  const uint32_t filled = filledFrames_;
  taskEXIT_CRITICAL(lock);
  return filled;
}

}  // namespace mmpr
