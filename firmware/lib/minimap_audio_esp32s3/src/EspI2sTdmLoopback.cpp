#include "mmpr/EspI2sTdmLoopback.h"

#include <cstring>

#include "esp_heap_caps.h"
#include "esp_log.h"

#include "mmpr/PcmSlotDecode.h"

namespace mmpr {
namespace {

constexpr const char* kTag = "mmpr-i2s-lb";

// One TDM frame's worth of pattern is repeated across this many frames per
// write, purely to keep the feeder's syscall rate low.
constexpr size_t kPatternFrames = 128;

// Distinct per-slot 24-bit constants: 0x111111, 0x222222, 0x333333, 0x444444.
// Distinct is the point -- a slot-rotation or an off-by-one channel map shows up
// as the wrong constant rather than as plausible-looking audio.
constexpr uint32_t kSlotPatternBase = 0x111111u;

}  // namespace

EspI2sTdmLoopback::EspI2sTdmLoopback(EspI2sTdmSource& source) : source_(source) {}

EspI2sTdmLoopback::~EspI2sTdmLoopback() {
  if (feederTask_ != nullptr) {
    vTaskDelete(feederTask_);
    feederTask_ = nullptr;
  }
  heap_caps_free(patternBuffer_);
}

uint32_t EspI2sTdmLoopback::patternWordForSlot(const EspI2sTdmConfig& config, uint8_t slot) {
  const uint32_t payload = kSlotPatternBase * static_cast<uint32_t>(slot + 1u);
  const uint8_t paddingBits = static_cast<uint8_t>(config.slotBits - config.validBits);
  // Left-justify the payload in the slot, matching what the ADAU7112 puts on the
  // wire: validBits of PCM at the top, pad below.
  return payload << paddingBits;
}

int16_t EspI2sTdmLoopback::expectedPcmForSlot(const EspI2sTdmConfig& config, uint8_t slot) {
  return decodeLeftJustifiedSlotToPcm16(
      patternWordForSlot(config, slot),
      config.slotBits,
      config.validBits,
      config.captureBitOffset);
}

bool EspI2sTdmLoopback::begin() {
  if (source_.loopbackTxHandle() == nullptr) {
    ESP_LOGE(kTag, "no TX channel -- set pins.loopbackDout before EspI2sTdmSource::begin()");
    return false;
  }
  if (feederTask_ != nullptr) {
    return true;
  }

  const size_t frameWords = source_.rawFrameBytes() / sizeof(uint32_t);
  patternBufferBytes_ = kPatternFrames * source_.rawFrameBytes();
  patternBuffer_ = static_cast<uint32_t*>(
      heap_caps_malloc(patternBufferBytes_, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (patternBuffer_ == nullptr) {
    ESP_LOGE(kTag, "pattern buffer allocation failed");
    return false;
  }

  const EspI2sTdmConfig& config = source_.config();
  for (size_t frame = 0; frame < kPatternFrames; ++frame) {
    for (size_t slot = 0; slot < frameWords; ++slot) {
      patternBuffer_[(frame * frameWords) + slot] =
          patternWordForSlot(config, static_cast<uint8_t>(slot));
    }
  }

  const BaseType_t created = xTaskCreatePinnedToCore(
      &EspI2sTdmLoopback::feederTaskEntry,
      "mmpr_lb",
      3072,
      this,
      19,  // just under the capture task, so capture always wins
      &feederTask_,
      1);
  if (created != pdPASS) {
    ESP_LOGE(kTag, "feeder task create failed");
    return false;
  }
  ESP_LOGI(kTag, "loopback pattern feeder started (%u frames/write)",
           static_cast<unsigned>(kPatternFrames));
  return true;
}

void EspI2sTdmLoopback::feederTaskEntry(void* context) {
  static_cast<EspI2sTdmLoopback*>(context)->feederLoop();
}

void EspI2sTdmLoopback::feederLoop() {
  i2s_chan_handle_t tx = source_.loopbackTxHandle();
  while (true) {
    size_t written = 0;
    // Blocks until DMA has room, which paces this task to the sample rate.
    i2s_channel_write(tx, patternBuffer_, patternBufferBytes_, &written, portMAX_DELAY);
  }
}

bool EspI2sTdmLoopback::runSelfTest(
    EspI2sTdmLoopbackResult& result,
    uint32_t warmupFrames,
    uint32_t verifyFrames) {
  result = {};
  result.mismatchSampleIndex = SIZE_MAX;

  const EspI2sTdmConfig& config = source_.config();
  for (uint8_t channel = 0; channel < 4; ++channel) {
    result.expected[channel] = expectedPcmForSlot(config, config.outputChannelToSlot[channel]);
  }

  const size_t frameSamples = source_.frameSamples();
  const size_t channels = source_.channels();
  int16_t* frameBuffer = static_cast<int16_t*>(
      heap_caps_malloc(frameSamples * channels * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (frameBuffer == nullptr) {
    ESP_LOGE(kTag, "self-test buffer allocation failed");
    return false;
  }

  // The first frames after enable carry the DMA buffers' initial zeros and a
  // partially-filled first frame; they are not evidence of anything.
  for (uint32_t i = 0; i < warmupFrames; ++i) {
    if (!source_.readFrame(frameBuffer, frameSamples, nullptr)) {
      heap_caps_free(frameBuffer);
      ESP_LOGE(kTag, "no audio during warmup -- is the DOUT->DIN jumper fitted?");
      return false;
    }
  }

  for (uint32_t frame = 0; frame < verifyFrames; ++frame) {
    if (!source_.readFrame(frameBuffer, frameSamples, nullptr)) {
      heap_caps_free(frameBuffer);
      return false;
    }
    ++result.framesChecked;
    for (size_t sample = 0; sample < frameSamples; ++sample) {
      for (uint8_t channel = 0; channel < 4; ++channel) {
        const int16_t observed = frameBuffer[(sample * channels) + channel];
        if (observed != result.expected[channel]) {
          for (uint8_t c = 0; c < 4; ++c) {
            result.observed[c] = frameBuffer[(sample * channels) + c];
          }
          result.mismatchSampleIndex = sample;
          result.mismatchChannel = channel;
          result.passed = false;
          heap_caps_free(frameBuffer);
          ESP_LOGE(kTag,
                   "FAIL frame=%u sample=%u ch=%u expected=%d observed=%d "
                   "(all channels: %d %d %d %d, expected %d %d %d %d)",
                   static_cast<unsigned>(frame),
                   static_cast<unsigned>(sample),
                   static_cast<unsigned>(channel),
                   static_cast<int>(result.expected[channel]),
                   static_cast<int>(observed),
                   static_cast<int>(result.observed[0]), static_cast<int>(result.observed[1]),
                   static_cast<int>(result.observed[2]), static_cast<int>(result.observed[3]),
                   static_cast<int>(result.expected[0]), static_cast<int>(result.expected[1]),
                   static_cast<int>(result.expected[2]), static_cast<int>(result.expected[3]));
          return true;  // the test ran; result.passed reports the verdict
        }
      }
    }
  }

  heap_caps_free(frameBuffer);
  result.passed = true;
  ESP_LOGI(kTag,
           "PASS %u frames, channels decode to %d %d %d %d (slots %u %u %u %u)",
           static_cast<unsigned>(result.framesChecked),
           static_cast<int>(result.expected[0]), static_cast<int>(result.expected[1]),
           static_cast<int>(result.expected[2]), static_cast<int>(result.expected[3]),
           static_cast<unsigned>(config.outputChannelToSlot[0]),
           static_cast<unsigned>(config.outputChannelToSlot[1]),
           static_cast<unsigned>(config.outputChannelToSlot[2]),
           static_cast<unsigned>(config.outputChannelToSlot[3]));
  return true;
}

}  // namespace mmpr
