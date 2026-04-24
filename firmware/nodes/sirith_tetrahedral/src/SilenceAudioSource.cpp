#include "mmpr/SilenceAudioSource.h"

#include <cstdio>
#include <cstring>

#include "pico/time.h"

namespace mmpr {

SilenceAudioSource::SilenceAudioSource(uint32_t sampleRateHz, size_t frameSamples, uint8_t channels)
    : sampleRateHz_(sampleRateHz), frameSamples_(frameSamples), channels_(channels) {}

bool SilenceAudioSource::begin() {
  if (sampleRateHz_ == 0 || frameSamples_ == 0 || channels_ == 0) {
    std::printf("[node] invalid silence audio source config\n");
    return false;
  }

  frameDurationUs_ =
      static_cast<uint64_t>((static_cast<double>(frameSamples_) * 1000000.0) / static_cast<double>(sampleRateHz_));
  nextFrameAtUs_ = time_us_64();
  nextSampleIndex_ = 0;
  std::printf(
      "[node] silence audio source enabled channels=%u sample_rate=%lu frame_samples=%u\n",
      static_cast<unsigned>(channels_),
      static_cast<unsigned long>(sampleRateHz_),
      static_cast<unsigned>(frameSamples_));
  return true;
}

bool SilenceAudioSource::readFrame(
    int16_t* interleavedOut,
    size_t samplesPerChannel,
    AudioCaptureTimestamp* captureTimestamp) {
  if (interleavedOut == nullptr || samplesPerChannel != frameSamples_) {
    return false;
  }

  const int64_t nowUs = static_cast<int64_t>(time_us_64());
  if (nextFrameAtUs_ > nowUs) {
    sleep_us(static_cast<uint64_t>(nextFrameAtUs_ - nowUs));
  } else if ((nowUs - nextFrameAtUs_) > static_cast<int64_t>(frameDurationUs_ * 4u)) {
    nextFrameAtUs_ = nowUs;
  }

  const uint64_t frameStartUs = static_cast<uint64_t>(nextFrameAtUs_);
  const uint64_t startSampleIndex = nextSampleIndex_;
  std::memset(interleavedOut, 0, frameSamples_ * channels_ * sizeof(int16_t));
  nextFrameAtUs_ += static_cast<int64_t>(frameDurationUs_);
  nextSampleIndex_ += frameSamples_;
  if (captureTimestamp != nullptr) {
    captureTimestamp->startSampleIndex = startSampleIndex;
    captureTimestamp->endSampleIndex = startSampleIndex + frameSamples_;
    captureTimestamp->blockStartMonotonicUs = frameStartUs;
    captureTimestamp->blockEndMonotonicUs = static_cast<uint64_t>(nextFrameAtUs_);
    captureTimestamp->completedBlockCount = startSampleIndex / frameSamples_;
    captureTimestamp->dmaRingSlotIndex = 0;
    captureTimestamp->droppedBlocksBeforeCapture = 0;
  }
  return true;
}

}  // namespace mmpr
