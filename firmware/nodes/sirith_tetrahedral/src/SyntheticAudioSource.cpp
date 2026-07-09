#include "mmpr/SyntheticAudioSource.h"

#include <cstdio>

#include "pico/time.h"

namespace mmpr {

SyntheticAudioSource::SyntheticAudioSource(uint32_t sampleRateHz, size_t frameSamples, uint8_t channels)
    : sampleRateHz_(sampleRateHz), frameSamples_(frameSamples), channels_(channels) {}

bool SyntheticAudioSource::begin() {
  if (sampleRateHz_ == 0 || frameSamples_ == 0 || channels_ == 0) {
    std::printf("[node] invalid synthetic audio source config\n");
    return false;
  }

  frameDurationUs_ =
      static_cast<uint64_t>((static_cast<double>(frameSamples_) * 1000000.0) / static_cast<double>(sampleRateHz_));
  nextFrameAtUs_ = time_us_64();
  nextSampleIndex_ = 0;
  std::printf(
      "[node] synthetic audio source enabled channels=%u sample_rate=%lu frame_samples=%u\n",
      static_cast<unsigned>(channels_),
      static_cast<unsigned long>(sampleRateHz_),
      static_cast<unsigned>(frameSamples_));
  return true;
}

bool SyntheticAudioSource::readFrame(
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
  for (size_t sample = 0; sample < frameSamples_; ++sample) {
    const uint64_t absoluteSampleIndex = startSampleIndex + sample;
    for (uint8_t channel = 0; channel < channels_; ++channel) {
      interleavedOut[(sample * static_cast<size_t>(channels_)) + channel] =
          sampleValue(absoluteSampleIndex, channel);
    }
  }

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

uint32_t SyntheticAudioSource::availableFrames() const {
  const int64_t nowUs = static_cast<int64_t>(time_us_64());
  return nowUs >= nextFrameAtUs_ ? 1u : 0u;
}

bool SyntheticAudioSource::readFrameNonblocking(
    int16_t* interleavedOut,
    size_t samplesPerChannel,
    AudioCaptureTimestamp* captureTimestamp) {
  if (availableFrames() == 0) {
    return false;
  }
  return readFrame(interleavedOut, samplesPerChannel, captureTimestamp);
}

int16_t SyntheticAudioSource::sampleValue(uint64_t sampleIndex, uint8_t channel) const {
  const uint32_t word =
      (static_cast<uint32_t>(sampleIndex) * 257u) ^
      (static_cast<uint32_t>(channel) * 0x1f1fu) ^
      0x5a5au;
  return static_cast<int16_t>(word & 0xffffu);
}

}  // namespace mmpr
