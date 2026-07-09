#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mmpr/HttpFramePublisher.h"
#include "mmpr/IAudioSource.h"
#include "mmpr/IEnvironmentalSource.h"
#include "mmpr/NodeClock.h"
#include "mmpr/Types.h"

namespace mmpr {

struct RunnerStats {
  uint64_t framesCaptured = 0;
  uint64_t framesPublished = 0;
  uint64_t framesDropped = 0;
  uint64_t publishErrors = 0;
  uint64_t packetContinuityViolations = 0;
  uint64_t queueOverflows = 0;
  uint32_t queueDepth = 0;
  int lastPublishStatus = 0;
  PublishFailureStage lastPublishFailureStage = PublishFailureStage::kNone;
  int32_t lastPublishLwipError = 0;
  uint32_t consecutivePublishFailures = 0;
  uint64_t publishTimeoutFailures = 0;
  uint64_t publishConnectOrResetFailures = 0;
  uint64_t publishDnsFailures = 0;
  uint64_t publishWifiDownFailures = 0;
  uint32_t queueSlotsHighWater = 0;
  uint32_t queueSlotsCapacity = 0;
  uint32_t publishLatencyLastMs = 0;
  uint32_t publishLatencyEwmaMs = 0;
  uint32_t publishLatencyMaxMs = 0;
  uint32_t ringFramesHighWater = 0;
  uint32_t ringFramesCapacity = 0;
  int8_t wifiRssiDbm = 0;
  uint32_t heapFreeBytes = 0;
  uint32_t bootId = 0;
};

class NodeRunner {
 public:
  NodeRunner(
      const NodeDescriptor& descriptor,
      IAudioSource& audioSource,
      HttpFramePublisher& publisher,
      NodeClock& clock,
      uint32_t logEveryFrames = 100,
      IEnvironmentalSource* environmentalSource = nullptr,
      size_t maxPacketSamplesPerChannel = 0,
      uint32_t publishFailureBackoffMs = 0,
      size_t publishBatchFrames = 4,
      size_t publishBatchByteBudget = 0,
      bool usePublishBatchByteBudget = false,
      size_t queueSlots = 0);

  bool begin(
      bool syncNtp,
      const char* ntpServer = "pool.ntp.org",
      long gmtOffsetSeconds = 0,
      int daylightOffsetSeconds = 0);

  void loopOnce();

  const RunnerStats& stats() const { return stats_; }

 private:
  struct QueuedAudioPacket {
    std::vector<int16_t> interleavedSamples;
    uint64_t startSampleIndex = 0;
    uint64_t endSampleIndex = 0;
    uint64_t startMonotonicUs = 0;
    uint64_t sequence = 0;
    bool hasEnvironmentalSample = false;
    EnvironmentalSample environmentalSample = {};
  };

  bool enqueueCurrentPacket(
      uint64_t packetEndSampleIndex,
      const EnvironmentalSample* environmentalSample);
  void dropOldestQueuedPacket();
  bool popQueuedPacket(QueuedAudioPacket& packet);
  AudioFrame buildFrameForPacket(const QueuedAudioPacket& packet, uint64_t publishMonotonicUs) const;
  bool fillActivePublishBatch();
  void clearActivePublishBatch();
  void drainAvailableAudioFrames();
  void processCapturedFrame(
      const AudioCaptureTimestamp& captureTimestamp,
      const EnvironmentalSample* environmentalSample);
  void pumpPublisher();
  void startNextPublishIfPossible();
  void onPublishSuccess();
  void onPublishFailure(const PublishResult& result, uint32_t nowMs);
  uint32_t adaptivePublishBackoffMs() const;
  bool shouldBypassAdaptiveBackoff() const;
  uint32_t effectiveQueueDepth() const;
  void refreshSlowTelemetry(uint32_t nowMs);

  const NodeDescriptor& descriptor_;
  IAudioSource& audioSource_;
  HttpFramePublisher& publisher_;
  IEnvironmentalSource* environmentalSource_ = nullptr;
  bool environmentalSourceReady_ = false;
  NodeClock& clock_;

  int16_t* frameBuffer_ = nullptr;
  size_t frameBufferSamples_ = 0;
  std::vector<int16_t> packetInterleavedSamples_;
  bool packetOpen_ = false;
  uint64_t packetStartSampleIndex_ = 0;
  uint64_t packetStartMonotonicUs_ = 0;
  uint64_t packetTargetEndSampleIndex_ = 0;
  bool haveExpectedNextSampleIndex_ = false;
  uint64_t expectedNextSampleIndex_ = 0;
  uint64_t sequence_ = 0;
  uint32_t logEveryFrames_ = 100;
  uint64_t lastLoggedFrameCount_ = 0;
  size_t maxPacketSamplesPerChannel_ = 0;
  uint32_t publishFailureBackoffMs_ = 0;
  size_t publishBatchFrames_ = 4;
  size_t publishBatchByteBudget_ = 0;
  bool usePublishBatchByteBudget_ = false;
  uint32_t nextPublishAttemptMs_ = 0;
  std::vector<QueuedAudioPacket> queuedPackets_;
  size_t queueHead_ = 0;
  size_t queueDepth_ = 0;
  std::vector<QueuedAudioPacket> activePublishPackets_;
  std::vector<AudioFrame> publishFrameScratch_;
  std::vector<const EnvironmentalSample*> publishEnvironmentScratch_;
  size_t activePublishBatchSize_ = 0;
  bool activePublishBatchValid_ = false;
  uint32_t lastTelemetryRefreshMs_ = 0;

  RunnerStats stats_ = {};
};

}  // namespace mmpr
