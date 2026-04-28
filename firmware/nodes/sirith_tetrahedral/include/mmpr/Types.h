#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mmpr {

enum class PublishFailureStage : uint8_t {
  kNone = 0,
  kDns = 1,
  kConnect = 2,
  kSend = 3,
  kRecv = 4,
  kResponseParse = 5,
  kTimeout = 6,
  kWiFiDisconnected = 7,
};

enum class NodeType {
  kPoint,
  kSirithTetra,
  kUnknown,
};

enum class TimeQuality {
  kGpsLocked,
  kGpsHoldover,
  kNtpDisciplined,
  kFreeRunning,
};

struct GeoPoint {
  float lat;
  float lon;
  float altM;
};

struct Vec3 {
  float x;
  float y;
  float z;
};

struct NodeDescriptor {
  const char* id;
  NodeType type;
  Vec3 positionM;
  bool hasGeoPosition;
  GeoPoint geoPosition;

  const Vec3* sensorOffsetsM;
  size_t sensorCount;

  const char* const* capabilities;
  size_t capabilityCount;

  const char* hardwareName;
  const char* firmwareVersion;
  const char* gpsSignalStatus;
  const char* positionSource;
  uint32_t bootCount;
};

inline constexpr GeoPoint makeGeoPoint(float lat, float lon, float altM) {
  return GeoPoint{lat, lon, altM};
}

struct AudioFrame {
  uint64_t startTimeNs;
  uint64_t endTimeNs;
  uint64_t startSampleIndex;
  uint64_t endSampleIndex;
  uint32_t sampleRateHz;
  uint8_t channels;
  uint64_t sequence;
  uint64_t toaNs;
  uint64_t torNs;
  TimeQuality timeQuality;
  bool hasTimingDiagnostics;
  bool timingHasGpsAnchor;
  uint32_t ppsEdgeCount;
  uint32_t dmaRingSlotIndex;
  int64_t ppsPhaseErrorNs;
  double estimatedPpm;
  const int16_t* interleavedSamples;
  size_t samplesPerChannel;
  uint64_t runnerFramesCaptured = 0;
  uint64_t runnerFramesDropped = 0;
  uint64_t runnerContinuityViolations = 0;
  uint64_t runnerPublishErrors = 0;
  uint32_t runnerQueueDepth = 0;
  uint64_t runnerQueueOverflows = 0;
  int runnerLastPublishStatus = 0;
  uint64_t packetAgeUs = 0;
  PublishFailureStage runnerLastPublishFailureStage = PublishFailureStage::kNone;
  int32_t runnerLastPublishLwipError = 0;
  uint32_t runnerConsecutivePublishFailures = 0;
  uint64_t runnerPublishTimeoutFailures = 0;
  uint64_t runnerPublishConnectOrResetFailures = 0;
  uint64_t runnerPublishDnsFailures = 0;
  uint64_t runnerPublishWifiDownFailures = 0;
};

struct EnvironmentalSample {
  bool hasTemperatureC = false;
  float temperatureC = 0.0f;
  bool hasHumidityFraction = false;
  float humidityFraction = 0.0f;
  const char* temperatureSource = nullptr;
};

struct PublishResult {
  bool ok = false;
  int statusCode = -1;
  PublishFailureStage failureStage = PublishFailureStage::kNone;
  int32_t lwipError = 0;
  std::string responseBody;
};

}  // namespace mmpr
