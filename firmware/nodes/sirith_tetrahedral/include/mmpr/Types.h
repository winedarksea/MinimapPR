#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mmpr {

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
  uint32_t ppsEdgeCount;
  uint32_t dmaRingSlotIndex;
  int64_t ppsPhaseErrorNs;
  double estimatedPpm;

  const int16_t* interleavedSamples;
  size_t samplesPerChannel;
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
  std::string responseBody;
};

}  // namespace mmpr
