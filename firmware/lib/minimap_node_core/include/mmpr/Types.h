#pragma once

#include <Arduino.h>

#include <cstddef>
#include <cstdint>

namespace mmpr {

enum class NodeType {
  kPoint,
  kSirithTetra,
  kUnknown,
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

  const Vec3* sensorOffsetsM;
  size_t sensorCount;

  const char* const* capabilities;
  size_t capabilityCount;

  const char* hardwareName;
  const char* firmwareVersion;
};

struct AudioFrame {
  uint64_t startTimeNs;
  uint32_t sampleRateHz;
  uint8_t channels;
  uint64_t sequence;

  const int16_t* interleavedSamples;
  size_t samplesPerChannel;
};

struct EnvironmentalSample {
  bool hasTemperatureC = false;
  float temperatureC = 0.0f;
  const char* temperatureSource = nullptr;
};

struct PublishResult {
  bool ok;
  int statusCode;
  String responseBody;
};

}  // namespace mmpr
