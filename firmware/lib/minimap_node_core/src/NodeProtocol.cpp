#include "mmpr/NodeProtocol.h"

#include <inttypes.h>
#include <stdio.h>

#include "mmpr/Base64.h"

namespace mmpr {
namespace {

void appendEscapedString(String& out, const char* text) {
  if (text == nullptr) {
    return;
  }

  while (*text != '\0') {
    const char c = *text;
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
    ++text;
  }
}

void appendQuoted(String& out, const char* text) {
  out += '"';
  appendEscapedString(out, text);
  out += '"';
}

void appendFloat(String& out, float value) {
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "%.6f", static_cast<double>(value));
  out += buffer;
}

void appendUint64(String& out, uint64_t value) {
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "%" PRIu64, value);
  out += buffer;
}

void appendUint32(String& out, uint32_t value) {
  char buffer[16];
  snprintf(buffer, sizeof(buffer), "%" PRIu32, value);
  out += buffer;
}

}  // namespace

const char* nodeTypeToWire(NodeType type) {
  switch (type) {
    case NodeType::kPoint:
      return "point";
    case NodeType::kSirithTetra:
      return "sirith_tetra";
    default:
      return "unknown";
  }
}

bool buildIngestPayload(const NodeDescriptor& node, const AudioFrame& frame, String& outPayload) {
  if (node.id == nullptr || node.sensorOffsetsM == nullptr || node.sensorCount == 0 || frame.interleavedSamples == nullptr) {
    return false;
  }

  const size_t bytes = frame.samplesPerChannel * static_cast<size_t>(frame.channels) * sizeof(int16_t);
  const String encoded = encodeBase64(reinterpret_cast<const uint8_t*>(frame.interleavedSamples), bytes);

  outPayload = String();
  outPayload.reserve(640 + encoded.length());

  outPayload += "{\"node\":{";

  outPayload += "\"id\":";
  appendQuoted(outPayload, node.id);

  outPayload += ",\"node_type\":";
  appendQuoted(outPayload, nodeTypeToWire(node.type));

  outPayload += ",\"position_m\":[";
  appendFloat(outPayload, node.positionM.x);
  outPayload += ',';
  appendFloat(outPayload, node.positionM.y);
  outPayload += ',';
  appendFloat(outPayload, node.positionM.z);
  outPayload += ']';

  outPayload += ",\"sensor_offsets_m\":[";
  for (size_t i = 0; i < node.sensorCount; ++i) {
    if (i > 0) {
      outPayload += ',';
    }
    outPayload += '[';
    appendFloat(outPayload, node.sensorOffsetsM[i].x);
    outPayload += ',';
    appendFloat(outPayload, node.sensorOffsetsM[i].y);
    outPayload += ',';
    appendFloat(outPayload, node.sensorOffsetsM[i].z);
    outPayload += ']';
  }
  outPayload += ']';

  outPayload += ",\"capabilities\":[";
  for (size_t i = 0; i < node.capabilityCount; ++i) {
    if (i > 0) {
      outPayload += ',';
    }
    appendQuoted(outPayload, node.capabilities[i]);
  }
  outPayload += ']';

  outPayload += ",\"metadata\":{";
  outPayload += "\"hardware\":";
  appendQuoted(outPayload, node.hardwareName != nullptr ? node.hardwareName : "unknown");
  outPayload += ",\"firmware\":";
  appendQuoted(outPayload, node.firmwareVersion != nullptr ? node.firmwareVersion : "dev");
  outPayload += "}}";

  outPayload += ",\"frame\":{";

  outPayload += "\"start_time_ns\":";
  appendUint64(outPayload, frame.startTimeNs);

  outPayload += ",\"sample_rate_hz\":";
  appendUint32(outPayload, frame.sampleRateHz);

  outPayload += ",\"channels\":";
  outPayload += String(static_cast<unsigned>(frame.channels));

  outPayload += ",\"encoding\":\"pcm16le\"";

  outPayload += ",\"samples_b64\":";
  appendQuoted(outPayload, encoded.c_str());

  outPayload += ",\"sequence\":";
  appendUint64(outPayload, frame.sequence);

  outPayload += "}}";

  return true;
}

}  // namespace mmpr
