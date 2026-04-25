#pragma once

#include <cstddef>
#include <string>

#include "mmpr/Types.h"

namespace mmpr {

struct IngestPayloadParts {
  std::string prefix;
  std::string suffix;
  size_t rawAudioBytes = 0;
  size_t encodedAudioBytes = 0;
};

const char* nodeTypeToWire(NodeType type);

bool buildIngestPayload(const NodeDescriptor& node, const AudioFrame& frame, std::string& outPayload);
bool buildIngestPayload(
    const NodeDescriptor& node,
    const AudioFrame& frame,
    const EnvironmentalSample* environment,
    std::string& outPayload);
bool buildIngestPayloadParts(
    const NodeDescriptor& node,
    const AudioFrame& frame,
    const EnvironmentalSample* environment,
    IngestPayloadParts& outParts);

}  // namespace mmpr
