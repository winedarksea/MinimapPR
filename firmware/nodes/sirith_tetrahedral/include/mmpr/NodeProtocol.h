#pragma once

#include <cstddef>
#include <string>
#include <vector>

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
bool buildNodePayloadJson(const NodeDescriptor& node, std::string& outPayload);
bool buildFramePayloadParts(
    const AudioFrame& frame,
    const EnvironmentalSample* environment,
    IngestPayloadParts& outParts);
bool buildStoreForwardPayloadParts(
    const NodeDescriptor& node,
    const AudioFrame* frames,
    const EnvironmentalSample* const* environments,
    size_t frameCount,
    bool sortByToa,
    std::vector<IngestPayloadParts>& outParts);
bool buildBinaryStoreForwardPayloadParts(
    const NodeDescriptor& node,
    const AudioFrame* frames,
    const EnvironmentalSample* const* environments,
    size_t frameCount,
    bool sortByToa,
    std::vector<IngestPayloadParts>& outParts);

}  // namespace mmpr
