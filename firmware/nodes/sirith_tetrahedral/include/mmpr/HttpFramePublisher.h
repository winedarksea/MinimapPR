#pragma once

#include <cstdint>
#include <string>

#include "mmpr/Types.h"

namespace mmpr {

class HttpFramePublisher {
 public:
  HttpFramePublisher(const char* serverBaseUrl, const char* ingestPath, uint32_t timeoutMs);

  PublishResult publish(const NodeDescriptor& node, const AudioFrame& frame, bool keepResponseBody = false);
  PublishResult publish(
      const NodeDescriptor& node,
      const AudioFrame& frame,
      const EnvironmentalSample* environment,
      bool keepResponseBody = false);

  const std::string& endpointUrl() const { return endpointUrl_; }

 private:
  bool parseEndpoint();
  static void trimAsciiWhitespace(std::string& s);

  std::string endpointUrl_;
  std::string host_;
  std::string path_;
  uint16_t port_ = 80;
  bool endpointValid_ = false;
  uint32_t timeoutMs_ = 0;
};

}  // namespace mmpr
