#pragma once

#include <cstdint>
#include <string>

#include "mmpr/Types.h"

namespace mmpr {

class HttpFramePublisher {
 public:
  struct TransportState;
  using BackgroundPollCallback = void (*)(void*);

  HttpFramePublisher(const char* serverBaseUrl, const char* ingestPath, uint32_t timeoutMs);
  ~HttpFramePublisher();

  HttpFramePublisher(const HttpFramePublisher&) = delete;
  HttpFramePublisher& operator=(const HttpFramePublisher&) = delete;

  PublishResult publish(const NodeDescriptor& node, const AudioFrame& frame, bool keepResponseBody = false);
  PublishResult publish(
      const NodeDescriptor& node,
      const AudioFrame& frame,
      const EnvironmentalSample* environment,
      bool keepResponseBody = false);

  const std::string& endpointUrl() const { return endpointUrl_; }
  void setBackgroundPollCallback(BackgroundPollCallback callback, void* context);

 private:
  bool parseEndpoint();
  static void trimAsciiWhitespace(std::string& s);

  std::string endpointUrl_;
  std::string host_;
  std::string path_;
  uint16_t port_ = 80;
  bool endpointValid_ = false;
  uint32_t timeoutMs_ = 0;
  TransportState* transportState_ = nullptr;
};

}  // namespace mmpr
