#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mmpr/Types.h"

namespace mmpr {

// Transport-agnostic uplink seam used by NodeRunner. It is exactly the subset of
// the store-forward publish contract NodeRunner drives, so any transport that
// can perform a binary store-forward POST and report progress can back a node:
//   * HttpFramePublisher (CYW43/lwIP direct HTTP) -- see minimap_transport_cyw43
//   * EspC5Publisher (HTTP POST proxied over SPI to an ESP32-C5) -- Phase 3
//
// Keeping NodeRunner on this interface (rather than the concrete
// HttpFramePublisher) is what lets the planar node build without any CYW43/lwIP
// dependency.
class IUplinkTransport {
 public:
  using BackgroundPollCallback = void (*)(void*);

  virtual ~IUplinkTransport() = default;

  // Begin an asynchronous binary (MMB3) store-forward publish of a batch of
  // frames. Returns true if a publish was started (poll it via pollPublish),
  // false if it completed/failed synchronously (see immediateResult).
  virtual bool beginBinaryStoreForwardPublish(
      const NodeDescriptor& node,
      const std::vector<AudioFrame>& frames,
      const std::vector<const EnvironmentalSample*>& environments,
      bool sortByToa,
      bool keepResponseBody,
      PublishResult& immediateResult) = 0;

  // Advance an in-flight publish; returns true and fills result on completion.
  virtual bool pollPublish(PublishResult& result) = 0;
  virtual bool publishInProgress() const = 0;
  virtual void cancelPublish() = 0;

  // Optional cooperative callback pumped while the transport blocks on the link.
  virtual void setBackgroundPollCallback(BackgroundPollCallback callback, void* context) = 0;

  // Human-readable endpoint (for logging / control server).
  virtual const std::string& endpointUrl() const = 0;

  // Link signal strength in dBm if the transport knows it, else 0.
  virtual int8_t linkRssiDbm() const { return 0; }
};

}  // namespace mmpr
