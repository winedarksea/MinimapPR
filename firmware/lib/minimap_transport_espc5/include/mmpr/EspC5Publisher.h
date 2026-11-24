#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mmpr/EspC5Frame.h"
#include "mmpr/IUplinkTransport.h"
#include "mmpr/SpiHostLink.h"
#include "mmpr/Types.h"

namespace mmpr {

// D6: HTTP POST proxied over SPI to an ESP32-C5. The C5 bridge receives a
// DATA_POST frame (endpoint path + opaque MMB3 body), performs the POST
// itself against the backend, and returns a POST_STATUS frame. This class
// maps that exchange 1:1 onto the beginBinaryStoreForwardPublish/pollPublish
// contract used by HttpFramePublisher (see minimap_transport_cyw43) -- it is
// a drop-in IUplinkTransport for planar/ESP32-C5 nodes, so NodeRunner,
// NodeProtocol, and the backend are all unaware the uplink is SPI-proxied
// rather than a direct CYW43/lwIP socket.
//
// HARDWARE-DEPENDENT / UNVERIFIED WITHOUT A BENCH: this class's correctness
// against real hardware depends on SpiHostLink's bus timing and on the C5
// bridge firmware actually implementing DATA_POST/POST_STATUS per D6 -- see
// SpiHostLink.h for the specific caveats. The framing/state-machine logic
// itself (this file + EspC5Frame.cpp) is covered by host unit tests.
class EspC5Publisher : public IUplinkTransport {
 public:
  // `ingestPath` is the backend HTTP path (e.g. "/api/v1/ingest/binary") --
  // the C5 bridge is expected to already know the base URL/host, so only the
  // path travels in the DATA_POST frame (kept small; the host and C5 are
  // provisioned with the same backend target out of band, e.g. via
  // WIFI_CONFIG/future config frames).
  //
  // Timeout chain (2026-11-23 Pico bench: a host timeout shorter than the
  // layer below's loss recovery caused an abort/retry spiral and ~45 % loss):
  //   bridgePostDeadlineMs  -- sent in every DATA_POST; the C5's whole budget
  //                            for the POST, including its own TCP recovery.
  //   hostResponseTimeoutMs -- how long this side waits for POST_STATUS per
  //                            attempt. Must exceed bridgePostDeadlineMs plus
  //                            the SPI transfer, so it only fires when the
  //                            bridge itself is unresponsive. Enforced by
  //                            static_assert in node_config.h; clamped here
  //                            too so a bad caller can't reintroduce the race.
  EspC5Publisher(const char* ingestPath,
                 SpiHostLink& link,
                 uint32_t bridgePostDeadlineMs,
                 uint32_t hostResponseTimeoutMs);

  bool beginBinaryStoreForwardPublish(
      const NodeDescriptor& node,
      const std::vector<AudioFrame>& frames,
      const std::vector<const EnvironmentalSample*>& environments,
      bool sortByToa,
      bool keepResponseBody,
      PublishResult& immediateResult) override;

  bool pollPublish(PublishResult& result) override;
  bool publishInProgress() const override;
  void cancelPublish() override;
  void setBackgroundPollCallback(BackgroundPollCallback callback, void* context) override;
  const std::string& endpointUrl() const override { return endpointUrl_; }
  int8_t linkRssiDbm() const override { return lastRssiDbm_; }

  // Applies a decoded LINK_STATUS frame. LINK_STATUS is unsolicited C5->host
  // telemetry (not part of the DATA_POST/POST_STATUS request/response cycle),
  // so callers that observe one arriving out-of-band (e.g. the node main loop
  // polling the link between publishes) feed it in here rather than through
  // pollPublish. Currently unused by any producer -- see EspC5Frame.h D6
  // notes -- but wired up so the shape is exercised end-to-end once one
  // exists.
  void applyLinkStatus(const LinkStatusPayload& status);
  const LinkStatusPayload& lastLinkStatus() const { return lastLinkStatus_; }

  // DATA_POSTs re-sent after a kEspC5BridgeStatusBusy reply (backpressure).
  // A rising count means the bridge's buffering, not the radio, is the limit.
  uint32_t busyResendsTotal() const { return busyResendsTotal_; }

  // Pause before re-sending a DATA_POST the bridge refused as busy, so a
  // full bridge gets time to drain instead of being hammered over SPI.
  static constexpr uint32_t kBusyResendDelayMs = 20;

 private:
  enum class State {
    kIdle,
    kAwaitingStatus,
    kBusyBackoff,  // bridge refused the frame; resend inFlightFrame_ after a delay
  };

  SpiHostLink& link_;
  std::string endpointUrl_;
  std::string ingestPath_;
  uint32_t bridgePostDeadlineMs_;
  uint32_t hostResponseTimeoutMs_;

  uint16_t nextSeq_ = 1;
  uint16_t inFlightSeq_ = 0;
  State state_ = State::kIdle;
  uint32_t requestStartedMs_ = 0;   // first send of this publish (latency)
  uint32_t attemptStartedMs_ = 0;   // latest send (per-attempt timeout)
  uint32_t uploadMs_ = 0;           // SPI transfer time of the latest send
  bool uploadComplete_ = false;
  uint32_t busyBackoffStartedMs_ = 0;
  uint32_t busyResendsTotal_ = 0;
  // Encoded DATA_POST kept until POST_STATUS so a busy refusal can be re-sent
  // without rebuilding the MMB3 body (whose source frames NodeRunner may
  // already have recycled).
  std::vector<uint8_t> inFlightFrame_;
  LinkStatusPayload lastLinkStatus_;
  bool keepResponseBody_ = false;
  std::vector<uint8_t> rxAccumulator_;

  int8_t lastRssiDbm_ = 0;
  bool linkUp_ = false;

  BackgroundPollCallback backgroundPollCallback_ = nullptr;
  void* backgroundPollContext_ = nullptr;

  uint16_t allocateSeq();
  // Drains any bytes currently available from the link into rxAccumulator_
  // and tries to decode a POST_STATUS frame matching inFlightSeq_. Returns
  // true (and fills `result`) once the in-flight publish is resolved
  // (success, protocol failure, or CRC/decode failure) -- callers should
  // treat that as "poll complete" the same way HttpFramePublisher::pollPublish
  // does.
  bool tryDecodeStatusResponse(PublishResult& result);
  void sendInFlightFrame();
  void fillTiming(PublishResult& result, uint32_t nowMs) const;
  void finishPublish();
};

}  // namespace mmpr
