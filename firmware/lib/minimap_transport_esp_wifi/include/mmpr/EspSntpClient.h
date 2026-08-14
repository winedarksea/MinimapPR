#pragma once

// SNTP time discipline for NodeClock.
//
// Without this every frame goes out as time_quality=free_running, anchored only
// to the build timestamp (BuildTimestamp.h), and the server down-weights those
// in fusion. With it the node reports ntp_disciplined and its frames carry
// usable absolute UTC.
//
// THREADING: the IDF SNTP sync callback runs on the SNTP/lwIP task, while
// NodeClock is only ever touched from the runner task. So the callback does
// nothing but latch the observation, and the runner drains it via
// takePendingObservation() -> NodeClock::applyNtpObservation() at a point where
// it already owns the clock.

#include <cstdint>

namespace mmpr {

struct EspSntpConfig {
  const char* server = "pool.ntp.org";
  // Re-sync cadence. NodeClock treats an NTP observation as fresh for 2 hours
  // (kNtpFreshUs), so this only needs to be comfortably under that; more
  // frequent syncs keep the frequency estimate tighter.
  uint32_t syncIntervalMs = 600000;
  // Blocking wait for the first sync during boot. 0 skips the wait entirely and
  // lets the node start free-running until the first callback lands.
  uint32_t initialSyncTimeoutMs = 8000;
};

class EspSntpClient {
 public:
  bool begin(const EspSntpConfig& config);

  // Returns true once per completed sync, handing back an observation shaped for
  // NodeClock::applyNtpObservation. roundTripUs is reported as 0: the IDF SNTP
  // client does not expose the exchange RTT, and NodeClock only uses that value
  // to reject observations above kMaxAcceptedNtpRttUs (200 ms), so 0 means
  // "accept" rather than "instantaneous".
  bool takePendingObservation(uint64_t& utcNs, uint64_t& monotonicUs, uint64_t& roundTripUs);

  uint32_t syncCount() const;
  bool everSynced() const { return syncCount() > 0; }
};

}  // namespace mmpr
