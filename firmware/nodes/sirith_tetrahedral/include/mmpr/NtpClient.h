#pragma once

#include <cstdint>

#include "lwip/ip_addr.h"

// Forward-declare lwIP types to avoid pulling in the full lwIP headers here.
struct udp_pcb;
struct pbuf;

namespace mmpr {

class NodeClock;

// Non-blocking NTP client driven from the main poll loop.
//
// Priority contract (enforced by NodeClock::setUtcNs):
//   GPS lock  >  NTP sync  >  build timestamp  >  freerunning
//
// NtpClient only calls setUtcNs with kNtpSync quality, so it will:
//   - Upgrade build_timestamp clocks on first sync.
//   - Be silently ignored once a GPS lock has been established.
//
// Retry behaviour:
//   - On failure: retries every kRetryIntervalMs until it succeeds.
//   - On success: re-syncs every kResyncIntervalMs, but only while the clock
//     quality is still kNtpSync (GPS lock stops retries automatically).
class NtpClient {
 public:
  // Store server hostname and clock reference.  Does not initiate any network
  // activity — the first poll() call does that.  'server' must outlive this
  // object (use a config constant or a string literal).
  void begin(const char* server, NodeClock& clock);

  // Advance the state machine.  Call once per main-loop iteration, after
  // cyw43_arch_poll() has had a chance to fire pending lwIP callbacks.
  void poll();

 private:
  enum class State {
    kIdle,         // waiting to start (first query or post-failure/resync cooldown)
    kDnsLookup,    // waiting for DNS callback
    kWaitResponse, // UDP request sent, waiting for server reply
    kDone,         // sync succeeded; waiting for resync interval
    kFailed,       // query failed; waiting for retry interval
  };

  void startQuery();
  void sendRequest();
  void onDnsFound(const ip_addr_t& addr);
  void onDnsFailed();
  void onUdpRecv(struct pbuf* p);
  void closeUdp();

  // lwIP callbacks (C-compatible static functions).
  static void sDnsCallback(const char* name, const ip_addr_t* addr, void* arg);
  static void sUdpRecv(void* arg, struct udp_pcb* pcb, struct pbuf* p,
                       const ip_addr_t* addr, uint16_t port);

  const char* server_ = nullptr;
  NodeClock*  clock_  = nullptr;
  udp_pcb*    pcb_    = nullptr;
  ip_addr_t   serverAddr_ = {};
  State       state_  = State::kIdle;
  uint64_t    stateEnteredUs_ = 0;
  uint64_t    requestSentMonotonicUs_ = 0;  // T1: monotonic time when UDP request was sent

  static constexpr uint64_t kDnsTimeoutUs     = 10'000'000ULL;  //  10 s
  static constexpr uint64_t kResponseTimeoutUs =  5'000'000ULL;  //   5 s
  static constexpr uint64_t kRetryIntervalUs   = 30'000'000ULL;  //  30 s
  static constexpr uint64_t kResyncIntervalUs  = 3'600'000'000ULL; // 1 h

  static constexpr uint16_t kNtpPort          = 123;
  static constexpr uint32_t kNtpToUnixSeconds = 2208988800u; // 1900→1970
};

}  // namespace mmpr
