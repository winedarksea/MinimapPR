#include "mmpr/NtpClient.h"

#include <cstdio>
#include <cstring>

#include "lwip/dns.h"
#include "lwip/pbuf.h"
#include "lwip/udp.h"
#include "pico/time.h"

#include "mmpr/NodeClock.h"
#include "mmpr/Types.h"

namespace mmpr {

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void NtpClient::begin(const char* server, NodeClock& clock) {
  closeUdp();
  server_         = server;
  clock_          = &clock;
  state_          = State::kIdle;
  stateEnteredUs_ = 0;
}

void NtpClient::poll() {
  if (server_ == nullptr || clock_ == nullptr) {
    return;
  }

  const uint64_t nowUs = time_us_64();

  switch (state_) {
    case State::kIdle:
      startQuery();
      break;

    case State::kDnsLookup:
      if ((nowUs - stateEnteredUs_) >= kDnsTimeoutUs) {
        std::printf("[ntp] DNS timeout for %s\n", server_);
        state_          = State::kFailed;
        stateEnteredUs_ = nowUs;
      }
      break;

    case State::kWaitResponse:
      if ((nowUs - stateEnteredUs_) >= kResponseTimeoutUs) {
        std::printf("[ntp] response timeout\n");
        closeUdp();
        state_          = State::kFailed;
        stateEnteredUs_ = nowUs;
      }
      break;

    case State::kFailed:
      if ((nowUs - stateEnteredUs_) >= kRetryIntervalUs) {
        state_ = State::kIdle;
      }
      break;

    case State::kDone:
      // Only re-sync if a better source (GPS) hasn't taken over.
      if (clock_->timeQuality() == TimeQuality::kNtpSync &&
          (nowUs - stateEnteredUs_) >= kResyncIntervalUs) {
        state_ = State::kIdle;
      }
      break;
  }
}

// ---------------------------------------------------------------------------
// Private state machine helpers
// ---------------------------------------------------------------------------

void NtpClient::startQuery() {
  std::printf("[ntp] resolving %s\n", server_);

  ip_addr_t addr;
  const err_t result = dns_gethostbyname(server_, &addr, sDnsCallback, this);

  if (result == ERR_OK) {
    // Address already in DNS cache.
    onDnsFound(addr);
  } else if (result == ERR_INPROGRESS) {
    state_          = State::kDnsLookup;
    stateEnteredUs_ = time_us_64();
  } else {
    std::printf("[ntp] dns_gethostbyname err=%d\n", static_cast<int>(result));
    state_          = State::kFailed;
    stateEnteredUs_ = time_us_64();
  }
}

void NtpClient::onDnsFound(const ip_addr_t& addr) {
  serverAddr_ = addr;
  sendRequest();
}

void NtpClient::onDnsFailed() {
  std::printf("[ntp] DNS lookup failed for %s\n", server_);
  state_          = State::kFailed;
  stateEnteredUs_ = time_us_64();
}

void NtpClient::sendRequest() {
  closeUdp();

  pcb_ = udp_new();
  if (pcb_ == nullptr) {
    std::printf("[ntp] udp_new failed\n");
    state_          = State::kFailed;
    stateEnteredUs_ = time_us_64();
    return;
  }

  udp_recv(pcb_, sUdpRecv, this);

  struct pbuf* p = pbuf_alloc(PBUF_TRANSPORT, 48, PBUF_RAM);
  if (p == nullptr) {
    std::printf("[ntp] pbuf_alloc failed\n");
    closeUdp();
    state_          = State::kFailed;
    stateEnteredUs_ = time_us_64();
    return;
  }

  std::memset(p->payload, 0, 48);
  // LI=0 (no warning), VN=4 (NTPv4), Mode=3 (client) → 0b00100011 = 0x23
  static_cast<uint8_t*>(p->payload)[0] = 0x23;

  requestSentMonotonicUs_ = time_us_64();
  const err_t err = udp_sendto(pcb_, p, &serverAddr_, kNtpPort);
  pbuf_free(p);

  if (err != ERR_OK) {
    std::printf("[ntp] udp_sendto err=%d\n", static_cast<int>(err));
    closeUdp();
    requestSentMonotonicUs_ = 0;
    state_          = State::kFailed;
    stateEnteredUs_ = time_us_64();
    return;
  }

  state_          = State::kWaitResponse;
  stateEnteredUs_ = time_us_64();
}

void NtpClient::onUdpRecv(struct pbuf* p) {
  if (p->tot_len < 48) {
    std::printf("[ntp] short response (%u bytes)\n", static_cast<unsigned>(p->tot_len));
    closeUdp();
    state_          = State::kFailed;
    stateEnteredUs_ = time_us_64();
    return;
  }

  uint8_t buf[48];
  pbuf_copy_partial(p, buf, sizeof(buf), 0);
  closeUdp();
  const uint64_t receiptMonotonicUs = time_us_64();

  // Transmit Timestamp field: bytes 40–47 = seconds and fractional seconds
  // since 1 Jan 1900, big-endian.
  const uint32_t secsSince1900 =
      (static_cast<uint32_t>(buf[40]) << 24) |
      (static_cast<uint32_t>(buf[41]) << 16) |
      (static_cast<uint32_t>(buf[42]) <<  8) |
       static_cast<uint32_t>(buf[43]);
  const uint32_t fractionalSeconds =
      (static_cast<uint32_t>(buf[44]) << 24) |
      (static_cast<uint32_t>(buf[45]) << 16) |
      (static_cast<uint32_t>(buf[46]) <<  8) |
       static_cast<uint32_t>(buf[47]);

  if (secsSince1900 == 0) {
    std::printf("[ntp] zero transmit timestamp — server not ready\n");
    state_          = State::kFailed;
    stateEnteredUs_ = time_us_64();
    return;
  }

  const uint64_t secsSinceUnix = static_cast<uint64_t>(secsSince1900) - kNtpToUnixSeconds;
  const uint64_t fractionalNs =
      (static_cast<uint64_t>(fractionalSeconds) * 1'000'000'000ULL) >> 32;
  // T3 = server transmit time in UTC nanoseconds.
  const uint64_t utcNsT3 = (secsSinceUnix * 1'000'000'000ULL) + fractionalNs;

  // SNTP single-packet round-trip correction (RFC 4330 §5):
  //   best_utc ≈ T3 + one_way_delay  ≈ T3 + (T4 - T1) / 2
  // Anchor that estimate to the midpoint of the round trip on the monotonic
  // clock so the NodeClock interpolation starts from the most accurate point.
  // Guard against T1 not being set (e.g. called without a prior sendRequest).
  uint64_t anchorUtcNs = utcNsT3;
  uint64_t anchorMonotonicUs = receiptMonotonicUs;
  uint64_t rttUs = 0;
  if (requestSentMonotonicUs_ > 0 && receiptMonotonicUs >= requestSentMonotonicUs_) {
    rttUs = receiptMonotonicUs - requestSentMonotonicUs_;
    const uint64_t oneWayNs = (rttUs * 1'000ULL) / 2ULL;
    anchorUtcNs = utcNsT3 + oneWayNs;
    anchorMonotonicUs = requestSentMonotonicUs_ + (rttUs / 2ULL);
  }
  requestSentMonotonicUs_ = 0;

  std::printf(
      "[ntp] sync ok — %llu s + %llu ns utcT3  rtt=%llu us\n",
      static_cast<unsigned long long>(secsSinceUnix),
      static_cast<unsigned long long>(fractionalNs),
      static_cast<unsigned long long>(rttUs));

  clock_->setUtcAtMonotonicUs(anchorUtcNs, anchorMonotonicUs, TimeQuality::kNtpSync);

  state_          = State::kDone;
  stateEnteredUs_ = time_us_64();
}

void NtpClient::closeUdp() {
  if (pcb_ != nullptr) {
    udp_remove(pcb_);
    pcb_ = nullptr;
  }
}

// ---------------------------------------------------------------------------
// Static lwIP callbacks
// ---------------------------------------------------------------------------

void NtpClient::sDnsCallback(const char* /*name*/, const ip_addr_t* addr, void* arg) {
  NtpClient* self = static_cast<NtpClient*>(arg);
  if (addr != nullptr) {
    self->onDnsFound(*addr);
  } else {
    self->onDnsFailed();
  }
}

void NtpClient::sUdpRecv(void* arg, struct udp_pcb* /*pcb*/, struct pbuf* p,
                         const ip_addr_t* /*addr*/, uint16_t /*port*/) {
  NtpClient* self = static_cast<NtpClient*>(arg);
  self->onUdpRecv(p);
  pbuf_free(p);
}

}  // namespace mmpr
