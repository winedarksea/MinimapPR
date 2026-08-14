#include "mmpr/EspSntpClient.h"

#include <atomic>
#include <sys/time.h>

#include "esp_log.h"
#include "esp_netif_sntp.h"
#include "esp_sntp.h"

#include "mmpr/PlatformTime.h"

namespace mmpr {
namespace {

constexpr const char* kTag = "mmpr-sntp";
constexpr uint64_t kNsPerSecond = 1000000000ULL;

std::atomic<bool> gPendingObservation{false};
std::atomic<uint32_t> gSyncCount{0};
uint64_t gPendingUtcNs = 0;
uint64_t gPendingMonotonicUs = 0;

// Runs on the SNTP task. Latch only -- see the threading note in the header.
void onTimeSync(struct timeval* tv) {
  if (tv == nullptr) {
    return;
  }
  // Read the monotonic clock first so the pairing is as tight as possible; the
  // settimeofday that produced tv has already happened.
  gPendingMonotonicUs = nowMonotonicUs();
  gPendingUtcNs = (static_cast<uint64_t>(tv->tv_sec) * kNsPerSecond) +
                  (static_cast<uint64_t>(tv->tv_usec) * 1000ULL);
  gSyncCount.fetch_add(1, std::memory_order_relaxed);
  gPendingObservation.store(true, std::memory_order_release);
}

}  // namespace

bool EspSntpClient::begin(const EspSntpConfig& config) {
  esp_sntp_config_t sntpConfig = ESP_NETIF_SNTP_DEFAULT_CONFIG(config.server);
  // Step, don't slew: a smooth adjtime would make the monotonic<->UTC mapping
  // NodeClock builds drift under it. NodeClock does its own filtering.
  sntpConfig.smooth_sync = false;
  sntpConfig.start = true;
  sntpConfig.sync_cb = &onTimeSync;

  const esp_err_t err = esp_netif_sntp_init(&sntpConfig);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "sntp init failed: %s", esp_err_to_name(err));
    return false;
  }
  esp_sntp_set_sync_interval(config.syncIntervalMs);

  if (config.initialSyncTimeoutMs > 0) {
    const esp_err_t waited =
        esp_netif_sntp_sync_wait(pdMS_TO_TICKS(config.initialSyncTimeoutMs));
    if (waited != ESP_OK) {
      // Not fatal: the node runs free-running (build-timestamp anchored) and
      // upgrades to ntp_disciplined whenever the first sync lands.
      ESP_LOGW(kTag, "no NTP sync within %u ms; starting free-running",
               static_cast<unsigned>(config.initialSyncTimeoutMs));
      return true;
    }
    ESP_LOGI(kTag, "time synced from %s", config.server);
  }
  return true;
}

bool EspSntpClient::takePendingObservation(
    uint64_t& utcNs, uint64_t& monotonicUs, uint64_t& roundTripUs) {
  if (!gPendingObservation.exchange(false, std::memory_order_acquire)) {
    return false;
  }
  utcNs = gPendingUtcNs;
  monotonicUs = gPendingMonotonicUs;
  roundTripUs = 0;
  return true;
}

uint32_t EspSntpClient::syncCount() const {
  return gSyncCount.load(std::memory_order_relaxed);
}

}  // namespace mmpr
