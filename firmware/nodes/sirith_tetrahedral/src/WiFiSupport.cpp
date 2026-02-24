#include "mmpr/WiFiSupport.h"

#include <cstdio>

#include "cyw43.h"
#include "pico/cyw43_arch.h"
#include "pico/time.h"

namespace mmpr {
namespace {

uint32_t millis32() {
  return to_ms_since_boot(get_absolute_time());
}

bool isWiFiConnected() {
  return cyw43_tcpip_link_status(&cyw43_state, CYW43_ITF_STA) == CYW43_LINK_UP;
}

}  // namespace

bool connectWiFiBlocking(const char* ssid, const char* password, uint32_t timeoutMs, uint32_t retryDelayMs) {
  if (ssid == nullptr || ssid[0] == '\0') {
    return false;
  }

  if (isWiFiConnected()) {
    return true;
  }

  cyw43_arch_enable_sta_mode();

  const uint32_t auth = (password == nullptr || password[0] == '\0') ? CYW43_AUTH_OPEN : CYW43_AUTH_WPA2_AES_PSK;
  const int rc = cyw43_arch_wifi_connect_timeout_ms(ssid, password, auth, timeoutMs);
  if (rc == PICO_OK) {
    return true;
  }

  std::printf("[wifi] connect failed rc=%d\n", rc);
  sleep_ms(retryDelayMs);
  return false;
}

void ensureWiFiConnected(const char* ssid, const char* password, uint32_t timeoutMs, uint32_t checkIntervalMs) {
  static uint32_t lastCheckMs = 0;
  const uint32_t nowMs = millis32();
  if ((nowMs - lastCheckMs) < checkIntervalMs) {
    return;
  }
  lastCheckMs = nowMs;

  if (!isWiFiConnected()) {
    (void)connectWiFiBlocking(ssid, password, timeoutMs);
  }
}

}  // namespace mmpr
