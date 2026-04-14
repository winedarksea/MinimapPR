#include "mmpr/WiFiSupport.h"

#include <cstdio>

#include "pico/cyw43_arch.h"
#include "pico/time.h"

namespace mmpr {
namespace {

uint32_t millis32() {
  return to_ms_since_boot(get_absolute_time());
}

const char* linkStatusName(int status) {
  switch (status) {
    case CYW43_LINK_DOWN:
      return "down";
    case CYW43_LINK_JOIN:
      return "join";
    case CYW43_LINK_NOIP:
      return "noip";
    case CYW43_LINK_UP:
      return "up";
    case CYW43_LINK_FAIL:
      return "fail";
    case CYW43_LINK_NONET:
      return "nonet";
    case CYW43_LINK_BADAUTH:
      return "badauth";
    default:
      return "unknown";
  }
}

int linkStatus() {
  return cyw43_tcpip_link_status(&cyw43_state, CYW43_ITF_STA);
}

bool isWiFiConnected() {
  return linkStatus() == CYW43_LINK_UP;
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
  std::printf(
      "[wifi] attempting ssid=%s auth=%s timeout_ms=%lu status_before=%s(%d)\n",
      ssid,
      auth == CYW43_AUTH_OPEN ? "open" : "wpa2-aes-psk",
      static_cast<unsigned long>(timeoutMs),
      linkStatusName(linkStatus()),
      linkStatus());
  const int rc = cyw43_arch_wifi_connect_timeout_ms(ssid, password, auth, timeoutMs);
  if (rc == PICO_OK) {
    std::printf("[wifi] connected status=%s(%d)\n", linkStatusName(linkStatus()), linkStatus());
    return true;
  }

  std::printf(
      "[wifi] connect failed rc=%d status_after=%s(%d)\n",
      rc,
      linkStatusName(linkStatus()),
      linkStatus());
  sleep_ms(retryDelayMs);
  return false;
}

void ensureWiFiConnected(const char* ssid, const char* password, uint32_t timeoutMs, uint32_t checkIntervalMs) {
  static uint32_t lastCheckMs = 0;
  static int lastLoggedStatus = 999;
  const uint32_t nowMs = millis32();
  if ((nowMs - lastCheckMs) < checkIntervalMs) {
    return;
  }
  lastCheckMs = nowMs;

  const int status = linkStatus();
  if (status != lastLoggedStatus) {
    lastLoggedStatus = status;
    std::printf("[wifi] link status=%s(%d)\n", linkStatusName(status), status);
  }

  if (status != CYW43_LINK_UP) {
    (void)connectWiFiBlocking(ssid, password, timeoutMs);
  }
}

}  // namespace mmpr
