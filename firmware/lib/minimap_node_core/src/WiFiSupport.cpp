#include "mmpr/WiFiSupport.h"

#include <Arduino.h>
#include <WiFi.h>

namespace mmpr {

bool connectWiFiBlocking(const char* ssid, const char* password, uint32_t timeoutMs, uint32_t retryDelayMs) {
  if (ssid == nullptr || ssid[0] == '\0') {
    return false;
  }

  if (WiFi.status() == WL_CONNECTED) {
    return true;
  }

#ifdef WIFI_STA
  WiFi.mode(WIFI_STA);
#endif
  WiFi.begin(ssid, password);

  const uint32_t startMs = millis();
  while ((millis() - startMs) < timeoutMs) {
    if (WiFi.status() == WL_CONNECTED) {
      return true;
    }
    delay(retryDelayMs);
  }

  return false;
}

void ensureWiFiConnected(const char* ssid, const char* password, uint32_t timeoutMs, uint32_t checkIntervalMs) {
  static uint32_t lastCheckMs = 0;
  const uint32_t nowMs = millis();
  if ((nowMs - lastCheckMs) < checkIntervalMs) {
    return;
  }
  lastCheckMs = nowMs;

  if (WiFi.status() != WL_CONNECTED) {
    connectWiFiBlocking(ssid, password, timeoutMs);
  }
}

}  // namespace mmpr
