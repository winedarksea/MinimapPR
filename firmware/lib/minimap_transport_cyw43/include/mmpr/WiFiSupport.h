#pragma once

#include <cstdint>

namespace mmpr {

bool connectWiFiBlocking(
    const char* ssid,
    const char* password,
    uint32_t timeoutMs,
    uint32_t retryDelayMs = 500);

void ensureWiFiConnected(
    const char* ssid,
    const char* password,
    uint32_t timeoutMs,
    uint32_t checkIntervalMs = 5000);

}  // namespace mmpr
