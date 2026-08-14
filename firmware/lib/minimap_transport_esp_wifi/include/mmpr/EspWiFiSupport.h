#pragma once

// ESP-IDF station-mode Wi-Fi bring-up for MinimapPR nodes.
//
// PORTABILITY CONTRACT: nothing in minimap_transport_esp_wifi may reference an
// ESP32-S3 peripheral. The board redesign plans an ESP32-C5 as a thin
// SPI-slave -> Wi-Fi bridge for the planar node (see "Board Redesign Assessment
// -- 5xPDM Planar.md"), and that firmware is meant to link this library as-is
// for its uplink half and add only the SPI-slave receive side. Keep it to
// esp_wifi / esp_netif / lwIP sockets.

#include <cstdint>

namespace mmpr {

struct EspWiFiConfig {
  const char* ssid = nullptr;
  const char* password = nullptr;
  uint32_t connectTimeoutMs = 15000;
  // Wi-Fi power save costs uplink throughput and adds latency spikes; the CYW43
  // node already disables it (WiFiSupport.cpp) and it matters more here, since
  // the whole point of this node is measuring the uplink ceiling.
  bool disablePowerSave = true;
  const char* hostname = nullptr;
};

// Brings up NVS-backed station mode and blocks until the first IP is acquired
// or connectTimeoutMs elapses. Registers an auto-reconnect handler that keeps
// running for the life of the process, so a mid-run AP outage recovers without
// the node main having to poll anything.
bool espWiFiBegin(const EspWiFiConfig& config);

bool espWiFiConnected();

// Dotted-quad of the station interface, or "0.0.0.0" before the first bind.
const char* espWiFiIpAddress();

// AP RSSI in dBm, 0 when not associated.
int8_t espWiFiRssiDbm();

// Count of (re)connections completed since boot. 1 is the healthy steady state;
// a climbing value is a flapping link and is worth reading next to
// publishConnectOrResetFailures.
uint32_t espWiFiConnectCount();

}  // namespace mmpr
