#include <Arduino.h>
#include <WiFi.h>

#include "node_config.h"

#include "mmpr/Esp32I2SMonoSource.h"
#include "mmpr/HttpFramePublisher.h"
#include "mmpr/NodeClock.h"
#include "mmpr/NodeRunner.h"
#include "mmpr/WiFiSupport.h"

#ifndef MMPR_FW_VERSION
#define MMPR_FW_VERSION "dev"
#endif

namespace {

const mmpr::NodeDescriptor kNodeDescriptor = {
    nodecfg::kNodeId,
    nodecfg::kNodeType,
    nodecfg::kNodePositionM,
    nodecfg::kSensorOffsetsM,
    sizeof(nodecfg::kSensorOffsetsM) / sizeof(nodecfg::kSensorOffsetsM[0]),
    nodecfg::kCapabilities,
    sizeof(nodecfg::kCapabilities) / sizeof(nodecfg::kCapabilities[0]),
    nodecfg::kHardwareName,
    MMPR_FW_VERSION,
};

mmpr::Esp32I2SMonoSource gAudioSource(nodecfg::kI2sPins, nodecfg::kAudioConfig);
mmpr::HttpFramePublisher gPublisher(nodecfg::kServerBaseUrl, nodecfg::kIngestPath, nodecfg::kHttpTimeoutMs);
mmpr::NodeClock gClock;
mmpr::NodeRunner gRunner(kNodeDescriptor, gAudioSource, gPublisher, gClock, nodecfg::kLogEveryFrames);

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(250);

  Serial.println("[point] booting");

  const bool wifiConnected = mmpr::connectWiFiBlocking(
      nodecfg::kWifiSsid,
      nodecfg::kWifiPassword,
      nodecfg::kWiFiConnectTimeoutMs);

  if (!wifiConnected) {
    Serial.println("[point] WiFi not connected at startup, will keep retrying");
  } else {
    Serial.printf("[point] WiFi connected IP=%s\n", WiFi.localIP().toString().c_str());
  }

  const bool started = gRunner.begin(
      nodecfg::kEnableNtpSync,
      nodecfg::kNtpServer,
      nodecfg::kGmtOffsetSeconds,
      nodecfg::kDaylightOffsetSeconds);

  if (!started) {
    Serial.println("[point] fatal: runner failed to start");
    while (true) {
      delay(1000);
    }
  }
}

void loop() {
  mmpr::ensureWiFiConnected(
      nodecfg::kWifiSsid,
      nodecfg::kWifiPassword,
      nodecfg::kWiFiConnectTimeoutMs);

  gRunner.loopOnce();
}
