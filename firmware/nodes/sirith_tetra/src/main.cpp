#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>

#include "node_config.h"

#include "mmpr/HttpFramePublisher.h"
#include "mmpr/Lsm6TemperatureSource.h"
#include "mmpr/NodeClock.h"
#include "mmpr/NodeRunner.h"
#include "mmpr/SirithDualI2SSource.h"
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

mmpr::SirithDualI2SSource gAudioSource(nodecfg::kI2sPins, nodecfg::kAudioConfig);
mmpr::HttpFramePublisher gPublisher(nodecfg::kServerBaseUrl, nodecfg::kIngestPath, nodecfg::kHttpTimeoutMs);
mmpr::Lsm6TemperatureSourceConfig gImuTempConfig = {
    nodecfg::kImuI2cAddressPrimary7Bit,
    nodecfg::kImuI2cAddressSecondary7Bit,
    nodecfg::kImuTemperatureSampleIntervalMs,
};
mmpr::Lsm6TemperatureSource gImuTempSource(Wire, gImuTempConfig);
mmpr::NodeClock gClock;
mmpr::NodeRunner gRunner(
    kNodeDescriptor,
    gAudioSource,
    gPublisher,
    gClock,
    nodecfg::kLogEveryFrames,
    nodecfg::kEnableImuTemperature ? static_cast<mmpr::IEnvironmentalSource*>(&gImuTempSource) : nullptr);

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(250);

  Serial.println("[sirith] booting");

  if (nodecfg::kEnableImuTemperature) {
    Wire.setSDA(nodecfg::kI2cSdaPin);
    Wire.setSCL(nodecfg::kI2cSclPin);
    Wire.begin();
    Serial.println("[sirith] IMU temperature telemetry enabled (optional)");
  }

  const bool wifiConnected = mmpr::connectWiFiBlocking(
      nodecfg::kWifiSsid,
      nodecfg::kWifiPassword,
      nodecfg::kWiFiConnectTimeoutMs);

  if (!wifiConnected) {
    Serial.println("[sirith] WiFi not connected at startup, will keep retrying");
  } else {
    Serial.printf("[sirith] WiFi connected IP=%s\n", WiFi.localIP().toString().c_str());
  }

  const bool started = gRunner.begin(
      nodecfg::kEnableNtpSync,
      nodecfg::kNtpServer,
      nodecfg::kGmtOffsetSeconds,
      nodecfg::kDaylightOffsetSeconds);

  if (!started) {
    Serial.println("[sirith] fatal: runner failed to start");
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
