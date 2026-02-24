#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>

#include "node_config.h"

#include "mmpr/HttpFramePublisher.h"
#include "mmpr/Lis2mdlAutoOrientation.h"
#include "mmpr/NodeClock.h"
#include "mmpr/NodeRunner.h"
#include "mmpr/SirithPicoTdmSource.h"
#include "mmpr/WiFiSupport.h"

#ifndef MMPR_FW_VERSION
#define MMPR_FW_VERSION "dev"
#endif

namespace {

constexpr uint8_t kMicCount = 4;

mmpr::Vec3 gSensorOffsetsOrdered[4] = {};
uint8_t gActiveBaseRotationSteps = static_cast<uint8_t>(nodecfg::kBasePlaneRotationSteps % 3);

mmpr::Lis2mdlAutoOrientation gAutoOrientation;
bool gAutoOrientationEnabled = false;

uint8_t rotateBaseMic(uint8_t micIndex, uint8_t baseRotationSteps) {
  if (micIndex >= 3) {
    return micIndex;
  }

  const uint8_t rot = static_cast<uint8_t>(baseRotationSteps % 3);
  return static_cast<uint8_t>((micIndex + rot) % 3);
}

void buildOrderedOffsetsFromSlotMap(uint8_t baseRotationSteps) {
  for (uint8_t channel = 0; channel < kMicCount; ++channel) {
    uint8_t slot = nodecfg::kAudioConfig.outputChannelToSlot[channel];
    if (slot >= kMicCount) {
      Serial.printf(
          "[sirith-pico] invalid slot map outputChannelToSlot[%u]=%u, using slot 0\n",
          static_cast<unsigned>(channel),
          static_cast<unsigned>(slot));
      slot = 0;
    }

    uint8_t rawMicIndex = nodecfg::kSlotToPhysicalMic[slot];
    if (rawMicIndex >= kMicCount) {
      Serial.printf(
          "[sirith-pico] invalid mic map slotToPhysicalMic[%u]=%u, using mic 0\n",
          static_cast<unsigned>(slot),
          static_cast<unsigned>(rawMicIndex));
      rawMicIndex = 0;
    }

    const uint8_t rotatedMicIndex = rotateBaseMic(rawMicIndex, baseRotationSteps);
    gSensorOffsetsOrdered[channel] = nodecfg::kPhysicalSensorOffsetsM[rotatedMicIndex];

    Serial.printf(
        "[sirith-pico] ch%u <- slot%u <- mic%u (rot=%u base=%u)\n",
        static_cast<unsigned>(channel),
        static_cast<unsigned>(slot + 1),
        static_cast<unsigned>(rawMicIndex),
        static_cast<unsigned>(rotatedMicIndex),
        static_cast<unsigned>(baseRotationSteps));
  }
}

mmpr::NodeDescriptor gNodeDescriptor = {
    nodecfg::kNodeId,
    nodecfg::kNodeType,
    nodecfg::kNodePositionM,
    gSensorOffsetsOrdered,
    sizeof(gSensorOffsetsOrdered) / sizeof(gSensorOffsetsOrdered[0]),
    nodecfg::kCapabilities,
    sizeof(nodecfg::kCapabilities) / sizeof(nodecfg::kCapabilities[0]),
    nodecfg::kHardwareName,
    MMPR_FW_VERSION,
};

mmpr::SirithPicoTdmSource gAudioSource(nodecfg::kTdmPins, nodecfg::kAudioConfig);
mmpr::HttpFramePublisher gPublisher(nodecfg::kServerBaseUrl, nodecfg::kIngestPath, nodecfg::kHttpTimeoutMs);
mmpr::NodeClock gClock;
mmpr::NodeRunner gRunner(gNodeDescriptor, gAudioSource, gPublisher, gClock, nodecfg::kLogEveryFrames);

void setupOptionalPeripherals() {
  if (nodecfg::kEnableGpsUart) {
    Serial1.setTX(nodecfg::kGpsTxPin);
    Serial1.setRX(nodecfg::kGpsRxPin);
    Serial1.begin(nodecfg::kGpsUartBaud);
    pinMode(nodecfg::kGpsPpsPin, INPUT_PULLDOWN);
    Serial.println("[sirith-pico] GPS UART enabled");
  }

  if (nodecfg::kEnableCompassAutoOrientation) {
    Wire.setSDA(nodecfg::kI2cSdaPin);
    Wire.setSCL(nodecfg::kI2cSclPin);
    Wire.begin();

    mmpr::Lis2mdlAutoOrientationConfig cfg = {};
    cfg.i2cAddress7Bit = nodecfg::kCompassI2cAddress7Bit;
    cfg.outputDataRateBits = nodecfg::kCompassOutputDataRateBits;
    cfg.sampleIntervalMs = nodecfg::kCompassSampleIntervalMs;
    cfg.smoothingAlpha = nodecfg::kCompassSmoothingAlpha;
    cfg.headingOffsetDeg = nodecfg::kCompassHeadingOffsetDeg;
    cfg.minHorizontalFieldLsb = nodecfg::kCompassMinHorizontalFieldLsb;
    cfg.stableSamplesRequired = nodecfg::kCompassStableSamplesRequired;

    gAutoOrientationEnabled = gAutoOrientation.begin(Wire, cfg, gActiveBaseRotationSteps);
    if (gAutoOrientationEnabled) {
      Serial.println("[sirith-pico] LIS2MDLTR auto-orientation enabled");
    } else {
      Serial.println("[sirith-pico] LIS2MDLTR auto-orientation unavailable; manual rotation retained");
    }
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(300);

  Serial.println("[sirith-pico] booting");

  buildOrderedOffsetsFromSlotMap(gActiveBaseRotationSteps);
  setupOptionalPeripherals();

  const bool wifiConnected = mmpr::connectWiFiBlocking(
      nodecfg::kWifiSsid,
      nodecfg::kWifiPassword,
      nodecfg::kWiFiConnectTimeoutMs);

  if (!wifiConnected) {
    Serial.println("[sirith-pico] WiFi not connected at startup, will keep retrying");
  } else {
    Serial.printf("[sirith-pico] WiFi connected IP=%s\n", WiFi.localIP().toString().c_str());
  }

  const bool started = gRunner.begin(
      nodecfg::kEnableNtpSync,
      nodecfg::kNtpServer,
      nodecfg::kGmtOffsetSeconds,
      nodecfg::kDaylightOffsetSeconds);

  if (!started) {
    Serial.println("[sirith-pico] fatal: runner failed to start");
    while (true) {
      delay(1000);
    }
  }
}

void loop() {
  if (gAutoOrientationEnabled) {
    uint8_t changedRotation = 0;
    if (gAutoOrientation.poll(&changedRotation)) {
      gActiveBaseRotationSteps = changedRotation;
      Serial.printf(
          "[sirith-pico] auto-rotation -> step=%u heading=%.1f deg\n",
          static_cast<unsigned>(gActiveBaseRotationSteps),
          static_cast<double>(gAutoOrientation.headingDeg()));
      buildOrderedOffsetsFromSlotMap(gActiveBaseRotationSteps);
    } else if (!gAutoOrientation.healthy()) {
      gAutoOrientationEnabled = false;
      Serial.println("[sirith-pico] LIS2MDLTR read fault; holding current manual rotation");
    }
  }

  mmpr::ensureWiFiConnected(
      nodecfg::kWifiSsid,
      nodecfg::kWifiPassword,
      nodecfg::kWiFiConnectTimeoutMs);

  gRunner.loopOnce();
}
