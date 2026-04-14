#include <cstdint>
#include <cstdio>

#include "pico/cyw43_arch.h"
#include "pico/stdlib.h"
#include "hardware/uart.h"

#include "node_config.h"

#include "mmpr/HttpFramePublisher.h"
#include "mmpr/FallbackEnvironmentalSource.h"
#include "mmpr/Lis2mdlMagnetometer.h"
#include "mmpr/Lsm6TemperatureSensor.h"
#include "mmpr/MagAutoOrientation.h"
#include "mmpr/NmeaGpsSource.h"
#include "mmpr/NodeClock.h"
#include "mmpr/NodeRunner.h"
#include "mmpr/Sht4xEnvironmentalSource.h"
#include "mmpr/SilenceAudioSource.h"
#include "mmpr/SirithPicoTdmSource.h"
#include "mmpr/TemperatureEnvironmentalSource.h"
#include "mmpr/WiFiSupport.h"

#ifndef MMPR_FW_VERSION
#define MMPR_FW_VERSION "dev"
#endif

namespace {

constexpr uint8_t kMicCount = 4;

mmpr::Vec3 gSensorOffsetsOrdered[4] = {};
uint8_t gActiveBaseRotationSteps = static_cast<uint8_t>(nodecfg::kBasePlaneRotationSteps % 3u);

mmpr::I2cBus gI2c;
mmpr::Lis2mdlMagnetometer gMagnetometer(gI2c, mmpr::Lis2mdlMagConfig{
    nodecfg::kCompassI2cAddress7Bit,
    nodecfg::kCompassOdrBits,
    nodecfg::kCompassEnableTempComp,
    nodecfg::kCompassEnableLpf,
    nodecfg::kCompassEnableOffsetCancel,
    nodecfg::kCompassHardIronX,
    nodecfg::kCompassHardIronY,
    nodecfg::kCompassHardIronZ,
});
mmpr::MagAutoOrientation gAutoOrientation;
bool gAutoOrientationEnabled = false;

mmpr::Lsm6TemperatureSensorConfig gImuTempConfig = {
    nodecfg::kImuI2cAddressPrimary7Bit,
    nodecfg::kImuI2cAddressSecondary7Bit,
    nodecfg::kImuTemperatureSampleIntervalMs,
};
mmpr::Lsm6TemperatureSensor gImuTempSensor(gI2c, gImuTempConfig);
mmpr::TemperatureEnvironmentalSource gImuEnvironmentSource(gImuTempSensor);
mmpr::Sht4xEnvironmentalSourceConfig gSht4xEnvironmentConfig = {
    nodecfg::kSht4xI2cAddress7Bit,
    nodecfg::kSht4xSampleIntervalMs,
};
mmpr::Sht4xEnvironmentalSource gSht4xEnvironmentSource(gI2c, gSht4xEnvironmentConfig);
mmpr::FallbackEnvironmentalSource gCombinedEnvironmentSource(
    nodecfg::kEnableSht45Environment ? static_cast<mmpr::IEnvironmentalSource*>(&gSht4xEnvironmentSource) : nullptr,
    nodecfg::kEnableImuTemperature ? static_cast<mmpr::IEnvironmentalSource*>(&gImuEnvironmentSource) : nullptr);
mmpr::NmeaGpsSourceConfig gGpsConfig = {
    uart0,
    nodecfg::kGpsUartBaud,
    nodecfg::kGpsTxPin,
    nodecfg::kGpsRxPin,
    nodecfg::kGpsPpsPin,
    {
        nodecfg::kNodeFallbackLatitudeDeg,
        nodecfg::kNodeFallbackLongitudeDeg,
        nodecfg::kNodeFallbackAltitudeM,
    },
    128,
    nodecfg::kGpsMissingSentenceTimeoutMs,
    nodecfg::kGpsStaleFixTimeoutMs,
};
mmpr::NmeaGpsSource gGpsSource(gGpsConfig);

uint8_t rotateBaseMic(uint8_t micIndex, uint8_t baseRotationSteps) {
  if (micIndex >= 3) {
    return micIndex;
  }
  const uint8_t rot = static_cast<uint8_t>(baseRotationSteps % 3u);
  return static_cast<uint8_t>((micIndex + rot) % 3u);
}

void buildOrderedOffsetsFromSlotMap(uint8_t baseRotationSteps) {
  for (uint8_t channel = 0; channel < kMicCount; ++channel) {
    uint8_t slot = nodecfg::kOutputChannelToSlot[channel];
    if (slot >= kMicCount) {
      std::printf(
          "[sirith-pico] invalid slot map outputChannelToSlot[%u]=%u, using slot 0\n",
          static_cast<unsigned>(channel),
          static_cast<unsigned>(slot));
      slot = 0;
    }

    uint8_t rawMicIndex = nodecfg::kSlotToPhysicalMic[slot];
    if (rawMicIndex >= kMicCount) {
      std::printf(
          "[sirith-pico] invalid mic map slotToPhysicalMic[%u]=%u, using mic 0\n",
          static_cast<unsigned>(slot),
          static_cast<unsigned>(rawMicIndex));
      rawMicIndex = 0;
    }

    const uint8_t rotatedMicIndex = rotateBaseMic(rawMicIndex, baseRotationSteps);
    gSensorOffsetsOrdered[channel] = mmpr::Vec3{
        nodecfg::kPhysicalSensorOffsetsM[rotatedMicIndex][0],
        nodecfg::kPhysicalSensorOffsetsM[rotatedMicIndex][1],
        nodecfg::kPhysicalSensorOffsetsM[rotatedMicIndex][2],
    };

    std::printf(
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
    mmpr::NodeType::kSirithTetra,
    {nodecfg::kNodePositionM[0], nodecfg::kNodePositionM[1], nodecfg::kNodePositionM[2]},
    nodecfg::kNodeHasFallbackGeoPosition,
    {
        nodecfg::kNodeFallbackLatitudeDeg,
        nodecfg::kNodeFallbackLongitudeDeg,
        nodecfg::kNodeFallbackAltitudeM,
    },
    gSensorOffsetsOrdered,
    sizeof(gSensorOffsetsOrdered) / sizeof(gSensorOffsetsOrdered[0]),
    nodecfg::kCapabilities,
    sizeof(nodecfg::kCapabilities) / sizeof(nodecfg::kCapabilities[0]),
    nodecfg::kHardwareName,
    MMPR_FW_VERSION,
    nodecfg::kGpsSignalStatus,
    nodecfg::kGpsPositionSource,
};

mmpr::SirithPicoTdmPins gTdmPins = {
    nodecfg::kTdmDataPin,
    nodecfg::kTdmBclkPin,
    nodecfg::kTdmWsPin,
};

mmpr::SirithPicoTdmConfig gTdmConfig = {
    nodecfg::kAudioSampleRateHz,
    nodecfg::kAudioFrameSamples,
    nodecfg::kAudioSampleShiftBits,
    nodecfg::kAudioTdmSlots,
    nodecfg::kAudioSlotBits,
    {
        nodecfg::kOutputChannelToSlot[0],
        nodecfg::kOutputChannelToSlot[1],
        nodecfg::kOutputChannelToSlot[2],
        nodecfg::kOutputChannelToSlot[3],
    },
    nodecfg::kUseSafeDriveStrength,
};

mmpr::SirithPicoTdmSource gAudioSource(gTdmPins, gTdmConfig);
mmpr::SilenceAudioSource gSilenceAudioSource(
    nodecfg::kAudioSampleRateHz,
    nodecfg::kAudioFrameSamples,
    kMicCount);
mmpr::HttpFramePublisher gPublisher(nodecfg::kServerBaseUrl, nodecfg::kIngestPath, nodecfg::kHttpTimeoutMs);
mmpr::NodeClock gClock;

mmpr::IAudioSource& gSelectedAudioSource = nodecfg::kEnableExternalAudioHardware
    ? static_cast<mmpr::IAudioSource&>(gAudioSource)
    : static_cast<mmpr::IAudioSource&>(gSilenceAudioSource);

mmpr::IEnvironmentalSource* gEnvironmentalSource =
    (nodecfg::kEnableExternalPeripheralBuses &&
     (nodecfg::kEnableSht45Environment || nodecfg::kEnableImuTemperature))
    ? static_cast<mmpr::IEnvironmentalSource*>(&gCombinedEnvironmentSource)
    : nullptr;

mmpr::NodeRunner gRunner(
    gNodeDescriptor,
    gSelectedAudioSource,
    gPublisher,
    gClock,
    nodecfg::kLogEveryFrames,
    gEnvironmentalSource);

void setExternalRailEnabled(bool enabled) {
  gpio_init(nodecfg::kLedPin);
  gpio_set_dir(nodecfg::kLedPin, GPIO_OUT);
  gpio_put(nodecfg::kLedPin, enabled ? 0 : 1);
}

void setStatusLed(bool enabled) {
#ifdef CYW43_WL_GPIO_LED_PIN
  cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, enabled ? 1 : 0);
#else
  (void)enabled;
#endif
}

void setupOptionalPeripherals() {
  const bool useI2c = nodecfg::kEnableCompassAutoOrientation ||
      nodecfg::kEnableImuTemperature ||
      nodecfg::kEnableSht45Environment;
  bool i2cReady = false;
  if (useI2c && nodecfg::kEnableExternalPeripheralBuses) {
    i2cReady = gI2c.begin(i2c1, nodecfg::kI2cSdaPin, nodecfg::kI2cSclPin, nodecfg::kI2cBaudHz);
    if (!i2cReady) {
      std::printf("[sirith-pico] I2C init failed\n");
    }
  } else if (useI2c) {
    std::printf("[sirith-pico] bare-board mode: I2C peripherals disabled\n");
  }

  if (nodecfg::kEnableGpsUart) {
    gGpsSource.begin();
    gGpsSource.poll(gNodeDescriptor, &gClock);
    std::printf("[sirith-pico] GPS UART enabled on uart0 tx=GP%d rx=GP%d pps=GP%d\n",
                nodecfg::kGpsTxPin,
                nodecfg::kGpsRxPin,
                nodecfg::kGpsPpsPin);
  }

  if (nodecfg::kEnableCompassAutoOrientation && i2cReady) {
    mmpr::MagAutoOrientationConfig cfg = {};
    cfg.mode = mmpr::OrientationMode::kAuto;
    cfg.sampleIntervalMs = nodecfg::kCompassSampleIntervalMs;
    cfg.headingOffsetDeg = nodecfg::kCompassHeadingOffsetDeg;
    cfg.minFieldMagnitude = nodecfg::kCompassMinFieldMagnitude;
    cfg.stableSamplesRequired = nodecfg::kCompassStableSamplesRequired;
    cfg.kalmanQ = nodecfg::kCompassKalmanQ;
    cfg.kalmanR = nodecfg::kCompassKalmanR;
    cfg.kalmanInitialP = nodecfg::kCompassKalmanInitP;

    gAutoOrientationEnabled = gAutoOrientation.begin(
        gMagnetometer, cfg, gActiveBaseRotationSteps);
    if (gAutoOrientationEnabled) {
      std::printf("[sirith-pico] auto-orientation enabled (Kalman Q=%.4f R=%.1f)\n",
                  static_cast<double>(cfg.kalmanQ),
                  static_cast<double>(cfg.kalmanR));
    } else {
      std::printf("[sirith-pico] magnetometer unavailable; manual rotation retained\n");
    }
  }

  if (nodecfg::kEnableImuTemperature) {
    std::printf("[sirith-pico] IMU temperature telemetry enabled (generic source interface)\n");
  }
  if (nodecfg::kEnableSht45Environment) {
    std::printf("[sirith-pico] SHT4x environmental telemetry enabled (temperature + humidity)\n");
  }
}

}  // namespace

int main() {
  stdio_init_all();
  sleep_ms(300);

  // GP26 controls the external Vin FET rail; keep it off unless explicitly enabled.
  setExternalRailEnabled(nodecfg::kEnableExternalVFetRail);

  std::printf("[sirith-pico] booting\n");
  if (nodecfg::kBareBoardValidationMode) {
    std::printf("[sirith-pico] bare-board validation mode enabled\n");
  }

  if (cyw43_arch_init()) {
    std::printf("[sirith-pico] fatal: Wi-Fi init failed\n");
    while (true) {
      sleep_ms(1000);
    }
  }

  setStatusLed(true);
  buildOrderedOffsetsFromSlotMap(gActiveBaseRotationSteps);
  setupOptionalPeripherals();

  const bool wifiConnected = mmpr::connectWiFiBlocking(
      nodecfg::kWifiSsid,
      nodecfg::kWifiPassword,
      nodecfg::kWiFiConnectTimeoutMs);

  if (!wifiConnected) {
    std::printf("[sirith-pico] WiFi not connected at startup, will keep retrying\n");
  } else {
    std::printf("[sirith-pico] WiFi connected\n");
  }

  const bool started = gRunner.begin(
      nodecfg::kEnableNtpSync,
      nodecfg::kNtpServer,
      nodecfg::kGmtOffsetSeconds,
      nodecfg::kDaylightOffsetSeconds);

  if (!started) {
    std::printf("[sirith-pico] fatal: runner failed to start\n");
    while (true) {
      cyw43_arch_poll();
      sleep_ms(1000);
    }
  }

  while (true) {
    if (gAutoOrientationEnabled) {
      uint8_t changedRotation = 0;
      if (gAutoOrientation.poll(&changedRotation)) {
        gActiveBaseRotationSteps = changedRotation;
        std::printf(
            "[sirith-pico] auto-rotation -> step=%u heading=%.1f deg\n",
            static_cast<unsigned>(gActiveBaseRotationSteps),
            static_cast<double>(gAutoOrientation.headingDeg()));
        buildOrderedOffsetsFromSlotMap(gActiveBaseRotationSteps);
      } else if (!gAutoOrientation.healthy()) {
        gAutoOrientationEnabled = false;
        std::printf("[sirith-pico] magnetometer read fault; holding current manual rotation\n");
      }
    }

    mmpr::ensureWiFiConnected(
        nodecfg::kWifiSsid,
        nodecfg::kWifiPassword,
        nodecfg::kWiFiConnectTimeoutMs);

    gRunner.loopOnce();

    if (nodecfg::kEnableGpsUart) {
      gGpsSource.poll(gNodeDescriptor, &gClock);
    }

    // Heartbeat blink on GP26 (P-channel FET: LOW=on, HIGH=off).
    static uint32_t ledCounter = 0;
    static bool ledState = false;
    if (++ledCounter >= nodecfg::kLedBlinkFrames) {
      ledCounter = 0;
      ledState = !ledState;
      setStatusLed(ledState);
    }

    cyw43_arch_poll();
  }
}
