#include <cstdint>
#include <cstdio>

#include "pico/cyw43_arch.h"
#include "pico/stdlib.h"
#include "hardware/uart.h"

#include "node_config.h"

#include "mmpr/HttpFramePublisher.h"
#include "mmpr/Lis2mdlAutoOrientation.h"
#include "mmpr/Lsm6TemperatureSensor.h"
#include "mmpr/NodeClock.h"
#include "mmpr/NodeRunner.h"
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
mmpr::Lis2mdlAutoOrientation gAutoOrientation;
bool gAutoOrientationEnabled = false;

mmpr::Lsm6TemperatureSensorConfig gImuTempConfig = {
    nodecfg::kImuI2cAddressPrimary7Bit,
    nodecfg::kImuI2cAddressSecondary7Bit,
    nodecfg::kImuTemperatureSampleIntervalMs,
};
mmpr::Lsm6TemperatureSensor gImuTempSensor(gI2c, gImuTempConfig);
mmpr::TemperatureEnvironmentalSource gImuEnvironmentSource(gImuTempSensor);

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
    gSensorOffsetsOrdered,
    sizeof(gSensorOffsetsOrdered) / sizeof(gSensorOffsetsOrdered[0]),
    nodecfg::kCapabilities,
    sizeof(nodecfg::kCapabilities) / sizeof(nodecfg::kCapabilities[0]),
    nodecfg::kHardwareName,
    MMPR_FW_VERSION,
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
mmpr::HttpFramePublisher gPublisher(nodecfg::kServerBaseUrl, nodecfg::kIngestPath, nodecfg::kHttpTimeoutMs);
mmpr::NodeClock gClock;

mmpr::IEnvironmentalSource* gEnvironmentalSource = nodecfg::kEnableImuTemperature
    ? static_cast<mmpr::IEnvironmentalSource*>(&gImuEnvironmentSource)
    : nullptr;

mmpr::NodeRunner gRunner(
    gNodeDescriptor,
    gAudioSource,
    gPublisher,
    gClock,
    nodecfg::kLogEveryFrames,
    gEnvironmentalSource);

void setupOptionalPeripherals() {
  const bool useI2c = nodecfg::kEnableCompassAutoOrientation || nodecfg::kEnableImuTemperature;
  bool i2cReady = false;
  if (useI2c) {
    i2cReady = gI2c.begin(i2c1, nodecfg::kI2cSdaPin, nodecfg::kI2cSclPin, nodecfg::kI2cBaudHz);
    if (!i2cReady) {
      std::printf("[sirith-pico] I2C init failed\n");
    }
  }

  if (nodecfg::kEnableGpsUart) {
    uart_init(uart0, nodecfg::kGpsUartBaud);
    gpio_set_function(nodecfg::kGpsTxPin, GPIO_FUNC_UART);
    gpio_set_function(nodecfg::kGpsRxPin, GPIO_FUNC_UART);

    gpio_init(nodecfg::kGpsPpsPin);
    gpio_set_dir(nodecfg::kGpsPpsPin, GPIO_IN);
    gpio_pull_down(nodecfg::kGpsPpsPin);
    std::printf("[sirith-pico] GPS UART enabled\n");
  }

  if (nodecfg::kEnableCompassAutoOrientation && i2cReady) {
    mmpr::Lis2mdlAutoOrientationConfig cfg = {};
    cfg.i2cAddress7Bit = nodecfg::kCompassI2cAddress7Bit;
    cfg.outputDataRateBits = nodecfg::kCompassOutputDataRateBits;
    cfg.sampleIntervalMs = nodecfg::kCompassSampleIntervalMs;
    cfg.smoothingAlpha = nodecfg::kCompassSmoothingAlpha;
    cfg.headingOffsetDeg = nodecfg::kCompassHeadingOffsetDeg;
    cfg.minHorizontalFieldLsb = nodecfg::kCompassMinHorizontalFieldLsb;
    cfg.stableSamplesRequired = nodecfg::kCompassStableSamplesRequired;

    gAutoOrientationEnabled = gAutoOrientation.begin(gI2c, cfg, gActiveBaseRotationSteps);
    if (gAutoOrientationEnabled) {
      std::printf("[sirith-pico] LIS2MDLTR auto-orientation enabled\n");
    } else {
      std::printf("[sirith-pico] LIS2MDLTR auto-orientation unavailable; manual rotation retained\n");
    }
  }

  if (nodecfg::kEnableImuTemperature) {
    std::printf("[sirith-pico] IMU temperature telemetry enabled (generic source interface)\n");
  }
}

}  // namespace

int main() {
  stdio_init_all();
  sleep_ms(300);

  // LED / status FET on GP26: LOW = FET on (LED lit).
  gpio_init(nodecfg::kLedPin);
  gpio_set_dir(nodecfg::kLedPin, GPIO_OUT);
  gpio_put(nodecfg::kLedPin, 0);  // LED on at boot

  std::printf("[sirith-pico] booting\n");

  if (cyw43_arch_init()) {
    std::printf("[sirith-pico] fatal: Wi-Fi init failed\n");
    while (true) {
      sleep_ms(1000);
    }
  }

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
        std::printf("[sirith-pico] LIS2MDLTR read fault; holding current manual rotation\n");
      }
    }

    mmpr::ensureWiFiConnected(
        nodecfg::kWifiSsid,
        nodecfg::kWifiPassword,
        nodecfg::kWiFiConnectTimeoutMs);

    gRunner.loopOnce();

    // Heartbeat blink on GP26 (P-channel FET: LOW=on, HIGH=off).
    static uint32_t ledCounter = 0;
    static bool ledState = false;
    if (++ledCounter >= nodecfg::kLedBlinkFrames) {
      ledCounter = 0;
      ledState = !ledState;
      gpio_put(nodecfg::kLedPin, ledState ? 1 : 0);
    }

    cyw43_arch_poll();
  }
}
