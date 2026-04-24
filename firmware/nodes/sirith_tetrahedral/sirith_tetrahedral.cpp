#include <cstdint>
#include <cstdio>

#include "pico/cyw43_arch.h"
#include "pico/stdlib.h"
#include "pico/unique_id.h"
#include "hardware/pwm.h"
#include "hardware/uart.h"

#include "node_config.h"

#include "mmpr/HttpFramePublisher.h"
#include "mmpr/NtpClient.h"
#include "mmpr/FallbackEnvironmentalSource.h"
#include "mmpr/FailureSnapshot.h"
#include "mmpr/Lis2mdlMagnetometer.h"
#include "mmpr/Lsm6TemperatureSensor.h"
#include "mmpr/MagAutoOrientation.h"
#include "mmpr/NmeaGpsSource.h"
#include "mmpr/NodeClock.h"
#include "mmpr/NodeRunner.h"
#include "mmpr/PicoI2SMonoSource.h"
#include "mmpr/Sht4xEnvironmentalSource.h"
#include "mmpr/SilenceAudioSource.h"
#include "mmpr/SirithPicoTdmSource.h"
#include "mmpr/TemperatureEnvironmentalSource.h"
#include "mmpr/WiFiSupport.h"

#ifndef MMPR_FW_VERSION
#define MMPR_FW_VERSION "dev"
#endif

namespace {

constexpr uint8_t kTetraMicCount = 4;

// --- GP27 I2C activity LED ---
// PWM wrap gives 100 duty steps; hardware runs the carrier autonomously.
constexpr uint32_t kActivityLedWrap = 99;
constexpr uint32_t kActivityLedDimLevel =
    (static_cast<uint32_t>(nodecfg::kActivityLedDimPercent) * (kActivityLedWrap + 1)) / 100;

volatile uint32_t gActivityLastReadMs = 0;
volatile bool gActivityPulsed = false;

mmpr::PicoSerialSampleEdge toPicoSerialSampleEdge(nodecfg::AudioSerialSampleEdge sampleEdge) {
  return sampleEdge == nodecfg::AudioSerialSampleEdge::kFalling
      ? mmpr::PicoSerialSampleEdge::kFalling
      : mmpr::PicoSerialSampleEdge::kRising;
}

mmpr::PicoSerialDataPinBias toPicoSerialDataPinBias(nodecfg::AudioDataPinBias dataPinBias) {
  return dataPinBias == nodecfg::AudioDataPinBias::kPullDown
      ? mmpr::PicoSerialDataPinBias::kPullDown
      : mmpr::PicoSerialDataPinBias::kDisabled;
}

const char* sampleEdgeName(nodecfg::AudioSerialSampleEdge sampleEdge) {
  return sampleEdge == nodecfg::AudioSerialSampleEdge::kFalling ? "falling" : "rising";
}

const char* dataPinBiasName(nodecfg::AudioDataPinBias dataPinBias) {
  return dataPinBias == nodecfg::AudioDataPinBias::kPullDown ? "pull_down" : "disabled";
}

void setupActivityLed() {
  gpio_set_function(nodecfg::kActivityLedPin, GPIO_FUNC_PWM);
  const uint slice = pwm_gpio_to_slice_num(nodecfg::kActivityLedPin);
  const uint channel = pwm_gpio_to_channel(nodecfg::kActivityLedPin);
  pwm_set_wrap(slice, kActivityLedWrap);
  pwm_set_chan_level(slice, channel, kActivityLedDimLevel);
  pwm_set_enabled(slice, true);
}

// Called from I2cBus::read() on each successful read — keeps to a few cycles.
void onI2cRead() {
  const uint slice = pwm_gpio_to_slice_num(nodecfg::kActivityLedPin);
  const uint channel = pwm_gpio_to_channel(nodecfg::kActivityLedPin);
  pwm_set_chan_level(slice, channel, kActivityLedWrap);  // full brightness
  gActivityLastReadMs = to_ms_since_boot(get_absolute_time());
  gActivityPulsed = true;
}

// Call once per main loop iteration — one timestamp comparison, no I/O unless dimming.
void pollActivityLed() {
  if (!gActivityPulsed) {
    return;
  }
  const uint32_t nowMs = to_ms_since_boot(get_absolute_time());
  if ((nowMs - gActivityLastReadMs) >= nodecfg::kActivityLedPulseMs) {
    const uint slice = pwm_gpio_to_slice_num(nodecfg::kActivityLedPin);
    const uint channel = pwm_gpio_to_channel(nodecfg::kActivityLedPin);
    pwm_set_chan_level(slice, channel, kActivityLedDimLevel);
    gActivityPulsed = false;
  }
}

mmpr::Vec3 gTetraSensorOffsetsOrdered[kTetraMicCount] = {};
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
  for (uint8_t channel = 0; channel < kTetraMicCount; ++channel) {
    uint8_t slot = nodecfg::kOutputChannelToSlot[channel];
    if (slot >= kTetraMicCount) {
      std::printf(
          "[sirith-pico] invalid slot map outputChannelToSlot[%u]=%u, using slot 0\n",
          static_cast<unsigned>(channel),
          static_cast<unsigned>(slot));
      slot = 0;
    }

    uint8_t rawMicIndex = nodecfg::kSlotToPhysicalMic[slot];
    if (rawMicIndex >= kTetraMicCount) {
      std::printf(
          "[sirith-pico] invalid mic map slotToPhysicalMic[%u]=%u, using mic 0\n",
          static_cast<unsigned>(slot),
          static_cast<unsigned>(rawMicIndex));
      rawMicIndex = 0;
    }

    const uint8_t rotatedMicIndex = rotateBaseMic(rawMicIndex, baseRotationSteps);
    gTetraSensorOffsetsOrdered[channel] = nodecfg::kPhysicalSensorOffsetsM[rotatedMicIndex];

    std::printf(
        "[sirith-pico] ch%u <- slot%u <- mic%u (rot=%u base=%u)\n",
        static_cast<unsigned>(channel),
        static_cast<unsigned>(slot + 1),
        static_cast<unsigned>(rawMicIndex),
        static_cast<unsigned>(rotatedMicIndex),
        static_cast<unsigned>(baseRotationSteps));
  }
}

const mmpr::Vec3* selectedSensorOffsets() {
  return nodecfg::kUseTdmAudio ? gTetraSensorOffsetsOrdered : nodecfg::kPointSensorOffsetsM;
}

size_t selectedSensorCount() {
  return nodecfg::kUseTdmAudio
      ? (sizeof(gTetraSensorOffsetsOrdered) / sizeof(gTetraSensorOffsetsOrdered[0]))
      : (sizeof(nodecfg::kPointSensorOffsetsM) / sizeof(nodecfg::kPointSensorOffsetsM[0]));
}

// Runtime node ID: prefix + last 4 hex digits of the Pico's unique board ID.
// Filled by initNodeId() before anything references gNodeDescriptor.
static char gNodeIdBuf[64] = {};

void initNodeId() {
  pico_unique_board_id_t uid;
  pico_get_unique_board_id(&uid);
  // Use last 2 bytes (4 hex chars) for a short but unique suffix.
  std::snprintf(gNodeIdBuf, sizeof(gNodeIdBuf), "%s%02x%02x",
                nodecfg::kNodeIdPrefix,
                uid.id[PICO_UNIQUE_BOARD_ID_SIZE_BYTES - 2],
                uid.id[PICO_UNIQUE_BOARD_ID_SIZE_BYTES - 1]);
}

mmpr::NodeDescriptor gNodeDescriptor = {
    gNodeIdBuf,  // populated by initNodeId() at boot
    nodecfg::kNodeType,
    nodecfg::kNodePositionM,
    nodecfg::kNodeHasFallbackGeoPosition,
    {
        nodecfg::kNodeFallbackLatitudeDeg,
        nodecfg::kNodeFallbackLongitudeDeg,
        nodecfg::kNodeFallbackAltitudeM,
    },
    selectedSensorOffsets(),
    selectedSensorCount(),
    nodecfg::kCapabilities,
    nodecfg::kCapabilityCount,
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
    nodecfg::kAudioTdmSlots,
    nodecfg::kAudioSlotBits,
    nodecfg::kAudioValidBits,
    toPicoSerialSampleEdge(nodecfg::kAudioTdmSampleEdge),
    nodecfg::kAudioTdmCaptureBitOffset,
    toPicoSerialDataPinBias(nodecfg::kAudioTdmDataPinBias),
    nodecfg::kAudioTdmEnableWordDiagnostics,
    {
        nodecfg::kOutputChannelToSlot[0],
        nodecfg::kOutputChannelToSlot[1],
        nodecfg::kOutputChannelToSlot[2],
        nodecfg::kOutputChannelToSlot[3],
    },
    nodecfg::kUseSafeDriveStrength,
};

mmpr::PicoI2SMonoPins gI2sMonoPins = {
    nodecfg::kI2sMonoDataPin,
    nodecfg::kI2sMonoBclkPin,
    nodecfg::kI2sMonoWsPin,
};

mmpr::PicoI2SMonoConfig gI2sMonoConfig = {
    nodecfg::kI2sMonoSampleRateHz,
    nodecfg::kI2sMonoFrameSamples,
    nodecfg::kI2sMonoSlotBits,
    nodecfg::kI2sMonoValidBits,
    nodecfg::kI2sMonoChannelSide == nodecfg::I2sMonoChannelSide::kRight
        ? mmpr::PicoI2SChannelSide::kRight
        : mmpr::PicoI2SChannelSide::kLeft,
    toPicoSerialSampleEdge(nodecfg::kAudioI2sMonoSampleEdge),
    nodecfg::kAudioI2sMonoCaptureBitOffset,
    toPicoSerialDataPinBias(nodecfg::kAudioI2sMonoDataPinBias),
    nodecfg::kAudioI2sMonoEnableWordDiagnostics,
    nodecfg::kUseSafeDriveStrength,
};

mmpr::SirithPicoTdmSource gTdmAudioSource(gTdmPins, gTdmConfig);
mmpr::PicoI2SMonoSource gI2sMonoAudioSource(gI2sMonoPins, gI2sMonoConfig);
mmpr::SilenceAudioSource gSilenceAudioSource(
    nodecfg::kActiveAudioSampleRateHz,
    nodecfg::kActiveAudioFrameSamples,
    nodecfg::kActiveAudioChannels);
mmpr::HttpFramePublisher gPublisher(nodecfg::kServerBaseUrl, nodecfg::kIngestPath, nodecfg::kHttpTimeoutMs);
mmpr::NodeClock gClock;

mmpr::IAudioSource& gConfiguredAudioSource = nodecfg::kUseTdmAudio
    ? static_cast<mmpr::IAudioSource&>(gTdmAudioSource)
    : static_cast<mmpr::IAudioSource&>(gI2sMonoAudioSource);

mmpr::IAudioSource& gSelectedAudioSource = nodecfg::kEnableExternalAudioHardware
    ? gConfiguredAudioSource
    : static_cast<mmpr::IAudioSource&>(gSilenceAudioSource);

mmpr::IEnvironmentalSource* gEnvironmentalSource =
    (nodecfg::kEnableExternalPeripheralBuses &&
     (nodecfg::kEnableSht45Environment || nodecfg::kEnableImuTemperature))
    ? static_cast<mmpr::IEnvironmentalSource*>(&gCombinedEnvironmentSource)
    : nullptr;

mmpr::NtpClient gNtpClient;

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
  const bool useAutoOrientation = nodecfg::kUseTdmAudio && nodecfg::kEnableCompassAutoOrientation;
  const bool useI2c = useAutoOrientation ||
      nodecfg::kEnableImuTemperature ||
      nodecfg::kEnableSht45Environment;
  bool i2cReady = false;
  if (useI2c && nodecfg::kEnableExternalPeripheralBuses) {
    i2cReady = gI2c.begin(i2c1, nodecfg::kI2cSdaPin, nodecfg::kI2cSclPin, nodecfg::kI2cBaudHz);
    if (!i2cReady) {
      std::printf("[sirith-pico] I2C init failed\n");
    } else {
      gI2c.setReadCallback(onI2cRead);
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

  if (nodecfg::kEnableCompassAutoOrientation && !nodecfg::kUseTdmAudio) {
    std::printf("[sirith-pico] auto-orientation ignored in mono-I2S point mode\n");
  }

  if (useAutoOrientation && i2cReady) {
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
  initNodeId();
  stdio_init_all();
  sleep_ms(300);
  mmpr::FailureSnapshot::initializeForBoot();
  mmpr::FailureSnapshot::feedWatchdog();

  // GP26 controls the external Vin FET rail; keep it off unless explicitly enabled.
  setExternalRailEnabled(nodecfg::kEnableExternalVFetRail);

  // GP27 activity LED: hardware PWM dims it to ~kActivityLedDimPercent% duty.
  setupActivityLed();

  std::printf(
      "[sirith-pico] booting id=%s mode=%s hardware=%s channels=%u\n",
      gNodeDescriptor.id,
      nodecfg::kUseTdmAudio ? "tdm4" : "i2s_mono",
      nodecfg::kHardwareName,
      static_cast<unsigned>(nodecfg::kActiveAudioChannels));
  if (nodecfg::kBareBoardValidationMode) {
    std::printf("[sirith-pico] bare-board validation mode enabled\n");
  }

  mmpr::FailureSnapshot::updatePhase(mmpr::FatalLifecyclePhase::kWiFiInit);
  if (cyw43_arch_init()) {
    std::printf("[sirith-pico] fatal: Wi-Fi init failed\n");
    while (true) {
      sleep_ms(1000);
    }
  }

  mmpr::FailureSnapshot::updatePhase(mmpr::FatalLifecyclePhase::kPeripheralInit);
  mmpr::FailureSnapshot::feedWatchdog();
  setStatusLed(true);
  if (nodecfg::kUseTdmAudio) {
    buildOrderedOffsetsFromSlotMap(gActiveBaseRotationSteps);
    std::printf(
        "[sirith-pico] tdm timing edge=%s bit_offset=%d bias=%s diag=%u\n",
        sampleEdgeName(nodecfg::kAudioTdmSampleEdge),
        static_cast<int>(nodecfg::kAudioTdmCaptureBitOffset),
        dataPinBiasName(nodecfg::kAudioTdmDataPinBias),
        static_cast<unsigned>(nodecfg::kAudioTdmEnableWordDiagnostics ? 1u : 0u));
  } else {
    std::printf(
        "[sirith-pico] mono-I2S point mode channel=%s offset=[0,0,0] edge=%s bit_offset=%d bias=%s diag=%u\n",
        nodecfg::kI2sMonoChannelSide == nodecfg::I2sMonoChannelSide::kRight ? "right" : "left",
        sampleEdgeName(nodecfg::kAudioI2sMonoSampleEdge),
        static_cast<int>(nodecfg::kAudioI2sMonoCaptureBitOffset),
        dataPinBiasName(nodecfg::kAudioI2sMonoDataPinBias),
        static_cast<unsigned>(nodecfg::kAudioI2sMonoEnableWordDiagnostics ? 1u : 0u));
    if (nodecfg::kI2sMonoPinsAliasTdmPins) {
      std::printf(
          "[sirith-pico] warning: mono-I2S pins still alias TDM pins (data=GP%u bclk=GP%u ws=GP%u); "
          "set board-specific I2S lane pins before using mono mode for hardware validation\n",
          static_cast<unsigned>(nodecfg::kI2sMonoDataPin),
          static_cast<unsigned>(nodecfg::kI2sMonoBclkPin),
          static_cast<unsigned>(nodecfg::kI2sMonoWsPin));
    }
  }
  setupOptionalPeripherals();
  mmpr::FailureSnapshot::feedWatchdog();

  mmpr::FailureSnapshot::updatePhase(mmpr::FatalLifecyclePhase::kStartupWiFiConnect);
  const bool wifiConnected = mmpr::connectWiFiBlocking(
      nodecfg::kWifiSsid,
      nodecfg::kWifiPassword,
      nodecfg::kWiFiConnectTimeoutMs);
  mmpr::FailureSnapshot::feedWatchdog();

  if (!wifiConnected) {
    std::printf("[sirith-pico] WiFi not connected at startup, will keep retrying\n");
  } else {
    std::printf("[sirith-pico] WiFi connected\n");
  }

  mmpr::FailureSnapshot::updatePhase(mmpr::FatalLifecyclePhase::kRunnerStart);
  const bool started = gRunner.begin(
      nodecfg::kEnableNtpSync,
      nodecfg::kNtpServer,
      nodecfg::kGmtOffsetSeconds,
      nodecfg::kDaylightOffsetSeconds);
  mmpr::FailureSnapshot::feedWatchdog();

  if (!started) {
    std::printf("[sirith-pico] fatal: runner failed to start\n");
    while (true) {
      cyw43_arch_poll();
      sleep_ms(1000);
    }
  }

  if (nodecfg::kEnableGpsUart) {
    gGpsSource.bindAudioSource(&gSelectedAudioSource);
  }

  if (nodecfg::kEnableNtpSync) {
    mmpr::FailureSnapshot::updatePhase(mmpr::FatalLifecyclePhase::kNtpStart);
    gNtpClient.begin(nodecfg::kNtpServer, gClock);
    std::printf("[sirith-pico] NTP client started, server=%s\n", nodecfg::kNtpServer);
    mmpr::FailureSnapshot::feedWatchdog();
  }

  mmpr::FailureSnapshot::updatePhase(mmpr::FatalLifecyclePhase::kMainLoop);
  while (true) {
    if (nodecfg::kUseTdmAudio && gAutoOrientationEnabled) {
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
    mmpr::FailureSnapshot::updateProgressMarker(static_cast<uint32_t>(gRunner.stats().framesCaptured));
    pollActivityLed();

    if (nodecfg::kEnableGpsUart) {
      gGpsSource.poll(gNodeDescriptor, &gClock);
    }

    if (nodecfg::kEnableNtpSync) {
      gNtpClient.poll();
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
    mmpr::FailureSnapshot::feedWatchdog();
  }
}
