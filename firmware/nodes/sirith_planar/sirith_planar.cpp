// sirith_planar — 5-mic PDM planar array node (RP2354A).
//
// Phase 1 skeleton: GPS + PPS + NodeClock fully live; audio is synthetic until
// PdmPlanarSource lands (Phase 2). The uplink is EspC5Publisher (Phase 3, D6):
// an HTTP POST proxied over SPI to an ESP32-C5 co-processor, framed per
// EspC5Frame.h. NodeRunner/NodeProtocol/backend are unaware of the SPI
// indirection -- see minimap_transport_espc5. No CYW43/lwIP/btstack in this
// build — the shared runtime is radio-free and the Wi-Fi radio is off-board on
// the ESP32-C5. HARDWARE-DEPENDENT / UNVERIFIED WITHOUT A BENCH: real SPI bus
// timing and the ESP32-C5 bridge firmware's DATA_POST/POST_STATUS handling
// have not been exercised against actual hardware (see SpiHostLink.h /
// EspC5Publisher.h for specifics); the framing/state-machine logic is covered
// by host unit tests in minimap_transport_espc5/tests/host.

#include <cstdint>
#include <cstdio>

#include "pico/stdlib.h"
#include "pico/unique_id.h"
#include "hardware/clocks.h"
#include "hardware/pwm.h"

#include "node_config.h"

#include "mmpr/EspC5Publisher.h"
#include "mmpr/FailureSnapshot.h"
#include "mmpr/FallbackEnvironmentalSource.h"
#include "mmpr/IUplinkTransport.h"
#include "mmpr/Lis2mdlMagnetometer.h"
#include "mmpr/Lsm6TemperatureSensor.h"
#include "mmpr/MagAutoOrientation.h"
#include "mmpr/NmeaGpsSource.h"
#include "mmpr/NodeClock.h"
#include "mmpr/NodeRunner.h"
#include "mmpr/PdmPlanarSource.h"
#include "mmpr/PiezoBuzzer.h"
#include "mmpr/Sht4xEnvironmentalSource.h"
#include "mmpr/SpiHostLink.h"
#include "mmpr/SyntheticAudioSource.h"
#include "mmpr/TemperatureEnvironmentalSource.h"

#ifndef MMPR_FW_VERSION
#define MMPR_FW_VERSION "dev"
#endif

namespace {

// --- GP27 I2C activity LED (hardware PWM dim) ---
constexpr uint32_t kActivityLedWrap = 99;
constexpr uint32_t kActivityLedDimLevel =
    (static_cast<uint32_t>(nodecfg::kActivityLedDimPercent) * (kActivityLedWrap + 1)) / 100;
volatile uint32_t gActivityLastReadMs = 0;
volatile bool gActivityPulsed = false;

void setupActivityLed() {
  gpio_set_function(nodecfg::kActivityLedPin, GPIO_FUNC_PWM);
  const uint slice = pwm_gpio_to_slice_num(nodecfg::kActivityLedPin);
  const uint channel = pwm_gpio_to_channel(nodecfg::kActivityLedPin);
  pwm_set_wrap(slice, kActivityLedWrap);
  pwm_set_chan_level(slice, channel, kActivityLedDimLevel);
  pwm_set_enabled(slice, true);
}

void onI2cRead() {
  const uint slice = pwm_gpio_to_slice_num(nodecfg::kActivityLedPin);
  const uint channel = pwm_gpio_to_channel(nodecfg::kActivityLedPin);
  pwm_set_chan_level(slice, channel, kActivityLedWrap);
  gActivityLastReadMs = to_ms_since_boot(get_absolute_time());
  gActivityPulsed = true;
}

void pollActivityLed() {
  if (!gActivityPulsed) return;
  const uint32_t nowMs = to_ms_since_boot(get_absolute_time());
  if ((nowMs - gActivityLastReadMs) >= nodecfg::kActivityLedPulseMs) {
    const uint slice = pwm_gpio_to_slice_num(nodecfg::kActivityLedPin);
    const uint channel = pwm_gpio_to_channel(nodecfg::kActivityLedPin);
    pwm_set_chan_level(slice, channel, kActivityLedDimLevel);
    gActivityPulsed = false;
  }
}

// GP26 status LED (simple GPIO on this board — no CYW43 WL GPIO).
void setStatusLed(bool enabled) {
  gpio_put(nodecfg::kLedPin, enabled ? 0 : 1);  // active-low
}

void setupStatusLed() {
  gpio_init(nodecfg::kLedPin);
  gpio_set_dir(nodecfg::kLedPin, GPIO_OUT);
  setStatusLed(false);
}

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
mmpr::PiezoBuzzer gBuzzer(nodecfg::kBuzzerPwmPin);
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
    512,
    nodecfg::kGpsMissingSentenceTimeoutMs,
    nodecfg::kGpsStaleFixTimeoutMs,
    nodecfg::kGpsBaudScanIntervalMs,
};
mmpr::NmeaGpsSource gGpsSource(gGpsConfig);

static char gNodeIdBuf[64] = {};
void initNodeId() {
  pico_unique_board_id_t uid;
  pico_get_unique_board_id(&uid);
  std::snprintf(gNodeIdBuf, sizeof(gNodeIdBuf), "%s%02x%02x",
                nodecfg::kNodeIdPrefix,
                uid.id[PICO_UNIQUE_BOARD_ID_SIZE_BYTES - 2],
                uid.id[PICO_UNIQUE_BOARD_ID_SIZE_BYTES - 1]);
}

mmpr::NodeDescriptor gNodeDescriptor = {
    gNodeIdBuf,
    nodecfg::kNodeType,
    nodecfg::kNodeHasFallbackGeoPosition,
    {
        nodecfg::kNodeFallbackLatitudeDeg,
        nodecfg::kNodeFallbackLongitudeDeg,
        nodecfg::kNodeFallbackAltitudeM,
    },
    nodecfg::kSensorOffsetsM,
    nodecfg::kSensorOffsetCount,
    nodecfg::kCapabilities,
    nodecfg::kCapabilityCount,
    nodecfg::kHardwareName,
    MMPR_FW_VERSION,
    nodecfg::kGpsSignalStatus,
    nodecfg::kGpsPositionSource,
    0u,
};

// Phase 2: real PDM capture (PIO+DMA+core-1 CIC/halfband decimation). Synthetic
// audio is kept for bring-up without PDM hardware (MMPR_NODECFG_AUDIO_INPUT_MODE=3,
// see node_config.h) since PdmPlanarSource has not been bench-verified (real PIO
// timing / mic data-valid window / multicore hand-off latency are all
// HARDWARE-DEPENDENT / UNVERIFIED -- see PDM_DESIGN.md). The CIC/halfband math
// itself (PdmCicDecimator) IS host-tested and bit-exact-verified; see
// nodes/sirith_planar/tests/host/test_pdm_cic_decimator.cpp.
mmpr::SyntheticAudioSource gSyntheticAudioSource(
    nodecfg::kActiveAudioSampleRateHz,
    nodecfg::kActiveAudioFrameSamples,
    nodecfg::kActiveAudioChannels);

mmpr::PdmPlanarPins gPdmPins = {
    nodecfg::kPdmDataPinBase,
    nodecfg::kPdmDataPinCount,
    nodecfg::kPdmClockPin,
};
mmpr::PdmPlanarConfig gPdmConfig = {
    nodecfg::kActiveAudioSampleRateHz,
    nodecfg::kActiveAudioFrameSamples,
    nodecfg::kAudioRingFrames,
    /*enableDither=*/true,
};
mmpr::PdmPlanarSource gPdmPlanarSource(gPdmPins, gPdmConfig);

// Selected once at static-init time from the compile-time nodecfg flag; both
// objects exist either way (cheap), but only the active one's begin() is
// ever called (see main()).
mmpr::IAudioSource& gActiveAudioSource = nodecfg::kUsePdmAudio
    ? static_cast<mmpr::IAudioSource&>(gPdmPlanarSource)
    : static_cast<mmpr::IAudioSource&>(gSyntheticAudioSource);

// ESP32-C5 SPI bridge uplink (Phase 3, D6): HTTP POST proxied over SPI. See
// minimap_transport_espc5/include/mmpr/{SpiHostLink,EspC5Publisher}.h for the
// hardware-dependent caveats (real bus timing / C5 bridge firmware behavior
// are unverified without a bench).
mmpr::SpiHostLinkConfig gSpiLinkConfig = {
    spi0,
    4 * 1000 * 1000,
    nodecfg::kEspSpiSckPin,
    nodecfg::kEspSpiTxPin,
    nodecfg::kEspSpiRxPin,
    nodecfg::kEspSpiCsPin,
    nodecfg::kEspHostWakePin,
};
mmpr::SpiHostLink gSpiLink(gSpiLinkConfig);
mmpr::EspC5Publisher gPublisher(nodecfg::kIngestPath, gSpiLink, nodecfg::kHttpTimeoutMs);
mmpr::NodeClock gClock;

mmpr::IEnvironmentalSource* gEnvironmentalSource =
    (nodecfg::kEnableExternalPeripheralBuses &&
     (nodecfg::kEnableSht45Environment || nodecfg::kEnableImuTemperature))
    ? static_cast<mmpr::IEnvironmentalSource*>(&gCombinedEnvironmentSource)
    : nullptr;

mmpr::NodeRunner gRunner(
    gNodeDescriptor,
    gActiveAudioSource,
    gPublisher,
    gClock,
    nodecfg::kLogEveryFrames,
    gEnvironmentalSource,
    nodecfg::kMaxPacketSamplesPerChannel,
    nodecfg::kPublishFailureBackoffMs,
    nodecfg::kPublishBatchFrames,
    nodecfg::kPublishBatchByteBudget,
    nodecfg::kUsePublishBatchByteBudget,
    nodecfg::kAudioQueueSlots,
    nodecfg::kPublishBatchMaxRetries,
    nodecfg::kEnableClockHoldoverMaintenance);

mmpr::GpsRuntimeStats gGpsRuntimeStats = {};

void setClockForPdm() {
  // Bench (Pico 2 W, 12 MHz XO): 153.6 MHz = 50 x 3.072 MHz gives a jitter-free
  // integer PDM divider (2.4% overclock, flagged -- see D2/PDM_DESIGN.md). The
  // final RP2354A board uses a 12.288 MHz TCXO -> 122.88 MHz = 40 x 3.072 MHz,
  // requiring no overclock. Set this BEFORE any PIO init so downstream clock
  // math (PPS/holdover, PdmPlanarSource's PIO cycle-count derivation) reads
  // the real clk_sys via clock_get_hz(clk_sys), never a hardcoded constant.
  //
  // HARDWARE-DEPENDENT / UNVERIFIED WITHOUT A BENCH: the 153.6 MHz bench
  // overclock has not been validated against this board's flash timing (see
  // Risk #3 in the firmware plan); synthetic-audio bring-up
  // (MMPR_NODECFG_AUDIO_INPUT_MODE=3) intentionally skips this override.
  if (nodecfg::kUsePdmAudio) {
    const bool ok = set_sys_clock_khz(153600, /*required=*/false);
    if (!ok) {
      std::printf("[planar] WARNING: set_sys_clock_khz(153600) failed to lock -- "
                  "falling back to stock clk_sys; PDM clock divider will NOT be "
                  "jitter-free (D2)\n");
    }
  }
  const uint32_t sysHz = clock_get_hz(clk_sys);
  std::printf("[planar] clk_sys=%lu Hz\n", static_cast<unsigned long>(sysHz));
}

void setupUplink() {
  gSpiLink.begin();
  std::printf("[planar] ESP32-C5 SPI link on spi0 sck=GP%u mosi=GP%u miso=GP%u cs=GP%u wake=GP%u -> %s\n",
              static_cast<unsigned>(nodecfg::kEspSpiSckPin),
              static_cast<unsigned>(nodecfg::kEspSpiTxPin),
              static_cast<unsigned>(nodecfg::kEspSpiRxPin),
              static_cast<unsigned>(nodecfg::kEspSpiCsPin),
              static_cast<unsigned>(nodecfg::kEspHostWakePin),
              nodecfg::kIngestPath);
}

void setupOptionalPeripherals() {
  const bool useI2c = nodecfg::kEnableImuTemperature ||
      nodecfg::kEnableSht45Environment ||
      nodecfg::kEnableCompassAutoOrientation;
  bool i2cReady = false;
  if (useI2c && nodecfg::kEnableExternalPeripheralBuses) {
    i2cReady = gI2c.begin(i2c1, nodecfg::kI2cSdaPin, nodecfg::kI2cSclPin, nodecfg::kI2cBaudHz);
    if (i2cReady) {
      gI2c.setReadCallback(onI2cRead);
    } else {
      std::printf("[planar] I2C init failed\n");
    }
  }
  if (nodecfg::kEnableGpsUart) {
    gGpsSource.begin();
    gGpsSource.poll(gNodeDescriptor, &gClock);
    std::printf("[planar] GPS UART on uart0 tx=GP%d rx=GP%d pps=GP%d\n",
                nodecfg::kGpsTxPin, nodecfg::kGpsRxPin, nodecfg::kGpsPpsPin);
  }
  std::printf("[planar] piezo buzzer pin reserved GP%u (chirp stub, unused)\n",
              static_cast<unsigned>(gBuzzer.pin()));
  if (nodecfg::kEnableSht45Environment) {
    std::printf("[planar] SHT4x environmental telemetry enabled\n");
  }
  if (nodecfg::kEnableImuTemperature) {
    std::printf("[planar] LSM6 IMU temperature telemetry enabled\n");
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

    // Planar's square array has no tetra-style 120° rotation steps; only the
    // heading estimate is used here (kProvisionedWorldHeadingDeg still drives
    // sensor-offset geometry until that integration lands).
    gAutoOrientationEnabled = gAutoOrientation.begin(gMagnetometer, cfg, /*initialRotation=*/0);
    if (gAutoOrientationEnabled) {
      std::printf("[planar] LIS2MDL auto-orientation enabled (Kalman Q=%.4f R=%.1f)\n",
                  static_cast<double>(cfg.kalmanQ),
                  static_cast<double>(cfg.kalmanR));
    } else {
      std::printf("[planar] magnetometer unavailable; heading unavailable\n");
    }
  }
}

}  // namespace

int main() {
  initNodeId();
  stdio_init_all();
  sleep_ms(300);
  mmpr::FailureSnapshot::initializeForBoot();
  gNodeDescriptor.bootCount = mmpr::FailureSnapshot::currentBootCount();
  mmpr::FailureSnapshot::feedWatchdog();

  setupStatusLed();
  setupActivityLed();
  setClockForPdm();

  std::printf("[planar] booting id=%s hardware=%s channels=%u radius=%.3fm\n",
              gNodeDescriptor.id,
              nodecfg::kHardwareName,
              static_cast<unsigned>(nodecfg::kActiveAudioChannels),
              static_cast<double>(nodecfg::kPlanarArrayRadiusM));

  {
    mmpr::NodeClockHoldoverConfig holdoverConfig;
    holdoverConfig.enableTempComp = nodecfg::kEnableHoldoverTempComp;
    holdoverConfig.driftUncertaintyPpm = nodecfg::kHoldoverDriftUncertaintyPpm;
    holdoverConfig.errorBudgetNs = nodecfg::kHoldoverErrorBudgetNs;
    holdoverConfig.maxAgeUs = nodecfg::kHoldoverMaxAgeUs;
    gClock.setHoldoverConfig(holdoverConfig);
  }

  mmpr::FailureSnapshot::updatePhase(mmpr::FatalLifecyclePhase::kPeripheralInit);
  setupUplink();
  setupOptionalPeripherals();
  mmpr::FailureSnapshot::feedWatchdog();

  mmpr::FailureSnapshot::updatePhase(mmpr::FatalLifecyclePhase::kRunnerStart);
  const bool started = gRunner.begin(/*syncNtp=*/false);
  mmpr::FailureSnapshot::feedWatchdog();
  if (!started) {
    std::printf("[planar] fatal: runner failed to start\n");
    while (true) { sleep_ms(1000); }
  }

  if (nodecfg::kEnableGpsUart) {
    gGpsSource.bindAudioSource(&gActiveAudioSource);
  }

  mmpr::FailureSnapshot::updatePhase(mmpr::FatalLifecyclePhase::kMainLoop);
  while (true) {
    if (gAutoOrientationEnabled) {
      const bool rotationStepChanged = gAutoOrientation.poll(nullptr);
      (void)rotationStepChanged;  // no rotation-step geometry on the square array
      if (!gAutoOrientation.healthy()) {
        gAutoOrientationEnabled = false;
        std::printf("[planar] magnetometer read fault; heading tracking disabled\n");
      }
    }

    if (nodecfg::kEnableGpsUart) {
      gGpsSource.poll(gNodeDescriptor, &gClock);
      gGpsRuntimeStats = gGpsSource.stats();
      gGpsRuntimeStats.clockQuality = gClock.timeQuality();
    }

    gRunner.loopOnce();
    mmpr::FailureSnapshot::updateProgressMarker(static_cast<uint32_t>(gRunner.stats().framesCaptured));
    pollActivityLed();

    static uint32_t ledCounter = 0;
    static bool ledState = false;
    if (++ledCounter >= nodecfg::kLedBlinkFrames) {
      ledCounter = 0;
      ledState = !ledState;
      setStatusLed(ledState);
    }
    mmpr::FailureSnapshot::feedWatchdog();
  }
}
