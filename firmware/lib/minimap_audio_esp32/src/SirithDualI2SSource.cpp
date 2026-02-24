#include "mmpr/SirithDualI2SSource.h"

#include <Arduino.h>
#include <esp_err.h>

namespace mmpr {

SirithDualI2SSource::SirithDualI2SSource(const SirithDualI2SPins& pins, const SirithDualI2SConfig& config)
    : pins_(pins), config_(config) {}

SirithDualI2SSource::~SirithDualI2SSource() {
  cleanup();
}

bool SirithDualI2SSource::initPort(i2s_port_t port, int bclk, int ws, int dataIn) {
  i2s_config_t cfg = {};
  cfg.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = static_cast<int>(config_.sampleRateHz);
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = config_.dmaBufferCount;
  cfg.dma_buf_len = config_.dmaBufferLength;
  cfg.use_apll = config_.useApll;
  cfg.tx_desc_auto_clear = false;
  cfg.fixed_mclk = 0;

  const esp_err_t installErr = i2s_driver_install(port, &cfg, 0, nullptr);
  if (installErr != ESP_OK) {
    Serial.printf("[sirith] i2s_driver_install failed port=%d err=%d\n", static_cast<int>(port), installErr);
    return false;
  }
  if (port == I2S_NUM_0) {
    port0Installed_ = true;
  } else if (port == I2S_NUM_1) {
    port1Installed_ = true;
  }

  i2s_pin_config_t pinCfg = {};
  pinCfg.bck_io_num = bclk;
  pinCfg.ws_io_num = ws;
  pinCfg.data_out_num = I2S_PIN_NO_CHANGE;
  pinCfg.data_in_num = dataIn;

  const esp_err_t pinErr = i2s_set_pin(port, &pinCfg);
  if (pinErr != ESP_OK) {
    Serial.printf("[sirith] i2s_set_pin failed port=%d err=%d\n", static_cast<int>(port), pinErr);
    stopAndUninstall(port);
    return false;
  }

  const esp_err_t clkErr = i2s_set_clk(
      port,
      static_cast<uint32_t>(config_.sampleRateHz),
      I2S_BITS_PER_SAMPLE_32BIT,
      I2S_CHANNEL_STEREO);
  if (clkErr != ESP_OK) {
    Serial.printf("[sirith] i2s_set_clk failed port=%d err=%d\n", static_cast<int>(port), clkErr);
    stopAndUninstall(port);
    return false;
  }

  i2s_zero_dma_buffer(port);
  i2s_start(port);
  return true;
}

void SirithDualI2SSource::stopAndUninstall(i2s_port_t port) {
  const bool installed = (port == I2S_NUM_0) ? port0Installed_ : port1Installed_;
  if (!installed) {
    return;
  }

  i2s_stop(port);
  i2s_driver_uninstall(port);
  if (port == I2S_NUM_0) {
    port0Installed_ = false;
  } else if (port == I2S_NUM_1) {
    port1Installed_ = false;
  }
}

void SirithDualI2SSource::cleanup() {
  if (raw0_ != nullptr) {
    delete[] raw0_;
    raw0_ = nullptr;
  }
  if (raw1_ != nullptr) {
    delete[] raw1_;
    raw1_ = nullptr;
  }

  stopAndUninstall(I2S_NUM_0);
  stopAndUninstall(I2S_NUM_1);
}

bool SirithDualI2SSource::begin() {
  cleanup();

  if (config_.sampleRateHz == 0 || config_.frameSamples == 0 || config_.dmaBufferCount == 0 || config_.dmaBufferLength == 0) {
    Serial.println("[sirith] invalid dual i2s config");
    return false;
  }

  if (!initPort(I2S_NUM_0, pins_.bclk0, pins_.ws0, pins_.dataIn0)) {
    return false;
  }
  if (!initPort(I2S_NUM_1, pins_.bclk1, pins_.ws1, pins_.dataIn1)) {
    stopAndUninstall(I2S_NUM_0);
    return false;
  }

  raw0_ = new int32_t[config_.frameSamples * 2];
  raw1_ = new int32_t[config_.frameSamples * 2];
  if (raw0_ == nullptr || raw1_ == nullptr) {
    Serial.println("[sirith] raw buffer allocation failed");
    cleanup();
    return false;
  }
  return true;
}

int16_t SirithDualI2SSource::toPcm16(int32_t raw) const {
  int32_t shifted = raw >> config_.sampleShiftBits;
  if (shifted > 32767) {
    shifted = 32767;
  } else if (shifted < -32768) {
    shifted = -32768;
  }
  return static_cast<int16_t>(shifted);
}

bool SirithDualI2SSource::readFrame(int16_t* interleavedOut, size_t samplesPerChannel) {
  if (interleavedOut == nullptr || raw0_ == nullptr || raw1_ == nullptr || samplesPerChannel != config_.frameSamples) {
    return false;
  }

  const size_t bytesTarget = config_.frameSamples * 2 * sizeof(int32_t);
  size_t bytesRead0 = 0;
  size_t bytesRead1 = 0;

  const esp_err_t readErr0 = i2s_read(I2S_NUM_0, raw0_, bytesTarget, &bytesRead0, portMAX_DELAY);
  const esp_err_t readErr1 = i2s_read(I2S_NUM_1, raw1_, bytesTarget, &bytesRead1, portMAX_DELAY);

  if (readErr0 != ESP_OK || readErr1 != ESP_OK || bytesRead0 < bytesTarget || bytesRead1 < bytesTarget) {
    return false;
  }

  for (size_t i = 0; i < config_.frameSamples; ++i) {
    const int32_t a0 = raw0_[2 * i + 0];
    const int32_t a1 = raw0_[2 * i + 1];
    const int32_t b0 = raw1_[2 * i + 0];
    const int32_t b1 = raw1_[2 * i + 1];

    interleavedOut[(i * 4) + 0] = toPcm16(a0);
    interleavedOut[(i * 4) + 1] = toPcm16(a1);
    interleavedOut[(i * 4) + 2] = toPcm16(b0);
    interleavedOut[(i * 4) + 3] = toPcm16(b1);
  }

  return true;
}

}  // namespace mmpr
