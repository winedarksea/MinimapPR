#include "mmpr/Esp32I2SMonoSource.h"

#include <Arduino.h>
#include <esp_err.h>

namespace mmpr {

Esp32I2SMonoSource::Esp32I2SMonoSource(const Esp32I2SPins& pins, const Esp32I2SMonoConfig& config)
    : pins_(pins), config_(config) {}

bool Esp32I2SMonoSource::begin() {
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

  const esp_err_t installErr = i2s_driver_install(config_.port, &cfg, 0, nullptr);
  if (installErr != ESP_OK) {
    Serial.printf("[point] i2s_driver_install failed err=%d\n", installErr);
    return false;
  }

  i2s_pin_config_t pinCfg = {};
  pinCfg.bck_io_num = pins_.bclk;
  pinCfg.ws_io_num = pins_.ws;
  pinCfg.data_out_num = I2S_PIN_NO_CHANGE;
  pinCfg.data_in_num = pins_.dataIn;

  const esp_err_t pinErr = i2s_set_pin(config_.port, &pinCfg);
  if (pinErr != ESP_OK) {
    Serial.printf("[point] i2s_set_pin failed err=%d\n", pinErr);
    return false;
  }

  const esp_err_t clkErr = i2s_set_clk(
      config_.port,
      static_cast<uint32_t>(config_.sampleRateHz),
      I2S_BITS_PER_SAMPLE_32BIT,
      I2S_CHANNEL_STEREO);
  if (clkErr != ESP_OK) {
    Serial.printf("[point] i2s_set_clk failed err=%d\n", clkErr);
    return false;
  }

  rawStereo_ = new int32_t[config_.frameSamples * 2];
  if (rawStereo_ == nullptr) {
    Serial.println("[point] raw buffer allocation failed");
    return false;
  }

  i2s_zero_dma_buffer(config_.port);
  i2s_start(config_.port);
  return true;
}

int16_t Esp32I2SMonoSource::toPcm16(int32_t raw) const {
  int32_t shifted = raw >> config_.sampleShiftBits;
  if (shifted > 32767) {
    shifted = 32767;
  } else if (shifted < -32768) {
    shifted = -32768;
  }
  return static_cast<int16_t>(shifted);
}

bool Esp32I2SMonoSource::readFrame(int16_t* interleavedOut, size_t samplesPerChannel) {
  if (interleavedOut == nullptr || rawStereo_ == nullptr || samplesPerChannel != config_.frameSamples) {
    return false;
  }

  const size_t bytesTarget = config_.frameSamples * 2 * sizeof(int32_t);
  size_t bytesRead = 0;
  const esp_err_t readErr = i2s_read(config_.port, rawStereo_, bytesTarget, &bytesRead, portMAX_DELAY);
  if (readErr != ESP_OK || bytesRead < bytesTarget) {
    return false;
  }

  const size_t channelIndex = (config_.stereoChannelIndex > 0) ? 1 : 0;
  for (size_t i = 0; i < config_.frameSamples; ++i) {
    interleavedOut[i] = toPcm16(rawStereo_[2 * i + channelIndex]);
  }

  return true;
}

}  // namespace mmpr
