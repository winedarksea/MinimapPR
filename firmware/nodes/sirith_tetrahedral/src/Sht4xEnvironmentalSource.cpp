#include "mmpr/Sht4xEnvironmentalSource.h"

#include <algorithm>
#include <cmath>

#include "pico/time.h"

namespace mmpr {
namespace {

constexpr uint8_t kMeasureHighPrecisionCommand[2] = {0xFD, 0x00};
constexpr uint32_t kMeasurementDelayMs = 10;

uint32_t millis32() {
  return to_ms_since_boot(get_absolute_time());
}

}  // namespace

Sht4xEnvironmentalSource::Sht4xEnvironmentalSource(
    I2cBus& bus,
    const Sht4xEnvironmentalSourceConfig& config)
    : bus_(&bus), config_(config) {}

bool Sht4xEnvironmentalSource::begin() {
  started_ = false;
  healthy_ = false;
  hasReading_ = false;

  if (bus_ == nullptr || !bus_->initialized()) {
    return false;
  }
  if (config_.address7Bit == 0) {
    return false;
  }
  if (config_.sampleIntervalMs == 0) {
    config_.sampleIntervalMs = 2000;
  }

  float temperatureC = 0.0f;
  float humidityFraction = 0.0f;
  if (!readMeasurement(temperatureC, humidityFraction)) {
    return false;
  }

  lastTemperatureC_ = temperatureC;
  lastHumidityFraction_ = humidityFraction;
  lastSampleMs_ = millis32();
  hasReading_ = true;
  healthy_ = true;
  started_ = true;
  return true;
}

bool Sht4xEnvironmentalSource::read(EnvironmentalSample& outSample) {
  if (!started_) {
    return false;
  }

  const uint32_t nowMs = millis32();
  if (!hasReading_ || (nowMs - lastSampleMs_) >= config_.sampleIntervalMs) {
    float temperatureC = 0.0f;
    float humidityFraction = 0.0f;
    if (readMeasurement(temperatureC, humidityFraction)) {
      lastTemperatureC_ = temperatureC;
      lastHumidityFraction_ = humidityFraction;
      lastSampleMs_ = nowMs;
      hasReading_ = true;
      healthy_ = true;
    } else {
      healthy_ = false;
      if (!hasReading_) {
        return false;
      }
    }
  }

  outSample = EnvironmentalSample();
  outSample.hasTemperatureC = true;
  outSample.temperatureC = lastTemperatureC_;
  outSample.hasHumidityFraction = true;
  outSample.humidityFraction = lastHumidityFraction_;
  outSample.temperatureSource = "sht45";
  return true;
}

bool Sht4xEnvironmentalSource::readMeasurement(float& outTemperatureC, float& outHumidityFraction) {
  if (bus_ == nullptr) {
    return false;
  }
  if (!bus_->write(config_.address7Bit, kMeasureHighPrecisionCommand, sizeof(kMeasureHighPrecisionCommand), false)) {
    return false;
  }

  sleep_ms(kMeasurementDelayMs);

  uint8_t raw[6] = {0};
  if (!bus_->read(config_.address7Bit, raw, sizeof(raw), false)) {
    return false;
  }
  if (!verifyCrc(raw, raw[2]) || !verifyCrc(raw + 3, raw[5])) {
    return false;
  }

  const uint16_t rawTemperature = static_cast<uint16_t>((static_cast<uint16_t>(raw[0]) << 8u) | raw[1]);
  const uint16_t rawHumidity = static_cast<uint16_t>((static_cast<uint16_t>(raw[3]) << 8u) | raw[4]);

  const float temperatureC = -45.0f + (175.0f * static_cast<float>(rawTemperature) / 65535.0f);
  const float humidityPercent = -6.0f + (125.0f * static_cast<float>(rawHumidity) / 65535.0f);
  const float humidityFraction = std::clamp(humidityPercent / 100.0f, 0.0f, 1.0f);

  if (!std::isfinite(temperatureC) || !std::isfinite(humidityFraction)) {
    return false;
  }

  outTemperatureC = temperatureC;
  outHumidityFraction = humidityFraction;
  return true;
}

bool Sht4xEnvironmentalSource::verifyCrc(const uint8_t* data, uint8_t expectedCrc) {
  if (data == nullptr) {
    return false;
  }

  uint8_t crc = 0xFFu;
  for (uint8_t index = 0; index < 2; ++index) {
    crc ^= data[index];
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x80u) ? static_cast<uint8_t>((crc << 1u) ^ 0x31u) : static_cast<uint8_t>(crc << 1u);
    }
  }
  return crc == expectedCrc;
}

}  // namespace mmpr
