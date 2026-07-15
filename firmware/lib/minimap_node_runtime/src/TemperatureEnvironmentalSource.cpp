#include "mmpr/TemperatureEnvironmentalSource.h"

namespace mmpr {

bool TemperatureEnvironmentalSource::begin() {
  ready_ = sensor_.begin();
  return ready_;
}

bool TemperatureEnvironmentalSource::read(EnvironmentalSample& outSample) {
  if (!ready_) {
    return false;
  }

  float temperatureC = 0.0f;
  const char* source = nullptr;
  if (!sensor_.readTemperatureC(temperatureC, source)) {
    return false;
  }

  outSample = EnvironmentalSample();
  outSample.hasTemperatureC = true;
  outSample.temperatureC = temperatureC;
  outSample.temperatureSource = source;
  return true;
}

}  // namespace mmpr
