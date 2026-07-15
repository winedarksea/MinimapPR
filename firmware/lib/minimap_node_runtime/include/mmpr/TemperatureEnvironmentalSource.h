#pragma once

#include "mmpr/IEnvironmentalSource.h"
#include "mmpr/ITemperatureSensor.h"

namespace mmpr {

class TemperatureEnvironmentalSource : public IEnvironmentalSource {
 public:
  explicit TemperatureEnvironmentalSource(ITemperatureSensor& sensor) : sensor_(sensor) {}

  bool begin() override;
  bool read(EnvironmentalSample& outSample) override;

 private:
  ITemperatureSensor& sensor_;
  bool ready_ = false;
};

}  // namespace mmpr
