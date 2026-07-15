#pragma once

#include "mmpr/IEnvironmentalSource.h"

namespace mmpr {

class FallbackEnvironmentalSource : public IEnvironmentalSource {
 public:
  FallbackEnvironmentalSource(
      IEnvironmentalSource* primarySource,
      IEnvironmentalSource* fallbackSource);

  bool begin() override;
  bool read(EnvironmentalSample& outSample) override;

 private:
  IEnvironmentalSource* primarySource_ = nullptr;
  IEnvironmentalSource* fallbackSource_ = nullptr;
  IEnvironmentalSource* activeSource_ = nullptr;
};

}  // namespace mmpr
