#include "mmpr/FallbackEnvironmentalSource.h"

namespace mmpr {

FallbackEnvironmentalSource::FallbackEnvironmentalSource(
    IEnvironmentalSource* primarySource,
    IEnvironmentalSource* fallbackSource)
    : primarySource_(primarySource), fallbackSource_(fallbackSource) {}

bool FallbackEnvironmentalSource::begin() {
  activeSource_ = nullptr;

  if (primarySource_ != nullptr && primarySource_->begin()) {
    activeSource_ = primarySource_;
    return true;
  }

  if (fallbackSource_ != nullptr && fallbackSource_->begin()) {
    activeSource_ = fallbackSource_;
    return true;
  }

  return false;
}

bool FallbackEnvironmentalSource::read(EnvironmentalSample& outSample) {
  if (activeSource_ == nullptr) {
    return false;
  }

  if (activeSource_->read(outSample)) {
    return true;
  }

  if (activeSource_ == primarySource_ && fallbackSource_ != nullptr && fallbackSource_->begin()) {
    activeSource_ = fallbackSource_;
    return activeSource_->read(outSample);
  }

  return false;
}

}  // namespace mmpr
