#pragma once

#include "mmpr/Types.h"

namespace mmpr {

class IEnvironmentalSource {
 public:
  virtual ~IEnvironmentalSource() = default;

  // Initializes optional environmental hardware.
  // Returning false must not be treated as fatal to audio streaming.
  virtual bool begin() = 0;

  // Non-blocking poll/read. Returns true when at least one field is valid.
  virtual bool read(EnvironmentalSample& outSample) = 0;
};

}  // namespace mmpr
