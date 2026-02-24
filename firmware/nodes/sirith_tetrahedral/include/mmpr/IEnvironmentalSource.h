#pragma once

#include "mmpr/Types.h"

namespace mmpr {

class IEnvironmentalSource {
 public:
  virtual ~IEnvironmentalSource() = default;

  virtual bool begin() = 0;
  virtual bool read(EnvironmentalSample& outSample) = 0;
};

}  // namespace mmpr
