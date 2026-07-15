#include "mmpr/PiezoBuzzer.h"

#include <cmath>

namespace mmpr {

PiezoBuzzer::PiezoBuzzer(uint8_t pwmPin) : pin_(pwmPin) {}

bool PiezoBuzzer::chirp(float fStartHz, float fEndHz, uint32_t durationMs) {
  if (!std::isfinite(fStartHz) || !std::isfinite(fEndHz) ||
      fStartHz <= 0.0f || fEndHz <= 0.0f || durationMs == 0) {
    return false;
  }
  // STUB: PWM chirp sweep generation is not yet implemented.
  return false;
}

}  // namespace mmpr
