#pragma once

#include <cstdint>

namespace mmpr {

/// Piezo buzzer driver (Phase 6 stub).
///
/// Reserves the PWM pin for a piezo driver intended for ultrasonic
/// self-localization. That capability is not built out yet, so chirp()
/// only validates its parameters — it does not configure PWM hardware or
/// emit any sound.
class PiezoBuzzer {
 public:
  explicit PiezoBuzzer(uint8_t pwmPin);

  /// Sweep a chirp from fStartHz to fEndHz over durationMs.
  /// STUB: PWM chirp generation is not yet implemented.
  bool chirp(float fStartHz, float fEndHz, uint32_t durationMs);

  uint8_t pin() const { return pin_; }

 private:
  uint8_t pin_ = 0;
};

}  // namespace mmpr
