#pragma once

// Internal TDM loopback self-test.
//
// Validates framing and channel ordering on a bare Feather, before an ADAU7112
// exists: a TX channel on the same I2S port (full duplex, so it shares the RX
// channel's BCLK/FSYNC) emits a distinct constant per slot -- 0x111111 in slot
// 0, 0x222222 in slot 1, and so on -- and the test asserts that every slot
// arrives decoded to the right value in the right output-channel position.
//
// Wiring: jumper the loopback DOUT pin to the SDATA DIN pin
// (nodecfg::kTdmLoopbackDoutPin -> nodecfg::kTdmDataPin, D10 -> D9 by default)
// and build with MMPR_NODECFG_ENABLE_I2S_LOOPBACK=1.
//
// WHAT IT DOES NOT PROVE: TX and RX use the same slot config, so the test is
// self-consistent by construction with respect to the ADAU7112's actual
// waveform. It catches a wrong outputChannelToSlot map, a wrong slot count, a
// wrong validBits/captureBitOffset decode, and a dead data path. It cannot
// catch a physical mic <-> slot strapping error, or a ws_width/ws_pol that
// disagrees with the real part. Those need the mics attached and a bearing
// sanity check through the pipeline.

#include <cstddef>
#include <cstdint>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "mmpr/EspI2sTdmSource.h"

namespace mmpr {

struct EspI2sTdmLoopbackResult {
  bool passed = false;
  uint32_t framesChecked = 0;
  int16_t expected[4] = {};
  int16_t observed[4] = {};
  // Index of the first differing sample within the failing frame, or SIZE_MAX.
  size_t mismatchSampleIndex = 0;
  uint8_t mismatchChannel = 0;
};

class EspI2sTdmLoopback {
 public:
  explicit EspI2sTdmLoopback(EspI2sTdmSource& source);
  ~EspI2sTdmLoopback();

  EspI2sTdmLoopback(const EspI2sTdmLoopback&) = delete;
  EspI2sTdmLoopback& operator=(const EspI2sTdmLoopback&) = delete;

  // Starts the pattern feeder. The source must already have begun with a
  // loopbackDout pin assigned.
  bool begin();

  // Discards `warmupFrames` (the first frames after enable contain the DMA
  // buffers' initial zeros and a partial first frame), then checks
  // `verifyFrames` complete frames sample-for-sample. Blocking; call from the
  // node main before starting NodeRunner.
  bool runSelfTest(
      EspI2sTdmLoopbackResult& result,
      uint32_t warmupFrames = 8,
      uint32_t verifyFrames = 4);

  // The PCM16 value slot `slot`'s planted constant must decode to, given the
  // source's slotBits / validBits / captureBitOffset. Derived rather than
  // hardcoded so a decode-config change cannot make the test vacuously pass.
  static int16_t expectedPcmForSlot(const EspI2sTdmConfig& config, uint8_t slot);
  static uint32_t patternWordForSlot(const EspI2sTdmConfig& config, uint8_t slot);

 private:
  static void feederTaskEntry(void* context);
  void feederLoop();

  EspI2sTdmSource& source_;
  TaskHandle_t feederTask_ = nullptr;
  uint32_t* patternBuffer_ = nullptr;
  size_t patternBufferBytes_ = 0;
};

}  // namespace mmpr
