#pragma once

// Platform time/delay seam for the shared node runtime.
//
// The runtime (NodeClock, NodeRunner, the synthetic/silence audio sources) needs
// a handful of primitives from its host platform: a monotonic microsecond clock,
// a monotonic millisecond clock, two blocking delays, and a free-heap reading.
// Everything else in the portable subset is pure C++. Routing those through this
// header is what
// lets the identical NodeClock.cpp / NodeRunner.cpp / NodeProtocol.cpp build for
// both the RP2350 (pico-sdk) and the ESP32-S3 (ESP-IDF/FreeRTOS) without any
// fork of the protocol or runner code.
//
// Everything here is inline and resolves to the same underlying call the code
// used before the seam existed, so Pico codegen is unchanged.
//
// Adding a third platform means adding a branch here, not touching call sites.

#include <cstdint>

#if defined(ESP_PLATFORM)

#include "esp_heap_caps.h"
#include "esp_rom_sys.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#else

#include <malloc.h>

#include "pico/time.h"

// pico-sdk linker-script symbols bounding the newlib heap.
extern "C" char __end__;
extern "C" char __HeapLimit;

#endif

namespace mmpr {

#if defined(ESP_PLATFORM)

inline uint64_t nowMonotonicUs() {
  return static_cast<uint64_t>(esp_timer_get_time());
}

inline uint32_t nowMonotonicMs() {
  return static_cast<uint32_t>(esp_timer_get_time() / 1000);
}

// Yields to the scheduler rather than spinning. Requires CONFIG_FREERTOS_HZ=1000
// for 1 ms resolution (set in each ESP node's sdkconfig.defaults); at the IDF
// default of 100 Hz every sub-10 ms sleep rounds up to 10 ms and the runner's
// 1 ms idle nap becomes a 10 ms stall.
inline void sleepMs(uint32_t ms) {
  vTaskDelay(pdMS_TO_TICKS(ms));
}

// Sleeps whole milliseconds on the scheduler and busy-waits only the sub-ms
// remainder. Callers (SyntheticAudioSource/SilenceAudioSource frame pacing) pass
// values up to a frame period, so spinning the whole interval would burn a core.
inline void sleepUs(uint64_t us) {
  const uint32_t wholeMs = static_cast<uint32_t>(us / 1000ULL);
  if (wholeMs > 0) {
    vTaskDelay(pdMS_TO_TICKS(wholeMs));
  }
  const uint32_t remainderUs = static_cast<uint32_t>(us % 1000ULL);
  if (remainderUs > 0) {
    esp_rom_delay_us(remainderUs);
  }
}

inline uint32_t freeHeapBytes() {
  return static_cast<uint32_t>(esp_get_free_heap_size());
}

#else

inline uint64_t nowMonotonicUs() {
  return time_us_64();
}

inline uint32_t nowMonotonicMs() {
  return to_ms_since_boot(get_absolute_time());
}

inline void sleepMs(uint32_t ms) {
  ::sleep_ms(ms);
}

inline void sleepUs(uint64_t us) {
  ::sleep_us(us);
}

// Heap is [__end__, __HeapLimit) in the pico-sdk linker script; newlib's
// mallinfo().uordblks is the in-use total. mallinfo walks newlib's bins, so
// callers sample it on a slow cadence (NodeRunner: every 15 s).
inline uint32_t freeHeapBytes() {
  const struct mallinfo info = mallinfo();
  const uintptr_t heapBytes = reinterpret_cast<uintptr_t>(&__HeapLimit) - reinterpret_cast<uintptr_t>(&__end__);
  return heapBytes > static_cast<uintptr_t>(info.uordblks)
      ? static_cast<uint32_t>(heapBytes - static_cast<uintptr_t>(info.uordblks))
      : 0u;
}

#endif

}  // namespace mmpr
