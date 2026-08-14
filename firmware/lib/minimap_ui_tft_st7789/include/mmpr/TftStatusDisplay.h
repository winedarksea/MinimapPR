#pragma once

// ST7789 status page for the Adafruit ESP32-S3 TFT Feather (240x135).
//
// Ten lines of 8x8 text at 2 Hz, rendered from a low-priority task: node id and
// uptime, Wi-Fi/IP/RSSI, clock quality, audio ring occupancy, publish latency,
// capture/drop/overflow counters, and HTTP status. It exists because the whole
// point of this node is a number, and reading that number off a display beats
// tailing a serial console while walking a board around.
//
// esp_lcd_panel_st7789.h ships inside IDF's esp_lcd component, so this needs no
// component-manager download and no LVGL: an embedded 8x8 font and a 240x8
// single-row RGB565 strip buffer that is reused for each line.
//
// BOARD GOTCHA: TFT_I2C_POWER (GPIO 21) must be driven HIGH or the panel is
// simply dead -- it gates the display's rail on this board. TFT_BACKLITE
// (GPIO 45) must be high too or the panel is on but black.

#include <cstdint>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace mmpr {

struct RunnerStats;

struct TftStatusPins {
  int sck = 36;
  int mosi = 35;
  int cs = 7;
  int dc = 39;
  int reset = 40;
  int backlight = 45;
  // Rail-enable for the TFT + STEMMA connector. Must be driven high.
  int powerEnable = 21;
};

struct TftStatusConfig {
  uint16_t width = 240;
  uint16_t height = 135;
  // The 240x135 panel is a window into a 240x320 ST7789 controller; in the
  // landscape orientation this driver selects, the visible area starts at these
  // offsets. Wrong gaps show as a shifted or wrapped image.
  int xGap = 40;
  int yGap = 53;
  bool mirrorX = false;
  bool mirrorY = true;
  bool swapXy = true;
  bool invertColor = true;  // ST7789 panels on this board are IPS (inverted)
  uint32_t pixelClockHz = 40 * 1000 * 1000;
  int spiHost = 1;  // SPI2_HOST
  uint32_t refreshIntervalMs = 500;
  int taskCoreId = 0;
  uint32_t taskPriority = 1;
  uint32_t taskStackBytes = 4096;
};

// Everything the status page shows that does not come from RunnerStats. The node
// main owns a single instance and updates it in place; the render task reads it
// without locking, since every field is a word-sized scalar or a fixed buffer
// whose worst-case tear is one stale character.
struct TftStatusModel {
  const char* nodeId = "";
  const char* ssid = "";
  const char* ipAddress = "0.0.0.0";
  const char* clockQuality = "free_running";
  const char* audioMode = "synthetic";
  uint32_t sampleRateHz = 0;
  uint8_t channels = 0;
  double estimatedPpm = 0.0;
  uint32_t ringFramesCapacity = 0;
  uint32_t ringFramesAvailable = 0;
  uint32_t driverOverflowCount = 0;
  uint32_t ntpSyncCount = 0;
};

class TftStatusDisplay {
 public:
  TftStatusDisplay(
      const TftStatusPins& pins,
      const TftStatusConfig& config,
      const RunnerStats& runnerStats,
      const TftStatusModel& model);
  ~TftStatusDisplay();

  TftStatusDisplay(const TftStatusDisplay&) = delete;
  TftStatusDisplay& operator=(const TftStatusDisplay&) = delete;

  // Powers the panel, initialises SPI + ST7789, and starts the render task.
  bool begin();

 private:
  static void taskEntry(void* context);
  void renderLoop();
  void renderFrame();
  void drawTextRow(int row, const char* text);

  TftStatusPins pins_;
  TftStatusConfig config_;
  const RunnerStats& runnerStats_;
  const TftStatusModel& model_;

  void* panel_ = nullptr;      // esp_lcd_panel_handle_t
  void* panelIo_ = nullptr;    // esp_lcd_panel_io_handle_t
  uint16_t* rowBuffer_ = nullptr;
  TaskHandle_t taskHandle_ = nullptr;
};

}  // namespace mmpr
