#include "mmpr/TftStatusDisplay.h"

#include <cstdio>
#include <cstring>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_heap_caps.h"
#include "esp_lcd_io_spi.h"
#include "esp_lcd_panel_dev.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_st7789.h"
#include "esp_log.h"

#include "mmpr/Font8x8.h"
#include "mmpr/NodeRunner.h"
#include "mmpr/PlatformTime.h"

namespace mmpr {
namespace {

constexpr const char* kTag = "mmpr-tft";

constexpr int kGlyphHeight = 8;
constexpr int kGlyphWidth = 8;
constexpr int kRowPitch = 16;
constexpr int kTopMargin = 4;
constexpr int kStatusRows = 8;
constexpr size_t kMaxColumns = 30;  // 240 px / 8 px per glyph

// The panel is configured for MSB-first RGB565 (LCD_RGB_DATA_ENDIAN_BIG) while
// the S3 is little-endian, so colour literals are byte-swapped once here rather
// than at every pixel write.
constexpr uint16_t panelColor(uint16_t rgb565) {
  return static_cast<uint16_t>((rgb565 >> 8) | (rgb565 << 8));
}

constexpr uint16_t kColorBackground = panelColor(0x0000);  // black
constexpr uint16_t kColorText = panelColor(0xFFFF);        // white
constexpr uint16_t kColorHeading = panelColor(0x07FF);     // cyan
constexpr uint16_t kColorAlarm = panelColor(0xF800);       // red

}  // namespace

TftStatusDisplay::TftStatusDisplay(
    const TftStatusPins& pins,
    const TftStatusConfig& config,
    const RunnerStats& runnerStats,
    const TftStatusModel& model)
    : pins_(pins), config_(config), runnerStats_(runnerStats), model_(model) {}

TftStatusDisplay::~TftStatusDisplay() {
  if (taskHandle_ != nullptr) {
    vTaskDelete(taskHandle_);
    taskHandle_ = nullptr;
  }
  if (panel_ != nullptr) {
    esp_lcd_panel_del(static_cast<esp_lcd_panel_handle_t>(panel_));
    panel_ = nullptr;
  }
  if (panelIo_ != nullptr) {
    esp_lcd_panel_io_del(static_cast<esp_lcd_panel_io_handle_t>(panelIo_));
    panelIo_ = nullptr;
  }
  heap_caps_free(rowBuffer_);
}

bool TftStatusDisplay::begin() {
  // TFT_I2C_POWER gates the display rail on this board. Without it the SPI
  // transactions all "succeed" and the screen stays dark, which is a genuinely
  // confusing failure to debug.
  if (pins_.powerEnable >= 0) {
    gpio_config_t powerConfig = {};
    powerConfig.pin_bit_mask = 1ULL << pins_.powerEnable;
    powerConfig.mode = GPIO_MODE_OUTPUT;
    gpio_config(&powerConfig);
    gpio_set_level(static_cast<gpio_num_t>(pins_.powerEnable), 1);
    sleepMs(10);
  }

  spi_bus_config_t busConfig = {};
  busConfig.sclk_io_num = pins_.sck;
  busConfig.mosi_io_num = pins_.mosi;
  busConfig.miso_io_num = -1;
  busConfig.quadwp_io_num = -1;
  busConfig.quadhd_io_num = -1;
  busConfig.max_transfer_sz = config_.width * kGlyphHeight * static_cast<int>(sizeof(uint16_t));
  esp_err_t err = spi_bus_initialize(
      static_cast<spi_host_device_t>(config_.spiHost), &busConfig, SPI_DMA_CH_AUTO);
  if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
    ESP_LOGE(kTag, "spi_bus_initialize failed: %s", esp_err_to_name(err));
    return false;
  }

  esp_lcd_panel_io_spi_config_t ioConfig = {};
  ioConfig.cs_gpio_num = static_cast<gpio_num_t>(pins_.cs);
  ioConfig.dc_gpio_num = static_cast<gpio_num_t>(pins_.dc);
  ioConfig.spi_mode = 0;
  ioConfig.pclk_hz = config_.pixelClockHz;
  ioConfig.trans_queue_depth = 4;
  ioConfig.lcd_cmd_bits = 8;
  ioConfig.lcd_param_bits = 8;
  esp_lcd_panel_io_handle_t io = nullptr;
  err = esp_lcd_new_panel_io_spi(
      reinterpret_cast<esp_lcd_spi_bus_handle_t>(config_.spiHost), &ioConfig, &io);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "panel io failed: %s", esp_err_to_name(err));
    return false;
  }
  panelIo_ = io;

  esp_lcd_panel_dev_config_t panelConfig = {};
  panelConfig.reset_gpio_num = static_cast<gpio_num_t>(pins_.reset);
  panelConfig.rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB;
  panelConfig.data_endian = LCD_RGB_DATA_ENDIAN_BIG;
  panelConfig.bits_per_pixel = 16;
  esp_lcd_panel_handle_t panel = nullptr;
  err = esp_lcd_new_panel_st7789(io, &panelConfig, &panel);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "st7789 create failed: %s", esp_err_to_name(err));
    return false;
  }
  panel_ = panel;

  esp_lcd_panel_reset(panel);
  esp_lcd_panel_init(panel);
  esp_lcd_panel_invert_color(panel, config_.invertColor);
  esp_lcd_panel_swap_xy(panel, config_.swapXy);
  esp_lcd_panel_mirror(panel, config_.mirrorX, config_.mirrorY);
  esp_lcd_panel_set_gap(panel, config_.xGap, config_.yGap);
  esp_lcd_panel_disp_on_off(panel, true);

  rowBuffer_ = static_cast<uint16_t*>(heap_caps_malloc(
      static_cast<size_t>(config_.width) * kGlyphHeight * sizeof(uint16_t),
      MALLOC_CAP_DMA | MALLOC_CAP_8BIT));
  if (rowBuffer_ == nullptr) {
    ESP_LOGE(kTag, "row buffer allocation failed");
    return false;
  }

  // Clear once. Every later write is a full-width padded text row, so the only
  // pixels never rewritten are the inter-row gaps -- which this clear owns.
  for (size_t i = 0; i < static_cast<size_t>(config_.width) * kGlyphHeight; ++i) {
    rowBuffer_[i] = kColorBackground;
  }
  for (int y = 0; y < config_.height; y += kGlyphHeight) {
    const int yEnd = (y + kGlyphHeight) > config_.height ? config_.height : (y + kGlyphHeight);
    esp_lcd_panel_draw_bitmap(panel, 0, y, config_.width, yEnd, rowBuffer_);
  }

  if (pins_.backlight >= 0) {
    gpio_config_t backlightConfig = {};
    backlightConfig.pin_bit_mask = 1ULL << pins_.backlight;
    backlightConfig.mode = GPIO_MODE_OUTPUT;
    gpio_config(&backlightConfig);
    gpio_set_level(static_cast<gpio_num_t>(pins_.backlight), 1);
  }

  const BaseType_t created = xTaskCreatePinnedToCore(
      &TftStatusDisplay::taskEntry,
      "mmpr_tft",
      config_.taskStackBytes,
      this,
      config_.taskPriority,
      &taskHandle_,
      config_.taskCoreId);
  if (created != pdPASS) {
    ESP_LOGE(kTag, "render task create failed");
    return false;
  }
  ESP_LOGI(kTag, "status page up (%ux%u)",
           static_cast<unsigned>(config_.width), static_cast<unsigned>(config_.height));
  return true;
}

void TftStatusDisplay::taskEntry(void* context) {
  static_cast<TftStatusDisplay*>(context)->renderLoop();
}

void TftStatusDisplay::renderLoop() {
  while (true) {
    renderFrame();
    sleepMs(config_.refreshIntervalMs);
  }
}

void TftStatusDisplay::drawTextRow(int row, const char* text) {
  const int y = kTopMargin + (row * kRowPitch);
  if (y + kGlyphHeight > config_.height) {
    return;
  }
  const size_t width = config_.width;
  for (size_t i = 0; i < width * kGlyphHeight; ++i) {
    rowBuffer_[i] = kColorBackground;
  }

  // Row 0 is the identity line and rows carrying a non-zero fault counter are
  // coloured, so a bad node is legible from across the bench.
  uint16_t color = kColorText;
  if (row == 0) {
    color = kColorHeading;
  } else if (std::strstr(text, "!") != nullptr) {
    color = kColorAlarm;
  }

  const size_t columns = width / kGlyphWidth;
  for (size_t column = 0; column < columns && text[column] != '\0'; ++column) {
    const uint8_t* glyph = font8x8Glyph(text[column]);
    for (int glyphRow = 0; glyphRow < kGlyphHeight; ++glyphRow) {
      const uint8_t bits = glyph[glyphRow];
      uint16_t* destination = rowBuffer_ + (static_cast<size_t>(glyphRow) * width) +
                              (column * kGlyphWidth);
      for (int bit = 0; bit < kGlyphWidth; ++bit) {
        if ((bits >> bit) & 1u) {
          destination[bit] = color;
        }
      }
    }
  }

  esp_lcd_panel_draw_bitmap(
      static_cast<esp_lcd_panel_handle_t>(panel_),
      0,
      y,
      static_cast<int>(width),
      y + kGlyphHeight,
      rowBuffer_);
}

void TftStatusDisplay::renderFrame() {
  const RunnerStats& stats = runnerStats_;
  const uint32_t uptimeSeconds = nowMonotonicMs() / 1000u;
  char line[kMaxColumns + 1] = {};

  std::snprintf(line, sizeof(line), "%s %02u:%02u:%02u",
                model_.nodeId,
                static_cast<unsigned>(uptimeSeconds / 3600u),
                static_cast<unsigned>((uptimeSeconds / 60u) % 60u),
                static_cast<unsigned>(uptimeSeconds % 60u));
  drawTextRow(0, line);

  // Free heap rides along with the radio line. Both answer "is the node
  // healthy", and folding it here is what pays for dropping the audio-format
  // row, whose values never change after boot.
  std::snprintf(line, sizeof(line), "wifi %-8s %4ddB %luk",
                model_.ssid,
                static_cast<int>(stats.wifiRssiDbm),
                static_cast<unsigned long>(stats.heapFreeBytes / 1024u));
  drawTextRow(1, line);

  std::snprintf(line, sizeof(line), "ip   %s", model_.ipAddress);
  drawTextRow(2, line);

  // Rendered from integer tenths rather than "%.1f": newlib's nano formatting
  // (a per-target IDF default) silently drops float conversions, and a status
  // page that prints "f" instead of a drift figure is worse than useless.
  const int32_t ppmTenths = static_cast<int32_t>(model_.estimatedPpm * 10.0);
  const int32_t ppmAbsTenths = ppmTenths < 0 ? -ppmTenths : ppmTenths;
  std::snprintf(line, sizeof(line), "clk %-6s %c%ld.%luppm n%u",
                model_.clockQuality,
                ppmTenths < 0 ? '-' : '+',
                static_cast<long>(ppmAbsTenths / 10),
                static_cast<unsigned long>(ppmAbsTenths % 10),
                static_cast<unsigned>(model_.ntpSyncCount));
  drawTextRow(3, line);

  std::snprintf(line, sizeof(line), "ring %2u/%-2u  q %2u/%-2u  hw%2u",
                static_cast<unsigned>(model_.ringFramesAvailable),
                static_cast<unsigned>(model_.ringFramesCapacity),
                static_cast<unsigned>(stats.queueDepth),
                static_cast<unsigned>(stats.queueSlotsCapacity),
                static_cast<unsigned>(stats.ringFramesHighWater));
  drawTextRow(4, line);

  std::snprintf(line, sizeof(line), "pub %5u lat%4u ewma%4u",
                static_cast<unsigned>(stats.framesPublished % 100000u),
                static_cast<unsigned>(stats.publishLatencyLastMs),
                static_cast<unsigned>(stats.publishLatencyEwmaMs));
  drawTextRow(5, line);

  // The counters the whole exercise is about. A '!' anywhere on the line turns
  // it red (see drawTextRow), so "no loss" is a colour, not a reading exercise.
  const bool anyLoss = stats.framesDropped > 0 || stats.queueOverflows > 0 ||
                       stats.packetContinuityViolations > 0 || model_.driverOverflowCount > 0;
  std::snprintf(line, sizeof(line), "%sdrop%llu ovf%llu gap%llu",
                anyLoss ? "!" : " ",
                static_cast<unsigned long long>(stats.framesDropped),
                static_cast<unsigned long long>(stats.queueOverflows),
                static_cast<unsigned long long>(stats.packetContinuityViolations));
  drawTextRow(6, line);

  std::snprintf(line, sizeof(line), "%shttp %4d fail %u/%u",
                stats.consecutivePublishFailures > 0 ? "!" : " ",
                stats.lastPublishStatus,
                static_cast<unsigned>(stats.consecutivePublishFailures),
                static_cast<unsigned>(stats.publishErrors % 1000u));
  drawTextRow(7, line);

  static_assert(kStatusRows == 8, "renderFrame draws exactly kStatusRows lines");
}

}  // namespace mmpr
