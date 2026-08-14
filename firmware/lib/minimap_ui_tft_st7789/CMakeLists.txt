# minimap_ui_tft_st7789 — 240x135 ST7789 status page (IDF component)
#
# Uses only esp_lcd's built-in esp_lcd_panel_st7789 driver plus an embedded 8x8
# font, so there is no component-manager dependency and no LVGL.

idf_component_register(
    SRCS ${CMAKE_CURRENT_LIST_DIR}/src/TftStatusDisplay.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/include
    REQUIRES minimap_node_runtime esp_lcd esp_driver_spi esp_driver_gpio esp_timer)
