# minimap_audio_esp32s3 — ESP32-S3 I2S/TDM capture (IDF component)
#
# EspI2sTdmSource is the S3 counterpart of minimap_audio_pico's
# SirithPicoTdmSource; EspI2sTdmLoopback is its bare-board self-test. Unlike
# minimap_transport_esp_wifi this library IS chip-specific (SOC_I2S_SUPPORTS_TDM).
#
# Not to be confused with the older PlatformIO-era lib/minimap_audio_esp32, which
# is Arduino driver/i2s.h code for the point_single_mic node.

idf_component_register(
    SRCS ${CMAKE_CURRENT_LIST_DIR}/src/EspI2sTdmSource.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/EspI2sTdmLoopback.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/include
    REQUIRES minimap_node_runtime esp_driver_i2s esp_driver_gpio esp_timer)
