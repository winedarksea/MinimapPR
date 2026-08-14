# sirith_tetra_s3 — ESP-IDF project for the ESP32-S3 tetrahedral node.
#
# A directory named in EXTRA_COMPONENT_DIRS that itself contains a CMakeLists.txt
# is registered as a single component named after the directory
# (tools/cmake/project.cmake). So the four shared libraries below are listed
# individually; nothing else under firmware/lib/ is ever scanned, which is what
# keeps the pico-only libraries out of this build.
cmake_minimum_required(VERSION 3.16)

set(FIRMWARE_LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../../lib)
set(EXTRA_COMPONENT_DIRS
    ${FIRMWARE_LIB_DIR}/minimap_node_runtime
    ${FIRMWARE_LIB_DIR}/minimap_audio_esp32s3
    ${FIRMWARE_LIB_DIR}/minimap_transport_esp_wifi
    ${FIRMWARE_LIB_DIR}/minimap_ui_tft_st7789)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# node_config.h knobs overridable from the command line, e.g.
#   idf.py -DMMPR_NODECFG_AUDIO_INPUT_MODE=0 -DMMPR_NODECFG_ENABLE_TFT_STATUS=0 build
# The ESP-IDF equivalent of lib/cmake/mmpr_nodecfg.cmake on the Pico side: each
# name that is actually set on the command line is forwarded as a compile
# definition to every component, so an override reaches node_config.h's #ifndef
# guards without editing the file. Unset names fall through to the header's
# defaults, so a plain `idf.py build` is the documented default configuration.
set(MMPR_NODECFG_FORWARDED_OPTIONS
    MMPR_NODECFG_AUDIO_INPUT_MODE
    MMPR_NODECFG_SERVER_BASE_URL
    MMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ
    MMPR_NODECFG_AUDIO_CHANNELS
    MMPR_NODECFG_AUDIO_FRAME_SAMPLES
    MMPR_NODECFG_AUDIO_RING_FRAMES
    MMPR_NODECFG_AUDIO_QUEUE_SLOTS
    MMPR_NODECFG_PUBLISH_BATCH_FRAMES
    MMPR_NODECFG_PUBLISH_BATCH_BYTE_BUDGET
    MMPR_NODECFG_USE_PUBLISH_BATCH_BYTE_BUDGET
    MMPR_NODECFG_HTTP_TIMEOUT_MS
    MMPR_NODECFG_TDM_WS_WIDTH
    MMPR_NODECFG_TDM_WS_POLARITY_HIGH
    MMPR_NODECFG_TDM_BIT_SHIFT
    MMPR_NODECFG_TDM_LEFT_ALIGN
    MMPR_NODECFG_TDM_CAPTURE_BIT_OFFSET
    MMPR_NODECFG_ENABLE_TFT_STATUS
    MMPR_NODECFG_ENABLE_I2S_LOOPBACK
    MMPR_NODECFG_ENABLE_STATS_SERVER
    MMPR_NODECFG_ENABLE_CLOCK_HOLDOVER_MAINTENANCE)
foreach(option IN LISTS MMPR_NODECFG_FORWARDED_OPTIONS)
    if(DEFINED ${option})
        idf_build_set_property(COMPILE_DEFINITIONS "${option}=${${option}}" APPEND)
        message(STATUS "nodecfg override: ${option}=${${option}}")
    endif()
endforeach()

project(sirith_tetra_s3)
