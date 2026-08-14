idf_component_register(
    SRCS sirith_tetra_s3.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../include
    REQUIRES minimap_node_runtime
             minimap_audio_esp32s3
             minimap_transport_esp_wifi
             minimap_ui_tft_st7789
             nvs_flash
             esp_hw_support)

# Build stamp reported to the server as the node's firmware version.
if(NOT DEFINED MMPR_FW_VERSION)
    set(MMPR_FW_VERSION "dev")
endif()
target_compile_definitions(${COMPONENT_LIB} PRIVATE MMPR_FW_VERSION="${MMPR_FW_VERSION}")
