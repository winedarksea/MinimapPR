# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2_w CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(sirith_tetrahedral C CXX ASM)

set(PICO_CXX_ENABLE_EXCEPTIONS 1)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Shared firmware libraries (INTERFACE) + the nodecfg forwarding helper.
set(FIRMWARE_LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../../lib)
include(${FIRMWARE_LIB_DIR}/cmake/mmpr_nodecfg.cmake)
add_subdirectory(${FIRMWARE_LIB_DIR}/minimap_node_runtime
        ${CMAKE_BINARY_DIR}/minimap_node_runtime)
add_subdirectory(${FIRMWARE_LIB_DIR}/minimap_transport_cyw43
        ${CMAKE_BINARY_DIR}/minimap_transport_cyw43)
add_subdirectory(${FIRMWARE_LIB_DIR}/minimap_audio_pico
        ${CMAKE_BINARY_DIR}/minimap_audio_pico)

# Add executable. Default name is the project name, version 0.1
add_executable(sirith_tetrahedral
        sirith_tetrahedral.cpp
        )

# Generated PIO headers (need the concrete executable target).
minimap_audio_pico_generate_pio(sirith_tetrahedral)
minimap_node_runtime_generate_pio(sirith_tetrahedral)

pico_set_program_name(sirith_tetrahedral "sirith_tetrahedral")
pico_set_program_version(sirith_tetrahedral "0.1")

# Modify the below lines to enable/disable output over UART/USB
pico_enable_stdio_uart(sirith_tetrahedral 0)
pico_enable_stdio_usb(sirith_tetrahedral 1)

# Node-local include (node_config.h).
target_include_directories(sirith_tetrahedral PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/include
)

# Runtime + transports + audio capture come from the shared libs.
target_link_libraries(sirith_tetrahedral
        minimap_node_runtime
        minimap_transport_cyw43
        minimap_audio_pico
        )

# Forward node_config.h-derived settings (also exports MMPR_ENABLE_BLE_SCAN).
mmpr_apply_nodecfg(sirith_tetrahedral ${CMAKE_CURRENT_LIST_DIR}/include/node_config.h)

if(MMPR_ENABLE_BLE_SCAN)
    target_link_libraries(sirith_tetrahedral
            pico_btstack_ble
            pico_btstack_cyw43)
    target_compile_definitions(sirith_tetrahedral PRIVATE CYW43_ENABLE_BLUETOOTH=1)
endif()

pico_add_extra_outputs(sirith_tetrahedral)
