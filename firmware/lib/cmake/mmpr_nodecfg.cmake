# mmpr_nodecfg.cmake
#
# Shared helper that forwards a node's `node_config.h`-derived settings onto a
# firmware target. Hoisted verbatim (behaviour-preserving) out of the per-node
# CMakeLists so every node target (tetra, planar, ...) applies node config the
# same way.
#
#   include(${FIRMWARE_LIB_CMAKE_DIR}/mmpr_nodecfg.cmake)
#   mmpr_apply_nodecfg(<target> <path-to-node_config.h>)
#
# Effects on <target>:
#   * MMPR_ENABLE_BLE_SCAN is read from node_config.h (single source of truth)
#     and exported to the CALLER's scope as ${MMPR_ENABLE_BLE_SCAN} so the node
#     CMakeLists can conditionally link the BLE/btstack stacks. It is also passed
#     to the target as a compile definition.
#   * Any -DMMPR_NODECFG_<NAME> cache/CLI overrides are forwarded as compile
#     definitions (numeric knobs), plus the two string URL knobs.
#   * The header is registered in CMAKE_CONFIGURE_DEPENDS so editing it alone
#     triggers a reconfigure (otherwise a stale cached value would win).

function(mmpr_apply_nodecfg target node_config_path)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${node_config_path}")

    file(STRINGS "${node_config_path}" _mmpr_ble_line
            REGEX "^#define[ \t]+MMPR_ENABLE_BLE_SCAN[ \t]+[0-9]+")
    if(_mmpr_ble_line MATCHES "MMPR_ENABLE_BLE_SCAN[ \t]+([0-9]+)")
        set(_mmpr_ble ${CMAKE_MATCH_1})
    else()
        message(FATAL_ERROR "mmpr_apply_nodecfg: could not find '#define MMPR_ENABLE_BLE_SCAN <0|1>' in ${node_config_path}")
    endif()
    # Export to caller scope so the node CMakeLists can gate btstack linking.
    set(MMPR_ENABLE_BLE_SCAN ${_mmpr_ble} PARENT_SCOPE)
    target_compile_definitions(${target} PRIVATE MMPR_ENABLE_BLE_SCAN=${_mmpr_ble})

    # Numeric knobs: forward only those explicitly overridden at configure time.
    foreach(_def
            AUDIO_INPUT_MODE
            AUDIO_SAMPLE_RATE_HZ
            AUDIO_CHANNELS
            AUDIO_FRAME_SAMPLES
            AUDIO_RING_FRAMES
            AUDIO_QUEUE_SLOTS
            PUBLISH_BATCH_FRAMES
            PUBLISH_BATCH_BYTE_BUDGET
            USE_PUBLISH_BATCH_BYTE_BUDGET
            HTTP_TIMEOUT_MS
            BLE_SCAN_INTERVAL_UNITS
            BLE_SCAN_WINDOW_UNITS
            BLE_REPORT_INTERVAL_MS
            BLE_REPORT_MAX_OBSERVATIONS
            GPS_PPS_PIN
            I2S_MONO_CHANNEL_SIDE
            I2S_MONO_SAMPLE_EDGE
            I2S_MONO_CAPTURE_BIT_OFFSET
            I2S_MONO_DATA_PIN_BIAS
            I2S_MONO_ENABLE_WORD_DIAGNOSTICS
            TDM_SAMPLE_EDGE
            TDM_CAPTURE_BIT_OFFSET
            TDM_DATA_PIN_BIAS
            TDM_ENABLE_WORD_DIAGNOSTICS)
        if(DEFINED MMPR_NODECFG_${_def})
            target_compile_definitions(${target}
                    PRIVATE MMPR_NODECFG_${_def}=${MMPR_NODECFG_${_def}})
        endif()
    endforeach()

    # String knobs (URLs) forwarded quoted.
    if(DEFINED MMPR_NODECFG_SERVER_BASE_URL)
        target_compile_definitions(${target}
                PRIVATE MMPR_NODECFG_SERVER_BASE_URL="${MMPR_NODECFG_SERVER_BASE_URL}")
    endif()
    if(DEFINED MMPR_NODECFG_BLE_SERVER_BASE_URL)
        target_compile_definitions(${target}
                PRIVATE MMPR_NODECFG_BLE_SERVER_BASE_URL="${MMPR_NODECFG_BLE_SERVER_BASE_URL}")
    endif()
endfunction()
