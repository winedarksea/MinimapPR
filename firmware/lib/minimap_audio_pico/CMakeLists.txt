# minimap_audio_pico — pico-sdk audio capture sources (INTERFACE library)
#
# TDM (ADAU7112) and mono-I2S PIO+DMA capture sources plus their shared serial
# timing helper. PdmCicDecimator (portable, host-testable CIC/halfband DSP
# core) and PdmPlanarSource (PIO+DMA capture + core-1 decimation worker) are
# the Phase 2 5-mic PDM planar capture path. The PIO program header is
# generated against the consuming executable target.

add_library(minimap_audio_pico INTERFACE)

target_sources(minimap_audio_pico INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/src/PicoI2SMonoSource.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/SirithPicoTdmSource.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/PdmCicDecimator.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/PdmPlanarSource.cpp
)

target_include_directories(minimap_audio_pico INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(minimap_audio_pico INTERFACE
        pico_stdlib
        pico_multicore
        hardware_dma
        hardware_pio
)

set(MINIMAP_AUDIO_PICO_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "minimap_audio_pico source dir")
function(minimap_audio_pico_generate_pio target)
    pico_generate_pio_header(${target} ${MINIMAP_AUDIO_PICO_DIR}/src/mmpr_audio_rx.pio)
    pico_generate_pio_header(${target} ${MINIMAP_AUDIO_PICO_DIR}/src/mmpr_pdm_rx.pio)
endfunction()
