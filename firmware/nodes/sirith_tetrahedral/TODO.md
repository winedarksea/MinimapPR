# Sirith Tetrahedral Pi Pico 2 W Firmware TODO

## Hardware Profile (Target)
- 4-mic tetrahedral array, regular tetrahedron geometry, 50 mm edge length (40.82 mm height, minus 1.6 mm pcb).
- MEMS mics through ADAU7112 in TDM mode.
- Optional GPS (M10Q-style): TX=`GP12`, RX=`GP13`, PPS=`GP10`.
- I2C: SDA=`GP18`, SCL=`GP19` (LIS2MDLTR + optional LSM6 temp).
- TDM: SDATA=`GP7`, BCLK=`GP8`, WS=`GP9`.
- MK4 is the top mic (expected default TDM slot 3).
- FET (p-channel connected to LED and switchable Vin power header which can be used to send power off board) is `GP26`. This could be used as a status LED (in addition to Pico onboard LED) as most deployments won't use the offboard power connection.
- Expected channels, MK1 is DATA2 (Right) of i2s1 (TDM2 likely), MK2 is DATA1 (Left) of i2s1 (TDM1 likely), MK3 is DATA2 (Right) of i2s2 (TDM4 likely), MK4 is DATA1 (Left) of i2s2 (TDM3 likely).
- Microphones and ADAU7112 are at 1.8V, talking to the MCU via a TXS0108E level shifter.
- Using the same X/Y/Z as the LIS2MDLTR and LSM6DSV16X, MK3 is (0, 0, 0), MK1 is (0, 50, 0), MK2 is (43.3, 25, 0), MK4 is (21.65, 25, 40.82)
- An optional I2C connection (qwiic) to an SHT45 temperature and humidity sensor board
- Note the Pico 2 W has an erratum 9 on gpio to be aware of. The board also has its own LED wired to WL_GPIO0.

## Phase 1: Project and Build Foundation
- [x] Replace generated demo firmware with Sirith tetra node firmware entrypoint.
- [x] Add local node configuration header for pinout, network, geometry, and feature flags.
- [x] Update CMake target to include required Pico SDK networking/peripheral libraries.
- [x] Ensure debug logging is enabled on a usable stdio transport for bring-up.

## Phase 2: Core Runtime Port (PlatformIO -> Pico SDK)
- [x] Port frame/data model types from `minimap_node_core` to Pico-SDK-compatible C++ (`std::string`, no Arduino deps).
- [x] Port JSON protocol encoder and Base64 frame encoding.
- [x] Port node clock behavior for frame timestamps (monotonic baseline, GPS/NTP hook-ready).
- [x] Port runner loop: capture frame, attach metadata/environmental sample, publish, collect stats.

## Phase 3: Sirith Audio + Peripherals
- [x] Port `SirithPicoTdmSource` to Pico SDK build (PIO-based TDM master receive).
- [x] Keep safe electrical defaults (low drive + slow slew where available on BCLK/WS).
- [x] Implement slot->physical mic mapping + base-plane rotation support.
- [x] Implement optional LIS2MDLTR auto-orientation polling with smoothing/stability logic.
- [x] Implement optional LSM6 temperature telemetry source on I2C.
- [x] Initialize optional GPS UART/PPS pins non-destructively.

## Phase 4: Pico W Networking + Ingest Publish
- [x] Implement Wi-Fi connect/reconnect utilities for Pico W (`cyw43`).
- [x] Implement HTTP POST publisher for `/api/v1/ingest/frame` without Arduino networking stack.
- [x] Support endpoint parsing (`http://host[:port]/path`) with hostname/IP resolution.
- [x] Return publish status + optional response text for diagnostics.

## Phase 5: Bring-Up and Verification
- [x] Build successfully for `pico2_w` in this folder using the Raspberry Pi VS Code CMake flow.
- [x] Verify no compile-time dependency on PlatformIO/Arduino headers.
- [x] Verify runtime loop sequencing (boot, optional peripheral init, Wi-Fi, runner start, loop/publish).
- [x] Document what remains hardware-in-the-loop only (mic/gps/imu/live backend tests).
- [ ] Hardware-in-the-loop: verify live TDM capture from ADAU7112 wiring/slot straps.
- [ ] Hardware-in-the-loop: verify end-to-end POST ingest to a reachable MinimapPR backend.
- [ ] Hardware-in-the-loop: verify optional LIS2MDLTR/LSM6/GPS behavior on real hardware.

## Phase 6: Modularization and Shared Library Reuse
- [x] Split monolithic node implementation into runtime modules (`NodeRunner`, protocol, publisher, clock, Wi-Fi support).
- [x] Reuse upstream shared audio source (`firmware/lib/minimap_audio_pico/src/SirithPicoTdmSource.cpp`) in this build.
- [x] Add generic temperature sensor interface (`ITemperatureSensor`) decoupled from specific IMU chips.
- [x] Adapt generic temperature sensors into node environmental telemetry via `TemperatureEnvironmentalSource`.
- [x] Keep optional LIS2 auto-orientation and GPS setup integrated in thin node entrypoint wiring.
- [x] Update CMake target to compile modular sources and shared audio library source/include paths.

## Notes / Calibration
- Manual base-plane rotation must remain configurable for installation alignment.
- Auto-orientation is optional and should gracefully fall back to manual rotation if unhealthy.
- Design emphasis: safe defaults and non-destructive GPIO behavior.
