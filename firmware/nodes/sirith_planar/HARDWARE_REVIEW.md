# Sirith Planar — Hardware Design Review (RP2354A + ESP32-C5)

**Status:** board in final design phase. This document is the firmware team's
review of the proposed schematic/pin map for the 5-mic PDM planar node. Items
marked **[BLOCKING]** must be resolved before layout is frozen; **[CONFIRM]**
items need a datasheet/measurement check by the board designer.

Node summary: 5 PDM MEMS mics in a center-square geometry (4 corners at a
25 mm radius + 1 center, coplanar), captured by an **RP2354A** (RP2350 core,
QFN-60, 2 MB in-package flash, **no CYW43439**). Wi-Fi/BT is offloaded to an
**ESP32-C5** that the RP2350 flashes over UART (esp-serial-flasher) and talks to
over SPI. On-board u-blox M10Q GPS provides PPS for ns-level timestamping.

The pin assignments below supersede the "Planar Array Proposed Hardware Mapping"
table in `firmware/nodes/sirith_tetrahedral/TODO.md` (lines ~144–168), which is
**stale on the PDM row** — it assumed 5 discrete data lines on GP0–GP4 + clock on
GP5. The corrected scheme is 3 data + 1 clock (see item 1).

---

## 1. [BLOCKING] PDM pin remap: 3 data + 1 clock, not 5 data + 1 clock

The five mics are wired **2 + 2 + 1** across three data lines using PDM
L/R-select (two mics per line, one driving on the clock rising edge and one on
the falling edge; the fifth mic uses a line to itself). A single PIO state
machine clocks and samples all three lines; core 1 deinterleaves during the CIC
pass, so the shared-line layout costs nothing in software and saves two GPIOs.

**PIO requires `in pins` to be a contiguous base+count block.** Proposed map:

| Signal        | GPIO       | Notes                                             |
|---------------|------------|---------------------------------------------------|
| PDM DATA0     | **GP1**    | mics ch0 (rising) + ch1 (falling)                 |
| PDM DATA1     | **GP2**    | mics ch2 (rising) + ch3 (falling)                 |
| PDM DATA2     | **GP3**    | mic  ch4 (rising); falling half-cycle discarded   |
| PDM CLK       | **GP4**    | PIO side-set output, 3.072 MHz                    |

`in pins` base = **GP1**, count = **3** (GP1–GP3 contiguous). Clock is a
separate side-set pin (GP4) and does not need to be contiguous with the data
block. This **frees GP0 and GP5** relative to the stale table — GP0 is then
reserved for PSRAM CS (item 2).

Channel/deinterleave mapping (firmware `node_config.h` documents the same table):
ch0..ch3 = the four corner mics, ch4 = center mic. Corner→physical-position
assignment is fixed in firmware geometry (item: geometry is firmware-defined,
25 mm radius, corners at (±r/√2, ±r/√2, 0), center at origin).

## 2. [BLOCKING] PSRAM CS reservation — GP0 primary, GP8 fallback

The RP2350 QMI chip-select #1 (`CS1n`, the APS PSRAM select) is only routable to
**GP0, GP8, or GP19** on the QFN-60 part. GP19 is already committed to I2C1 SCL,
so only **GP0 and GP8** remain. Even though PSRAM is not populated on this rev,
**reserve GP0 (primary) unconnected** — routable to a future PSRAM pad — and keep
**GP8 as fallback**. This is the reason PDM data starts at GP1 rather than GP0.

Leave the PSRAM footprint as a DNP option with GP0 routed to its CS pad; the
firmware ring-buffer sizing is one-constant expandable if PSRAM is later added
(`MMPR_NODECFG_PSRAM_CS_PIN`, reserved-but-unset in the planar config).

## 3. [BLOCKING] Erratum RP2350-E9 — external pull-downs on PDM data lines

RP2350 A2 silicon exhibits erratum **E9**: a GPIO configured as input with the
**internal pull-down** enabled can latch at ~2.1 V instead of pulling to 0. The
firmware therefore sets PDM data-pin bias = **none** (the internal `kPullDown`
option is unsafe on affected steppings and must not be used for these lines).

PDM data lines are **not driven continuously**:
- The single-mic line (GP3/ch4) floats on every *falling* half-cycle (that mic
  only drives on the rising edge).
- The shared lines (GP1, GP2) float briefly at the rising↔falling **handoff**
  between the two mics sharing the line.

A floating high-impedance input is exactly the E9 latch hazard. **Add external
~10 kΩ pull-downs on all three PDM data lines (GP1, GP2, GP3).** ~8.2–10 kΩ is
fine; the value must be weak enough not to fight the mic's active driver yet
strong enough to define the line during float.

Also review **GP11 (ESP host-wake input)** for the same hazard — it idles based
on the C5's drive and should have a defined external bias (pull-down if the C5
drives it high to signal, per item 7).

Confirm at design time which RP2354A **stepping** is procurable; if a non-E9
stepping is guaranteed, the external pull-downs are still recommended (they cost
nothing and define the float behavior of the shared PDM lines regardless).

## 4. [BLOCKING] Clock source — specify 12.288 MHz 2.5 ppm TCXO into XIN

The PDM clock must be an **integer divide of `clk_sys`** — any fractional PIO
`clkdiv` dithers the clock edges (~6.5 ns of jitter at these divisors), which is
not studio-grade and injects timing noise into cross-node TDOA.

- With the **stock 12.000 MHz** crystal the only integer path to a 3.072 MHz PDM
  clock is `clk_sys = 153.6 MHz = 50 × 3.072 MHz` — a **2.4 % overclock** (flagged
  as bench-only, item: risks), or a de-rated `76.8 MHz = 25 ×` fallback.
- The **final board must specify a 12.288 MHz, ≤2.5 ppm TCXO** driving XIN.
  Then `clk_sys = 122.88 MHz = 40 × 3.072 MHz`, fully in spec, PIO divider = 1.
  12.288 MHz is exactly divisible to standard audio rates, so no fractional-rate
  error is carried into sample-index→UTC mapping.

**[CONFIRM]** RP2350 XIN accepts an external CMOS clock — configure XOSC for
**external-clock mode**, not crystal mode. Verify the TCXO's output format
(clipped-sine vs CMOS), drive level, and load per the RP2350 datasheet XIN spec;
add the series/DC-block per the TCXO datasheet if it is a clipped-sine part.

Because the PDM clock and the CPU/PLL clocks both derive from this one TCXO, and
PPS disciplines absolute time, the node gets a known slowly-varying
sample-index→UTC mapping — the foundation for cross-node TDOA. Keep the M10Q PPS
routed to a GPIO (GP10, unchanged) so the existing PPS-capture path stays alive.

## 5. [CONFIRM] Buzzer pin missing from the pin map — propose GP6 (PWM)

The proposed table has no pin for the (future) ultrasonic self-localization
buzzer. Reserve **GP6** as a PWM-capable output to a piezo driver. The buzzer
firmware is a **stub** on this rev (chirp API present, body not built out), but
the pin and driver footprint should exist now.

Mechanical note: **mechanically isolate the piezo from the mic plane**
(grommet-mount or a flex pigtail). Board-conducted vibration rings the MEMS mics
well after a chirp ends and corrupts the very capture the chirp is meant to time.
A magnetic buzzer cannot produce useful SPL here — use a **piezo transducer with
a driver** (H-bridge or boost); PWM from GP6 drives the driver, not the piezo
directly. If a magnetic transducer is ever used, keep it away from the
magnetometer.

## 6. [CONFIRM] PDM mic timing at 3.072 MHz

Confirm the chosen MEMS PDM mic against the capture scheme:
- Supports a **3.072 MHz** clock in its high-performance/high-ratio mode (most
  parts support 1.0–3.6 MHz; verify the specific PN's max and that 3.072 MHz is
  in the high-performance band, not the low-power band).
- Data-valid delay **t_dv** after the sampling clock edge leaves a valid window
  at the firmware's **late-window sample point (~130–155 ns after the edge** at
  3.072 MHz). The PIO samples late in the half-cycle to clear t_dv; confirm the
  window from the mic's t_dv(max) and the line RC.
- Add a **22 Ω series resistor on the PDM clock** near the RP2350 to tame edge
  overshoot/EMI on the shared clock net.

## 7. [CONFIRM] ESP32-C5 link

- **[CONFIRM]** The 10 kΩ pull-ups on the SPI/DAT lines (GP20–23 ↔ C5 DAT0–3)
  must not fight the C5's **strapping** requirements. The C5 boot-mode strap pin
  (driven from RP2350 GP15 / ESP BOOT) is sampled at every reset — ensure nothing
  on that net (pull-ups, LEDs) overrides the strap, and that the RP2350 drives it
  to the **run** level except during flashing.
- **Flash path:** RP2350 UART1 (GP28 TX / GP29 RX) ↔ C5 UART0, plus **EN** (GP14,
  10 kΩ pull-up + 0.1 µF to GND for a clean reset RC) and **BOOT** (GP15).
  esp-serial-flasher over UART at **921600** baud. No USB on the C5.
- **Data path:** SPI0 (GP20–23), CS on GP21, host-wake on GP11. Prefer a
  **WROOM-1U** module (external 5 GHz antenna) for an enclosed outdoor node; keep
  the M10Q GPS on the board edge **opposite** the C5 antenna feed.

## 8. Flash budget

RP2354A has **2 MB in-package flash**. The planar firmware image carries **no
radio stacks** (no cyw43 driver, no lwIP, no btstack) — it fits comfortably. The
**C5 application image is NOT stored in RP2350 flash** (2 MB cannot hold it); the
C5 is **host-flashed at bench** via `scripts/flash_esp_c5.py` over UART. OTA
update of the C5 image (streamed from the backend through the RP2350) is a future
item; day-1 flasher scope is download-mode entry, SYNC, chip-ID, and
app-version check only.

---

## Consolidated pin map (planar, RP2354A QFN-60)

| Function            | GPIO(s)                     | Direction | Notes                                             |
|---------------------|-----------------------------|-----------|---------------------------------------------------|
| PSRAM CS (reserved) | **GP0** (fallback GP8)      | —         | DNP; QMI CS1n; unconnected this rev (item 2)      |
| PDM DATA0/1/2       | **GP1 / GP2 / GP3**         | in        | contiguous `in pins` base=GP1 count=3; ext 10k PD |
| PDM CLK             | **GP4**                     | out       | PIO side-set, 3.072 MHz; 22 Ω series (items 1,6)  |
| Buzzer PWM          | **GP6**                     | out       | to piezo driver; stub firmware (item 5)           |
| GPS PPS             | GP10                        | in        | M10Q PPS, existing PPS capture path               |
| ESP host-wake       | GP11                        | in        | ext bias per E9 (item 3)                          |
| GPS UART (TX/RX)    | GP12 / GP13                 | out/in    | to M10Q RX/TX                                      |
| ESP EN              | GP14                        | out       | 10k pull-up + 0.1 µF RC                            |
| ESP BOOT            | GP15                        | out       | strap-aware (item 7)                              |
| SDIO reserved       | GP16 / GP17                 | —         | future SDIO CLK/CMD; not used this rev            |
| I2C1 (SDA/SCL)      | GP18 / GP19                 | io        | 4.7k pull-ups; SHT45 + IMU + mag (+ baro)         |
| ESP SPI0 RX/CS/SCK/TX | GP20 / GP21 / GP22 / GP23 | io        | 10k pull-ups; strap-checked (item 7)              |
| Status LED          | GP26                        | out       | existing FET/LED pattern                          |
| I2C activity LED    | GP27                        | out       | existing PWM-dimmed pattern                       |
| ESP UART1 TX/RX     | GP28 / GP29                 | out/in    | flasher + console at 921600                        |

Changes vs the stale TODO table: PDM is 3 data + 1 clock on GP1–GP4 (was 5 data
GP0–GP4 + clock GP5); GP0 reserved for PSRAM; buzzer added on GP6.

---

## Open confirmations for the board designer (checklist)

- [ ] External ~10 kΩ pull-downs on GP1/GP2/GP3 (E9); defined bias on GP11.
- [ ] 12.288 MHz ≤2.5 ppm TCXO into XIN, external-clock XOSC mode, drive format verified.
- [ ] RP2354A stepping procurable; note if non-E9.
- [ ] Chosen PDM mic: 3.072 MHz high-performance support + t_dv window at ~130–155 ns.
- [ ] 22 Ω series on PDM clock (GP4).
- [ ] Buzzer driver footprint on GP6; piezo mechanically isolated from mic plane.
- [ ] C5 BOOT/EN strap integrity; SPI/DAT pull-ups don't fight straps.
- [ ] PSRAM footprint DNP with GP0 routed to CS pad.
- [ ] GPS antenna keepout, board edge opposite C5; SHT45 thermal isolation from C5/bucks.
