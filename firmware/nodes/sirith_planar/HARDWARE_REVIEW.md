# Sirith Planar — Hardware Design Review (RP2354A + ESP32-C5)

*Revised 2026-11-24. This revision changes the pin map; see "What changed" below.*

**Status:** board in final design phase. This is the firmware team's review of the
proposed schematic and pin map for the 5-mic PDM planar node. It is the single
authoritative pin map and board checklist, and it replaces the earlier pin table in
`firmware/nodes/sirith_tetrahedral/TODO.md`, now marked superseded. The reasoning
behind each item is in `Board Redesign Assessment — 5×PDM Planar.md` (repo root),
cited as "assessment §n".

- **[BLOCKING]** items must be resolved before layout is frozen.
- **[CONFIRM]** items need a datasheet or measurement check by the board designer.
- Datasheet references: **DS** = RP2350 datasheet, **HW** = *Hardware design with
  RP2350*, **C5** = ESP32-C5 series datasheet, **MOD** = ESP32-C5-WROOM-1/1U module
  datasheet.

**Node summary.** Five PDM MEMS mics in a centre-square layout (four corners on a
25 mm radius plus one centre mic, coplanar), captured by an **RP2354A** (RP2350 core,
QFN-60, 2 MB in-package flash, no CYW43439). Wi-Fi and BLE run on an **ESP32-C5**
module, which the RP2350 flashes over UART and sends audio to over SPI. An on-board
u-blox M10Q GPS supplies PPS for timestamping.

## What changed in this revision

| Item | Before | Now | Why |
|---|---|---|---|
| ESP32-C5 UART | GP28/GP29, labelled UART1 | **GP24/GP25** (UART1) | GP28/29 are UART0, the same UART as the GPS on GP12/13 (item 3) |
| ESP32-C5 data wiring | SPI on the C5's SDIO DAT pins, plus GP16/17 reserved for SDIO CLK/CMD | SPI on the C5's dedicated SPI pins; **no SDIO reservation** | SDIO cannot raise end-to-end throughput, and it used up the C5's USB pins (item 5) |
| Clock source | 12.288 MHz TCXO, blocking | 12.288 MHz, **dual footprint**: crystal fitted, CMOS TCXO optional | The TCXO only helps when GPS is lost; a clipped-sine TCXO does not meet XIN levels (item 4) |
| PSRAM | Footprint, not populated | **Populated** | Radio stalls outlast the internal-RAM queue (item 2) |
| PDM pull-downs | ~10 kΩ, blocking (erratum E9) | 8.2 kΩ, confirm | E9 is fixed from the A3 stepping; RP2354 parts are expected to be A4 (item 6) |
| SWD access | Suggested | **Required** | USB boot does not work at 12.288 MHz until OTP is programmed (item 9) |
| C5 "ready for frame" line | Not present | **GP16**, from C5 GPIO1 | Standard flow control for an ESP SPI slave; cheap to route now (item 5) |
| Spare GPIOs | GP5, GP7, GP8, GP9, GP24, GP25 | GP5, GP7, GP8, GP9, GP17, GP28, GP29 | From the rows above |

Items 1 (PDM pins) and 7 (mic timing) are unchanged.

---

## 1. [BLOCKING] PDM pins: 3 data + 1 clock

The five mics are wired **2 + 2 + 1** on three data lines using the PDM L/R select
pin: two mics share a line, one driving after the rising clock edge and one after
the falling edge. The fifth mic has a line to itself. One PIO state machine clocks
and samples all three lines, so the sharing saves two GPIOs at no software cost.

PIO input pins must be a contiguous block.

| Signal | GPIO | Notes |
|---|---|---|
| PDM DATA0 | **GP1** | ch0 (rising half) + ch1 (falling half) |
| PDM DATA1 | **GP2** | ch2 (rising half) + ch3 (falling half) |
| PDM DATA2 | **GP3** | ch4 (rising half); the falling half is discarded |
| PDM CLK | **GP4** | PIO side-set output, 3.072 MHz |

Channel order is fixed in firmware (`node_config.h`): ch0–ch3 are the corner mics
(NE, NW, SW, SE) at (±r/√2, ±r/√2, 0) with r = 25 mm, and ch4 is the centre mic at
the origin. Set each mic's L/R select pin to match the rising/falling column above.

GP0 is left for the PSRAM chip select (item 2), which is why the data block starts
at GP1.

## 2. [BLOCKING] PSRAM: populate it, chip select on GP0

Fit an 8 MB (64 Mbit) 3.3 V QSPI PSRAM, for example an APS6404L, on the first board
spin. The 2026-11-23 capacity bench showed radio stalls of 0.6–1.2 s, and the
RP2350's internal RAM holds under 0.7 s of 5 × 48 kHz audio. 8 MB holds about 16 s
(assessment §2).

- **Bus.** The PSRAM shares the RP2354A's QSPI pins with the in-package flash. Those
  pins are bonded out for this purpose (DS §14.3, HW §3.2). Keep the traces short.
- **Chip select.** The second QSPI chip select (QMI CS1n) can only go to GP0, GP8 or
  GP19 on the QFN-60 (DS Table 677). GP19 is I2C1 SCL, so use **GP0**. GP8 stays
  free as a fallback.
- **Pull-up on CS.** Add **10 kΩ from GP0 to 3.3 V**. RP2350 GPIOs power up with a
  weak pull-down, which would select the PSRAM during boot (HW §3.2).
- **Supplies.** IOVDD and QSPI_IOVDD must both be 3.3 V when a second QSPI device is
  fitted (DS Table 1688 note).

## 3. [BLOCKING] ESP32-C5 UART moves to GP24/GP25

The previous map put the GPS on GP12/GP13 and the ESP32-C5 flash/console UART on
GP28/GP29, labelled UART1. Both pairs are **UART0** (DS Table 677), and one UART
cannot serve two devices. GP28/29 have no alternative UART function.

- **ESP32-C5 UART: GP24 (UART1 TX) and GP25 (UART1 RX).** Both were unused.
- **GPS stays on GP12/GP13 (UART0)**, the same pins as the tetrahedral node.
- **GP28/GP29 become spare.** They are ADC inputs, so they suit supply-voltage
  sensing (item 11).

## 4. [BLOCKING] Clock source: 12.288 MHz, crystal with an optional TCXO

**Frequency.** Specify **12.288 MHz**, not the usual 12.000 MHz. The 3.072 MHz PDM
clock must be an exact integer division of the system clock; a fractional divider
moves the clock edges by about 6.5 ns, which adds timing noise to the audio. From
12.288 MHz the firmware can run the system clock at 122.88 MHz (40 × 3.072) or
147.456 MHz (48 × 3.072), both within the 150 MHz limit. From 12.000 MHz the
fastest in-spec integer option is 76.8 MHz, half speed; 153.6 MHz is an overclock.

**Dual footprint.** Lay out both parts on XIN and select with a 0 Ω link:

| Option | Part | Fit when |
|---|---|---|
| A (default) | 12.288 MHz crystal, with load capacitors and series resistor as in HW §4 | Normal builds |
| B | 12.288 MHz **CMOS-output** TCXO, 3.3 V, ≤ 2.5 ppm, into XIN; XOUT left open | Nodes that must hold timing through GPS loss, or that sit in direct sun |

The TCXO does not improve timing while GPS is locked, because PPS corrects the
oscillator every second. It only reduces drift after lock is lost (assessment §5).

- **[CONFIRM] The TCXO must have a CMOS (square-wave) output at 3.3 V.** XIN needs
  at least 0.65 × IOVDD high and at most 0.35 × IOVDD low (DS Table 1686). The common
  clipped-sine TCXOs (about 0.8 V peak-to-peak) do not meet this without a buffer.
- **[CONFIRM] Crystal start-up.** Raspberry Pi has validated only one crystal, the
  12 MHz Abracon ABM8-272-T3 with a 1 kΩ series resistor (DS §8.2.1.1, HW §4). A
  12.288 MHz crystal needs its own start-up check across temperature. Choosing the
  12.288 MHz part from the same family is the lowest-risk option.
- Place the link so the unfitted option leaves the shortest possible stub on XIN.

**Consequence for programming.** The RP2350's USB boot mode assumes 12 MHz. On a
12.288 MHz board it will not enumerate until the board's OTP is programmed
(assessment §5), so a blank board must first be programmed over SWD. See item 9.

## 5. [BLOCKING] ESP32-C5 link

Use an **ESP32-C5-WROOM-1** module, or the **WROOM-1U** (external antenna connector)
for an enclosed outdoor node. The XIAO ESP32-C5 does not break out enough pins.

### Wiring

| Function | RP2350 | ESP32-C5 | Board notes |
|---|---|---|---|
| UART, host → C5 | GP24 (UART1 TX) | GPIO12 (U0RXD) | Flashing and console |
| UART, C5 → host | GP25 (UART1 RX) | GPIO11 (U0TXD) | The module has a 499 Ω series resistor on this pin (MOD schematic) |
| Reset | GP14 | EN | 10 kΩ pull-up and RC to ground; **[CONFIRM]** the capacitor value against Espressif's C5 hardware design guidelines (the earlier table had 0.1 µF) |
| Boot strap | GP15 | GPIO28 | **10 kΩ pull-up to 3.3 V**; unpopulated capacitor footprint, as Espressif recommends |
| SPI clock | GP22 (SPI0 SCK) | GPIO6 (FSPICLK) | 22 Ω series at the RP2350 |
| SPI host → C5 | GP23 (SPI0 TX) | GPIO7 (FSPID) | |
| SPI C5 → host | GP20 (SPI0 RX) | GPIO2 (FSPIQ) | |
| SPI chip select | GP21 (SPI0 CSn) | GPIO10 (FSPICS0) | **10 kΩ pull-up**, so the C5 is deselected while the RP2350 is in reset |
| Response ready | GP11 (in) | GPIO4 (out) | The C5 raises it when a response is waiting; 10 kΩ pull-down |
| Ready for frame | GP16 (in) | GPIO1 (out) | New. The C5 raises it when it can accept the next SPI transfer; 10 kΩ pull-down |
| USB debug | — | GPIO13 (D−), GPIO14 (D+) | Test pads or an unpopulated USB connector; no pull-ups |

- **[CONFIRM]** the four SPI pin numbers against C5 Table 2-3 at schematic entry.
  These are the C5's dedicated SPI pins, rated to 40 MHz in slave mode (C5 §4.2.1.2).
- **GP16 "ready for frame"** is not used by the current firmware. Espressif's SPI
  slave examples and ESP-Hosted both use a line like this so the host never clocks
  data before the slave has a buffer queued. Routing it now costs one spare pin.

### Boot straps

- **Download mode needs only GPIO28 low** while EN rises, held for at least 3 ms
  after EN goes high (C5 Tables 3-2, 3-3). The RP2350 drives GP15 low for flashing
  and high at all other times.
- **The 10 kΩ pull-up on GPIO28 is required.** RP2350 pins are high-impedance with a
  weak pull-down while the RP2350 is in reset or unprogrammed, and the C5's internal
  pull-up is also weak. Without the resistor the C5 can start in download mode.
- **Leave GPIO27 unconnected.** It must read high for UART download and has an
  internal pull-up. Put nothing on it that pulls low (the Espressif devkit puts an
  RGB LED there).
- **Do not add pull resistors to GPIO2, GPIO3, GPIO7, GPIO25 or GPIO26.** These are
  the C5's other strapping pins. With default eFuses none of them affects booting
  from flash, and GPIO2 and GPIO7 are only driven after boot.

### Why SDIO is no longer reserved

The earlier map kept GP16/GP17 and the four SPI lines wired so the link could later
become SDIO. This revision drops that:

- Espressif's published open-air TCP upload figure for the C5 is about 20 Mbit/s,
  and a 40 MHz SPI link carries about 25 Mbit/s, so a faster bus would not raise
  end-to-end throughput (assessment §2).
- The RP2350 has no SDIO host peripheral. SDIO would need a PIO-built host and a
  port of Espressif's host stack (assessment §3).
- SDIO DAT2/DAT3 are the C5's USB pins. Freeing them gives the C5 a USB
  serial/JTAG port for flashing and debugging the bridge firmware directly.

## 6. [CONFIRM] Erratum RP2350-E9: pull-downs on the PDM data lines

Erratum E9 (an input pad holding itself near 2.2 V against the internal pull-down)
affects the **A2 stepping only**. It is fixed in A3, and A4 has no pad changes from
A3 (DS Appendix C and D.5.1). The RP2354 was launched together with A4, and
Raspberry Pi's change notice says other steppings are end-of-life, but no document
states the RP2354 stepping outright.

- **[CONFIRM] the stepping on received parts**: the last characters of the top
  marking (DS §14.4).
- **Fit 8.2 kΩ pull-downs on GP1, GP2 and GP3 regardless.** The PDM lines are not
  driven continuously: the single-mic line floats for half of every clock period,
  and the shared lines float briefly at each hand-over. The resistors define the
  line during those gaps, and 8.2 kΩ is also Raspberry Pi's stated E9 workaround
  for A2 parts.
- Firmware never enables the internal pull-downs on these pins.

## 7. [CONFIRM] PDM mic timing at 3.072 MHz

Check the chosen MEMS mic against the capture scheme:

- It accepts a **3.072 MHz** clock in its high-performance mode. Most parts cover
  1.0–3.6 MHz; confirm 3.072 MHz is not in a low-power band.
- Its data-valid delay leaves a valid window at the firmware's sample point,
  **about 130–155 ns after the clock edge** (the half period is 163 ns). Use the
  mic's maximum data-valid delay plus the line's RC.
- Add a **22 Ω series resistor on the PDM clock** at the RP2350.
- If ultrasonic capture is ever wanted, note the mic's maximum clock frequency. The
  next integer option is 6.144 MHz, which most mics do not accept (assessment §4).

## 8. Buzzer: footprint only, on GP6

Reserve **GP6** (PWM) for an ultrasonic ranging buzzer. Lay out the piezo and driver
footprints but **leave the parts unpopulated**; the firmware is a stub and the
feature is not scoped (assessment §4).

- Use a piezo with a driver (H-bridge or boost). GP6 drives the driver, not the
  piezo. A magnetic buzzer cannot produce useful output in this band.
- Mount the piezo off the mic plane (grommet or flex pigtail). Board-conducted
  vibration rings the MEMS mics after a chirp ends.
- Keep any magnetic part away from the magnetometer.

## 9. [BLOCKING] Programming and bring-up access

- **SWD header or Tag-Connect: required.** It is the only way to program a blank
  12.288 MHz board (item 4), and it does not depend on the crystal.
- **RP2350 USB**, for the bench console and for USB boot once OTP is programmed.
- **ESP32-C5 USB pads** (GPIO13/14, item 5). The C5 can be flashed and debugged
  from a laptop through these without involving the RP2350.
- **ESP32-C5 UART on a header or test pads** (the GP24/GP25 nets), plus EN and
  GPIO28.
- **Test points** on the PDM clock and all three PDM data lines, PPS, and the four
  SPI lines.
- **Buttons.** The usual RP2350 BOOTSEL and RUN buttons (or pads), and a user button
  on a spare pin (GP17 suggested). The two LEDs are already in the map.

## 10. Layout, RF, thermal and acoustic notes

- **GPS.** Put the M10Q and its antenna on the board edge opposite the C5 antenna,
  with a ground-plane keepout. The interferers at 1575 MHz are the C5 radio, the
  DC-DC converters and the RP2350 QSPI edges. Consider a SAW filter and LNA.
- **PDM routing.** Keep the PDM traces short, away from the C5 antenna feed, and
  guard the data lines with ground pour. Wi-Fi transmit bursts coupling into the
  mic lines is the main risk.
- **SHT45.** Slot the PCB around it and keep it away from the C5 and the DC-DC
  converters. Self-heating of 2–5 °C biases the speed-of-sound correction.
- **Mic ports.** Bottom-port mics need a drilled port, a gasket and, outdoors, an
  acoustic membrane. Test the membrane's high-frequency roll-off before committing
  the enclosure.
- **Pins during reset.** RP2350 GPIOs are high-impedance with a weak pull-down
  until firmware runs. Every net that must hold a level through reset has an
  external resistor in this document: PSRAM CS, C5 GPIO28, SPI CS, EN and the two
  C5-to-host signal lines.

## 11. Optional additions and spare pins

Seven RP2350 GPIOs are unassigned: **GP5, GP7, GP8, GP9, GP17, GP28, GP29**. In
rough order of value (assessment §6):

- **IMU and magnetometer** on I2C1. Already in firmware (LSM6 at 0x6A/0x6B, LIS2MDL
  at 0x1E). An interrupt line is optional.
- **Barometer** (BMP390 or LPS22) on I2C1.
- **Supply-voltage sense** through a divider on GP28 or GP29 (ADC inputs).
- **microSD or QSPI NAND** for local raw capture. No complete hardware-SPI pin group
  is free, so this would use a PIO SPI on the spare pins. Decide before layout if
  it is wanted.
- **Power.** PoE (802.3af) or a wide-input buck that tolerates passive PoE; USB-C
  for the bench.

## 12. Flash budget

The RP2354A's 2 MB flash holds the planar firmware comfortably, because the build
carries no radio stacks. The ESP32-C5 image is not stored on the RP2350. It is
flashed at the bench, over the C5's USB pads or through the RP2350's UART.
Over-the-air update of the C5 image through the RP2350 is future work.

---

## Consolidated pin map (RP2354A QFN-60)

| GPIO | Function | Dir | Notes |
|---|---|---|---|
| GP0 | PSRAM CS (QMI CS1n) | out | 10 kΩ pull-up; PSRAM populated (item 2) |
| GP1 / GP2 / GP3 | PDM DATA0 / 1 / 2 | in | 8.2 kΩ pull-downs (items 1, 6) |
| GP4 | PDM CLK | out | 3.072 MHz; 22 Ω series (items 1, 7) |
| GP5 | spare | | |
| GP6 | Buzzer PWM | out | Footprint only, unpopulated (item 8) |
| GP7 | spare | | |
| GP8 | spare | | PSRAM CS fallback |
| GP9 | spare | | |
| GP10 | GPS PPS | in | M10Q PPS |
| GP11 | C5 response ready | in | From C5 GPIO4; 10 kΩ pull-down |
| GP12 / GP13 | GPS UART0 TX / RX | out / in | To M10Q RX / TX |
| GP14 | C5 EN | out | 10 kΩ pull-up + RC (item 5) |
| GP15 | C5 boot strap | out | To C5 GPIO28; 10 kΩ pull-up (item 5) |
| GP16 | C5 ready for frame | in | From C5 GPIO1; 10 kΩ pull-down (item 5) |
| GP17 | spare | | |
| GP18 / GP19 | I2C1 SDA / SCL | io | 4.7 kΩ pull-ups; SHT45, IMU, magnetometer, barometer |
| GP20 | SPI0 RX | in | From C5 GPIO2 |
| GP21 | SPI0 CSn | out | To C5 GPIO10; 10 kΩ pull-up |
| GP22 | SPI0 SCK | out | To C5 GPIO6; 22 Ω series |
| GP23 | SPI0 TX | out | To C5 GPIO7 |
| GP24 / GP25 | UART1 TX / RX | out / in | To C5 GPIO12 / from C5 GPIO11 (item 3) |
| GP26 | Status LED | out | Existing FET/LED pattern |
| GP27 | I2C activity LED | out | Existing PWM-dimmed pattern |
| GP28 / GP29 | spare | | ADC-capable |

`firmware/nodes/sirith_planar/include/node_config.h` carries the same assignments.

---

## Before layout is frozen: bench the C5 link

The 2026-11-23 bench showed the Pico's Wi-Fi loss was a firmware fault, not a radio
limit. The C5 path is a new stack with the same kinds of risk, so the firmware team
will bench a Pico 2 wired to a C5 devkit before the board is committed (assessment
§3, "Bench gate"). Two results can change this document:

- the SPI clock the link actually sustains, which affects the series resistor and
  routing on GP20–GP23;
- whether the "ready for frame" line (GP16) is needed.

---

## Checklist for the board designer

- [ ] PDM on GP1–GP4 as in item 1; mic L/R select pins set to match.
- [ ] 8 MB PSRAM populated; CS on GP0 with 10 kΩ pull-up; IOVDD and QSPI_IOVDD at 3.3 V.
- [ ] C5 UART on GP24/GP25; GPS on GP12/GP13.
- [ ] 12.288 MHz dual footprint: crystal fitted, CMOS-output TCXO option, 0 Ω link.
- [ ] Crystal start-up checked across temperature; TCXO output is CMOS at 3.3 V.
- [ ] C5 SPI on GPIO6/7/2/10, confirmed against the C5 datasheet pin table.
- [ ] 10 kΩ pull-ups on C5 GPIO28, EN and SPI CS; 10 kΩ pull-downs on GP11 and GP16.
- [ ] Nothing on C5 GPIO27; no pull resistors on C5 GPIO2, 3, 7, 25, 26.
- [ ] C5 USB D−/D+ (GPIO13/14) on test pads; no SDIO routing.
- [ ] 8.2 kΩ pull-downs on GP1/GP2/GP3; RP2354A stepping read from the top marking.
- [ ] Chosen mic: 3.072 MHz in high-performance mode; data valid at 130–155 ns.
- [ ] 22 Ω series on the PDM clock (GP4) and SPI clock (GP22).
- [ ] Buzzer and driver footprints on GP6, unpopulated; piezo off the mic plane.
- [ ] SWD header; C5 UART, EN and GPIO28 on pads; test points per item 9; user button.
- [ ] GPS antenna keepout, opposite edge from the C5 antenna; SHT45 thermally isolated.
- [ ] Decision recorded on microSD and supply-voltage sense (item 11).
