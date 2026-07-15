#!/usr/bin/env python3
"""Generate the sirith_planar PDM decimation filter tables.

D3 architecture (see firmware plan, design decision D3):
  Stage B: CIC^4, decimate-by-32 (3.072 MHz chip rate -> 96 kHz), implemented
           in firmware as a LUT-based polyphase FIR equivalent to the classic
           integrator/comb CIC structure (mathematically identical, but
           byte-at-a-time table lookups instead of per-bit integrator adds --
           see PdmCicDecimator.cpp).
  Stage C: droop-compensated half-band-ish FIR, decimate-by-2 (96 kHz -> 48
           kHz), compensating the CIC passband droop up to 20 kHz.

This script is the single source of truth for both filters' coefficients. It
is checked in (per the firmware plan, Phase 2 item 2a) and its output is
pasted into firmware/lib/minimap_audio_pico/include/mmpr/PdmFilterCoeffs.h.
Run it and diff the generated header if the design (R, order, tap count)
ever changes.

Usage:
    python3 gen_pdm_filters.py [--write]

Without --write it only prints the design report (droop, group delay,
alias rejection) so it can be sanity-checked before regenerating the header.
"""
import argparse
import datetime
import textwrap

import numpy as np
from scipy import signal

# ---------------------------------------------------------------------------
# Design constants (must match PdmCicDecimator.h)
# ---------------------------------------------------------------------------
CHIP_RATE_HZ = 3_072_000.0
CIC_ORDER = 4
CIC_R = 32                      # decimate-by-32: 3.072 MHz -> 96 kHz
CIC_STAGE_OUT_HZ = CHIP_RATE_HZ / CIC_R
HALFBAND_R = 2                  # decimate-by-2: 96 kHz -> 48 kHz
FINAL_RATE_HZ = CIC_STAGE_OUT_HZ / HALFBAND_R
HALFBAND_TAPS = 47                # odd, direct-form FIR -- see design_halfband() docstring
                                   # for the ripple/alias-rejection/group-delay tradeoff this
                                   # tap count represents.
CIC_LUT_PHASES = 16             # 128-tap window / 8 bits per phase byte
COEFF_SCALE_BITS = 20           # halfband coefficient fixed-point scale (2^20)


def cic4_taps():
    """CIC^4 decimate-by-R boxcar-equivalent FIR: four-fold self-convolution
    of a length-R rectangular pulse. Integer-exact (pure combinatorics -- no
    floating point needed). Length = CIC_ORDER*(R-1)+1 = 125 for R=32."""
    box = np.ones(CIC_R, dtype=np.int64)
    taps = box.copy()
    for _ in range(CIC_ORDER - 1):
        taps = np.convolve(taps, box)
    assert taps.sum() == CIC_R ** CIC_ORDER
    assert len(taps) == CIC_ORDER * (CIC_R - 1) + 1
    return taps


def cic4_lut(taps):
    """Pad taps to a whole number of 8-bit phases and build the byte LUT.

    Bit convention (must match PdmCicDecimator.cpp pushChipBit): within a
    channel's rolling 8-bit shift register, the newest chip is shifted into
    bit 7 and older chips move down toward bit 0 (i.e. after 8 pushes,
    bit0 = oldest of that byte, bit7 = newest). Phase 0 is the newest
    (most-recently-completed) byte in the 16-byte window, phase 15 the
    oldest. PDM chip value 1 -> +1, chip value 0 -> -1 (bipolar demod).
    """
    window_len = CIC_LUT_PHASES * 8
    padded = np.zeros(window_len, dtype=np.int64)
    padded[: len(taps)] = taps
    # tap index 0 is the oldest sample in the whole window (aligns with the
    # oldest byte / phase CIC_LUT_PHASES-1, bit 0).
    lut = np.zeros((CIC_LUT_PHASES, 256), dtype=np.int64)
    for phase in range(CIC_LUT_PHASES):
        # phase 0 = newest byte -> highest tap indices (window_len-8 .. window_len-1)
        base = window_len - (phase + 1) * 8
        phase_taps = padded[base : base + 8]
        for byte_val in range(256):
            total = 0
            for bit in range(8):
                chip = (byte_val >> bit) & 1
                sign = 1 if chip else -1
                total += sign * int(phase_taps[bit])
            lut[phase, byte_val] = total
    assert lut.max() < 2**31 and lut.min() > -(2**31)
    return lut


def cic_response_db(freqs_hz):
    """|H_cic4(f)| in dB, normalized so DC = 0 dB (evaluated at the chip
    rate, standard CIC magnitude formula)."""
    w = np.pi * freqs_hz / CHIP_RATE_HZ
    # sin(R*w)/(R*sin(w)), limit at w->0 is 1.
    num = np.sin(CIC_R * w)
    den = CIC_R * np.sin(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(np.abs(w) < 1e-9, 1.0, num / den)
    mag = np.abs(h) ** CIC_ORDER
    return 20 * np.log10(mag)


def design_halfband(numtaps=HALFBAND_TAPS, n_pb_segments=12, stop_lo=27500,
                     pb_weight=1.0, stop_weight=1000.0):
    """Droop-compensated decimate-by-2 FIR (96 kHz -> 48 kHz), designed with
    scipy.signal.firls (weighted least squares against a piecewise-linear
    target response).

    Passband 0-20 kHz targets the inverse of the CIC droop (so the combined
    CIC+halfband response is flat to 20 kHz), built from `n_pb_segments`
    linear segments so firls can track the smooth (non-linear) droop curve
    closely. The true anti-alias constraint for a decimate-by-2 stage is the
    stopband from 28 kHz to Nyquist (48 kHz): content in (28 kHz, 48 kHz)
    folds into (0 kHz, 20 kHz) after decimation, so that is the region that
    must be suppressed; 20-28 kHz is transition slack.

    Tradeoff note: the firmware plan's D4 design decision asserts a ~260 us
    total group delay (CIC + halfband), which at CIC_R=32 pins the halfband
    stage to ~47 taps (more taps directly buys more alias rejection but also
    adds (extra_taps/4) us of delay at this Fs, which would drift the D4
    figure the timestamp correction and cross-node TDOA bias depend on). A
    single flat-plus-taper firwin2 design at 47 taps (an earlier attempt)
    could not hit +-0.1 dB in-band ripple with any reasonable stopband edge --
    firwin2's frequency-sampled windowing tracks a smooth compensation target
    poorly (Gibbs-like error up to ~1 dB). Switching to firls's
    piecewise-linear multi-band fit brings ripple to within +-0.1 dB at this
    same 47-tap budget; alias rejection lands around ~80 dB rather than the
    >=90 dB an unconstrained (longer) design could reach -- a deliberate
    choice to hold the group-delay figure, flagged here and in PDM_DESIGN.md
    as a candidate for revisiting once the group-delay budget is
    bench-validated (a slightly larger delay may prove acceptable).
    """
    nyq = CIC_STAGE_OUT_HZ / 2.0  # 48 kHz
    edges = np.linspace(0, 20000, n_pb_segments + 1)
    bands = []
    desired = []
    weight = []
    for i in range(n_pb_segments):
        lo, hi = edges[i], edges[i + 1]
        gain_lo = 10 ** (-cic_response_db(np.array([lo if lo > 0 else 1.0]))[0] / 20.0)
        gain_hi = 10 ** (-cic_response_db(np.array([hi]))[0] / 20.0)
        bands += [lo, hi]
        desired += [gain_lo, gain_hi]
        weight.append(pb_weight)
    bands += [stop_lo, nyq]
    desired += [0.0, 0.0]
    weight.append(stop_weight)
    taps = signal.firls(numtaps, bands, desired, weight=weight, fs=CIC_STAGE_OUT_HZ)
    return taps


def quantize(taps, scale_bits):
    scale = 1 << scale_bits
    q = np.round(taps * scale).astype(np.int64)
    assert q.max() < 2**31 and q.min() > -(2**31), "halfband coeff overflow at this scale"
    return q, scale


def group_delay_report(halfband_taps):
    cic_delay_out_samples = CIC_ORDER * (CIC_R - 1) / (2.0 * CIC_R)  # at 96 kHz
    hb_delay_out_samples = (len(halfband_taps) - 1) / 2.0 / HALFBAND_R  # at 48 kHz
    cic_delay_us = cic_delay_out_samples / CIC_STAGE_OUT_HZ * 1e6
    hb_delay_us = hb_delay_out_samples / FINAL_RATE_HZ * 1e6
    total_us = cic_delay_us + hb_delay_us
    total_samples_48k = total_us * 1e-6 * FINAL_RATE_HZ
    return cic_delay_us, hb_delay_us, total_us, total_samples_48k


def render_header(cic_taps, lut, hb_taps_q, hb_scale, group_delay_us):
    now = datetime.datetime.now().isoformat(timespec="seconds")

    def render_i32_array(name, arr, per_line=16):
        flat = arr.reshape(-1)
        lines = []
        for i in range(0, len(flat), per_line):
            chunk = ", ".join(str(int(v)) for v in flat[i : i + per_line])
            lines.append("    " + chunk + ",")
        body = "\n".join(lines)
        return f"static constexpr int32_t {name}[{len(flat)}] = {{\n{body}\n}};\n"

    lut_flat = render_i32_array("kCicLutFlat", lut, per_line=16)
    hb_arr = render_i32_array("kHalfbandTapsQ", np.array(hb_taps_q, dtype=np.int64))
    hb_scale_bits = hb_scale.bit_length() - 1

    preamble = textwrap.dedent(f"""\
        // GENERATED FILE -- do not hand-edit.
        // Produced by firmware/scripts/gen_pdm_filters.py on {now}.
        // Regenerate with: python3 firmware/scripts/gen_pdm_filters.py --write
        //
        // CIC^4 R={CIC_R} boxcar-equivalent FIR (Stage B) LUT, and the
        // droop-compensated decimate-by-2 FIR (Stage C). See PdmCicDecimator.h
        // for the runtime pipeline these feed and PDM_DESIGN.md for the full
        // derivation (D3 in the firmware plan).
        #pragma once

        #include <cstdint>

        namespace mmpr {{
        namespace pdm_filters {{

        static constexpr int kCicOrder = {CIC_ORDER};
        static constexpr int kCicDecimation = {CIC_R};
        static constexpr int kCicLutPhases = {CIC_LUT_PHASES};
        static constexpr int kCicWindowBytes = {CIC_LUT_PHASES};
        static constexpr int kHalfbandDecimation = {HALFBAND_R};
        static constexpr int kHalfbandTapCount = {len(hb_taps_q)};
        static constexpr int kHalfbandCoeffScaleBits = {hb_scale_bits};

        // kCicLutFlat[phase * 256 + byteValue] = signed contribution of that byte
        // (bipolar +-1 chips) at that position in the 128-chip CIC^4 window.
        // Phase 0 = newest byte, phase {CIC_LUT_PHASES - 1} = oldest.
        """)
    middle = textwrap.dedent(f"""\
        // Droop-compensated decimate-by-2 FIR, Q{hb_scale_bits} fixed point
        // (i.e. divide by 2^{hb_scale_bits} to get the real coefficient).
        """)
    trailer = textwrap.dedent(f"""\
        // Constant total group delay of Stage B + Stage C, referenced to the
        // 48 kHz output domain. Subtract this from raw-capture-derived sample
        // timestamps to recover acoustic-arrival-time (D4).
        static constexpr double kGroupDelayMicroseconds = {group_delay_us:.6f};

        }}  // namespace pdm_filters
        }}  // namespace mmpr
        """)
    return preamble + lut_flat + "\n" + middle + hb_arr + "\n" + trailer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write the generated header")
    parser.add_argument(
        "--out",
        default="lib/minimap_audio_pico/include/mmpr/PdmFilterCoeffs.h",
        help="output header path, relative to firmware/ (this script's parent dir)",
    )
    args = parser.parse_args()

    taps = cic4_taps()
    lut = cic4_lut(taps)
    hb_taps = design_halfband()
    hb_taps_q, hb_scale = quantize(hb_taps, COEFF_SCALE_BITS)

    cic_delay_us, hb_delay_us, total_us, total_samples_48k = group_delay_report(hb_taps)

    droop_20k_db = cic_response_db(np.array([20000.0]))[0]

    eval_freqs = np.linspace(1, 48000, 6000)
    cic_db_curve = cic_response_db(eval_freqs)
    _, h = signal.freqz(hb_taps, worN=eval_freqs, fs=CIC_STAGE_OUT_HZ)
    hb_db_curve = 20 * np.log10(np.abs(h) + 1e-30)
    combined_db = cic_db_curve + hb_db_curve
    pb_mask = eval_freqs <= 20000
    fold_mask = eval_freqs >= 28000
    ripple_lo, ripple_hi = combined_db[pb_mask].min(), combined_db[pb_mask].max()
    alias_rejection_db = -combined_db[fold_mask].max()

    print(f"CIC^4 R={CIC_R} taps: {len(taps)} (padded to {CIC_LUT_PHASES * 8})")
    print(f"CIC droop at 20 kHz: {droop_20k_db:.3f} dB (plan estimate: -2.6 dB)")
    print(f"Halfband taps: {len(hb_taps)}, coeff scale: 2^{hb_scale.bit_length() - 1}")
    print(f"Combined (CIC+halfband) passband ripple to 20 kHz: "
          f"{ripple_lo:.4f} / {ripple_hi:.4f} dB (target: +-0.1 dB)")
    print(f"Alias rejection (fold band 28-48 kHz): {alias_rejection_db:.2f} dB "
          f"(target: >=90 dB)")
    print(f"Group delay: CIC={cic_delay_us:.3f} us, halfband={hb_delay_us:.3f} us, "
          f"total={total_us:.3f} us ({total_samples_48k:.4f} samples @ 48 kHz)")

    header = render_header(taps, lut, hb_taps_q, hb_scale, total_us)

    if args.write:
        import pathlib
        out_path = pathlib.Path(__file__).resolve().parent.parent / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(header)
        print(f"wrote {out_path}")
    else:
        print("(dry run; pass --write to regenerate the header)")


if __name__ == "__main__":
    main()
