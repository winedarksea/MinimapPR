package com.minimappr.field.testsupport

/**
 * Low-level xorshift noise so matched-filter noise floors are never exactly
 * zero (the detector's SNR is relative to the floor).
 */
fun deterministicNoise(length: Int, amplitudeLsb: Int, seed: Long = 0x2545F4914F6CDD1DL): ShortArray {
    var state = seed
    return ShortArray(length) {
        state = state xor (state shl 13)
        state = state xor (state ushr 7)
        state = state xor (state shl 17)
        ((state ushr 40) % (2 * amplitudeLsb + 1) - amplitudeLsb).toInt().toShort()
    }
}
