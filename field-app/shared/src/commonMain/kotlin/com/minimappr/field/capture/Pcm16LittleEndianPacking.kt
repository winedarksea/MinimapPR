package com.minimappr.field.capture

private const val PCM16_BYTES_PER_SAMPLE = 2
private const val LOW_BYTE_MASK = 0xFF
private const val BITS_PER_BYTE = 8

/**
 * Packs PCM16 for the FFI. minimappr-ffi takes PCM as little-endian bytes
 * because UniFFI would box a `Vec<i16>` into `List<Short>` per sample.
 */
fun ShortArray.toPcm16LittleEndianBytes(sampleCount: Int = size): ByteArray {
    val packed = ByteArray(sampleCount * PCM16_BYTES_PER_SAMPLE)
    for (sampleIndex in 0 until sampleCount) {
        val sample = this[sampleIndex].toInt()
        packed[sampleIndex * PCM16_BYTES_PER_SAMPLE] = (sample and LOW_BYTE_MASK).toByte()
        packed[sampleIndex * PCM16_BYTES_PER_SAMPLE + 1] = ((sample shr BITS_PER_BYTE) and LOW_BYTE_MASK).toByte()
    }
    return packed
}

fun ByteArray.pcm16LittleEndianToShortArray(): ShortArray {
    require(size % PCM16_BYTES_PER_SAMPLE == 0) { "PCM16 byte count $size is odd" }
    return ShortArray(size / PCM16_BYTES_PER_SAMPLE) { sampleIndex ->
        val lowByte = this[sampleIndex * PCM16_BYTES_PER_SAMPLE].toInt() and LOW_BYTE_MASK
        val highByte = this[sampleIndex * PCM16_BYTES_PER_SAMPLE + 1].toInt()
        ((highByte shl BITS_PER_BYTE) or lowByte).toShort()
    }
}
