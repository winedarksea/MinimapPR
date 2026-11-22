package com.minimappr.field.capture

/**
 * One contiguous block of mono PCM16 on the device audio clock.
 *
 * `firstDeviceSampleIndex` comes from counting samples read since capture
 * start, never from wall-clock time: TDOA timing is sample-count based.
 */
class CaptureBlock(
    val firstDeviceSampleIndex: Long,
    val samples: ShortArray,
) {
    val nextDeviceSampleIndex: Long get() = firstDeviceSampleIndex + samples.size
}

/** Receives capture blocks on the capture thread; must not block for long. */
fun interface CaptureBlockSink {
    fun onCaptureBlock(block: CaptureBlock)
}
