package com.minimappr.field.capture

/**
 * Last N mono samples keyed by device sample index. Written by the capture
 * thread only; [copyWindow] runs on the same thread (the self-chirp monitor
 * copies windows inside its capture callback), so no locking.
 */
class CaptureRingBuffer(private val capacitySamples: Int) {
    private val samples = ShortArray(capacitySamples)

    /** Index one past the newest sample held. */
    var endDeviceSampleIndex: Long = 0
        private set

    val firstHeldDeviceSampleIndex: Long get() = maxOf(0L, endDeviceSampleIndex - capacitySamples)

    fun append(block: CaptureBlock) {
        // A jump (failed read) leaves zeros in between, as the WAV writer does,
        // so window indices stay on the device timeline.
        var index = endDeviceSampleIndex
        while (index < block.firstDeviceSampleIndex) {
            samples[(index % capacitySamples).toInt()] = 0
            index += 1
        }
        for (offset in block.samples.indices) {
            samples[((block.firstDeviceSampleIndex + offset) % capacitySamples).toInt()] = block.samples[offset]
        }
        endDeviceSampleIndex = block.nextDeviceSampleIndex
    }

    /** Copy `[firstIndex, endIndex)`, or null when any part is not held. */
    fun copyWindow(firstIndex: Long, endIndex: Long): ShortArray? {
        if (firstIndex < firstHeldDeviceSampleIndex || endIndex > endDeviceSampleIndex || endIndex <= firstIndex) return null
        return ShortArray((endIndex - firstIndex).toInt()) { offset ->
            samples[((firstIndex + offset) % capacitySamples).toInt()]
        }
    }
}
