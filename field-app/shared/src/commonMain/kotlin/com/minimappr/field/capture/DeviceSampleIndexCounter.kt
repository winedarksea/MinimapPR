package com.minimappr.field.capture

/**
 * Assigns device-clock sample indices to capture reads and counts gaps.
 *
 * A failed or zero-length read means the OS may have dropped audio we will
 * never see. The counter cannot know how many samples went missing, so it
 * flags the gap; the node then restarts its rolling buffer and invalidates
 * its clock model instead of silently stitching across the hole.
 */
class DeviceSampleIndexCounter {
    var nextDeviceSampleIndex: Long = 0
        private set
    var captureGapCount: Long = 0
        private set
    private var gapPendingBeforeNextBlock = false

    /** Returns the block's first index and advances by `samplesRead`. */
    fun claimIndicesForRead(samplesRead: Int): Long {
        require(samplesRead > 0) { "only successful reads claim indices" }
        val firstIndex = nextDeviceSampleIndex
        nextDeviceSampleIndex += samplesRead
        gapPendingBeforeNextBlock = false
        return firstIndex
    }

    /**
     * Records a failed read. The index jumps by one so the next block is
     * visibly discontinuous to the rolling buffer and Stage 1.
     */
    fun recordFailedRead() {
        if (!gapPendingBeforeNextBlock) {
            captureGapCount += 1
            nextDeviceSampleIndex += 1
            gapPendingBeforeNextBlock = true
        }
    }
}
