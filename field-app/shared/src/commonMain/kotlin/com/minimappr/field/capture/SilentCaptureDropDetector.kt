package com.minimappr.field.capture

import com.minimappr.field.playback.FrameTimeAnchor

/**
 * Finds capture audio lost without a failed read. Android can deliver fewer
 * frames than real time with every read succeeding (Pixel 10 Pro XL,
 * 2026-11-26: 527 ms gone in one audio-server stall, `capture_gaps` 0).
 * `AudioRecord.getTimestamp` still shows it: between two timestamps the
 * monotonic clock advances further than the frame count explains.
 *
 * The threshold sits above one HAL burst of timestamp jitter (the Pixel 10's
 * timestamps wander by 256 frames = 5.33 ms); smaller drops are left to the
 * chirp-derived θ silent-gap check in minimappr-core.
 */
class SilentCaptureDropDetector(
    private val sampleRateHz: Int,
    private val stepThresholdNanoseconds: Long,
) {
    private var previous: FrameTimeAnchor? = null

    /** A gap report when audio went missing between the previous timestamp and this one. */
    fun observe(anchor: FrameTimeAnchor): CaptureGapReport? {
        val last = previous
        previous = anchor
        if (last == null || anchor.framePosition < last.framePosition) return null
        val elapsedNanoseconds = anchor.nanoTime - last.nanoTime
        val framesAdvanced = anchor.framePosition - last.framePosition
        val expectedFrames = elapsedNanoseconds * sampleRateHz / NANOSECONDS_PER_SECOND
        val missingFrames = expectedFrames - framesAdvanced
        if (missingFrames * NANOSECONDS_PER_SECOND / sampleRateHz <= stepThresholdNanoseconds) return null
        return CaptureGapReport(expectedFrame = last.framePosition + expectedFrames, gotFrame = anchor.framePosition, missingFrames = missingFrames)
    }

    private companion object {
        const val NANOSECONDS_PER_SECOND = 1_000_000_000L
    }
}
