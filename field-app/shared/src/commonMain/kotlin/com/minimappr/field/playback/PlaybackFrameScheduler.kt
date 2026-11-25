package com.minimappr.field.playback

import kotlin.math.roundToLong

private const val NANOSECONDS_PER_SECOND = 1_000_000_000.0
private const val MILLISECONDS_PER_SECOND = 1_000L

/**
 * One OS audio timestamp: `framePosition` on the stream's own frame timeline
 * was presented (playback) or captured (record) at monotonic `nanoTime`
 * (CLOCK_MONOTONIC, the System.nanoTime clock family).
 */
data class FrameTimeAnchor(val framePosition: Long, val nanoTime: Long)

/** Frame ↔ monotonic-ns conversion through one anchor at the nominal rate. */
class FrameTimeline(private val sampleRateHz: Int) {
    /** Fractional frame presented/captured at `nanoTime`. */
    fun frameAtNanoTime(anchor: FrameTimeAnchor, nanoTime: Long): Double =
        anchor.framePosition + (nanoTime - anchor.nanoTime) * sampleRateHz / NANOSECONDS_PER_SECOND

    fun nanoTimeAtFrame(anchor: FrameTimeAnchor, frame: Long): Long =
        anchor.nanoTime + ((frame - anchor.framePosition) * NANOSECONDS_PER_SECOND / sampleRateHz).roundToLong()

    fun framesForMilliseconds(milliseconds: Long): Long = milliseconds * sampleRateHz / MILLISECONDS_PER_SECOND
}

sealed interface PlaybackScheduleDecision {
    /** Splice at `outputFrameIndex`; it should be presented at `predictedPresentationNs`. */
    data class Scheduled(val outputFrameIndex: Long, val predictedPresentationNs: Long) : PlaybackScheduleDecision

    /**
     * The target frame is already written (or inside the safety margin): the
     * chirp would play late, so it is reported `played = false`, never faked.
     */
    data class RejectedTooLate(val targetFrame: Long, val earliestSchedulableFrame: Long) : PlaybackScheduleDecision

    /** No valid AudioTrack timestamp yet: presentation time is unknown. */
    data object RejectedNoTimestamp : PlaybackScheduleDecision
}

/**
 * Maps a target presentation time (local monotonic ns) to an output frame
 * through the latest AudioTrack timestamp anchor. Pure and thread-confined
 * by the caller (the playback engine holds its lock around it).
 *
 * Why the anchor and not "now + latency": the anchor already contains the
 * real output latency (frames written but not yet presented), so the only
 * extra margin needed is the writer's in-flight block.
 */
class PlaybackFrameScheduler(sampleRateHz: Int, safetyMarginMilliseconds: Long) {
    private val timeline = FrameTimeline(sampleRateHz)
    private val safetyMarginFrames = timeline.framesForMilliseconds(safetyMarginMilliseconds)

    var latestAnchor: FrameTimeAnchor? = null
        private set

    fun updateAnchor(anchor: FrameTimeAnchor) {
        latestAnchor = anchor
    }

    fun decide(targetPresentationNs: Long, framesWritten: Long): PlaybackScheduleDecision {
        val anchor = latestAnchor ?: return PlaybackScheduleDecision.RejectedNoTimestamp
        val targetFrame = timeline.frameAtNanoTime(anchor, targetPresentationNs).roundToLong()
        val earliestSchedulableFrame = framesWritten + safetyMarginFrames
        if (targetFrame < earliestSchedulableFrame) {
            return PlaybackScheduleDecision.RejectedTooLate(targetFrame, earliestSchedulableFrame)
        }
        return PlaybackScheduleDecision.Scheduled(targetFrame, timeline.nanoTimeAtFrame(anchor, targetFrame))
    }

    /** Predicted presentation ns of an already-chosen frame (chirp lab), or null. */
    fun predictPresentationNs(outputFrameIndex: Long): Long? =
        latestAnchor?.let { timeline.nanoTimeAtFrame(it, outputFrameIndex) }

    /** Frames still queued ahead of the speaker at `nowNs` (output latency estimate). */
    fun queuedFramesAt(nowNs: Long, framesWritten: Long): Long? =
        latestAnchor?.let { framesWritten - timeline.frameAtNanoTime(it, nowNs).roundToLong() }
}
