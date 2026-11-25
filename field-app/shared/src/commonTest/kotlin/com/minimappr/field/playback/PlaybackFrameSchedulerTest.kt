package com.minimappr.field.playback

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue

class PlaybackFrameSchedulerTest {
    private val sampleRateHz = 48_000
    private val safetyMarginMilliseconds = 20L
    private val oneSecondNs = 1_000_000_000L

    @Test
    fun noTimestampMeansNoSchedulingRatherThanAGuess() {
        val scheduler = PlaybackFrameScheduler(sampleRateHz, safetyMarginMilliseconds)
        assertEquals(PlaybackScheduleDecision.RejectedNoTimestamp, scheduler.decide(5 * oneSecondNs, 0))
        assertNull(scheduler.predictPresentationNs(100))
    }

    @Test
    fun targetMapsThroughTheAnchorAtTheNominalRate() {
        val scheduler = PlaybackFrameScheduler(sampleRateHz, safetyMarginMilliseconds)
        // Frame 48 000 was presented at t = 10 s; 9 600 frames (200 ms) are written.
        scheduler.updateAnchor(FrameTimeAnchor(framePosition = 48_000, nanoTime = 10 * oneSecondNs))
        val decision = scheduler.decide(targetPresentationNs = 11 * oneSecondNs, framesWritten = 57_600)
        assertIs<PlaybackScheduleDecision.Scheduled>(decision)
        assertEquals(96_000, decision.outputFrameIndex)
        assertEquals(11 * oneSecondNs, decision.predictedPresentationNs)
    }

    @Test
    fun predictedPresentationIsQuantisedToTheChosenFrame() {
        val scheduler = PlaybackFrameScheduler(sampleRateHz, safetyMarginMilliseconds)
        scheduler.updateAnchor(FrameTimeAnchor(0, 0))
        val decision = scheduler.decide(targetPresentationNs = oneSecondNs + 7_000, framesWritten = 0)
        assertIs<PlaybackScheduleDecision.Scheduled>(decision)
        // 7 µs is a third of a frame: rounds to frame 48 000, presented at exactly 1 s.
        assertEquals(48_000, decision.outputFrameIndex)
        assertEquals(oneSecondNs, decision.predictedPresentationNs)
    }

    @Test
    fun targetsInsideTheWrittenAudioOrTheMarginAreRejectedAsTooLate() {
        val scheduler = PlaybackFrameScheduler(sampleRateHz, safetyMarginMilliseconds)
        scheduler.updateAnchor(FrameTimeAnchor(48_000, 10 * oneSecondNs))
        // 100 ms of audio queued past the anchor: a target 50 ms out is already written.
        val framesWritten = 48_000L + 4_800
        val alreadyWritten = scheduler.decide(10 * oneSecondNs + 50_000_000, framesWritten)
        assertIs<PlaybackScheduleDecision.RejectedTooLate>(alreadyWritten)
        // 110 ms out is past the written audio but inside the 20 ms margin.
        val insideMargin = scheduler.decide(10 * oneSecondNs + 110_000_000, framesWritten)
        assertIs<PlaybackScheduleDecision.RejectedTooLate>(insideMargin)
        assertEquals(framesWritten + 960, insideMargin.earliestSchedulableFrame)
        assertIs<PlaybackScheduleDecision.Scheduled>(scheduler.decide(10 * oneSecondNs + 130_000_000, framesWritten))
    }

    @Test
    fun queuedFramesEstimateTheOutputLatency() {
        val scheduler = PlaybackFrameScheduler(sampleRateHz, safetyMarginMilliseconds)
        scheduler.updateAnchor(FrameTimeAnchor(1_000, oneSecondNs))
        val queued = scheduler.queuedFramesAt(nowNs = oneSecondNs, framesWritten = 1_000 + 2_400)
        assertEquals(2_400, queued)
        assertTrue(FrameTimeline(sampleRateHz).framesForMilliseconds(50) == 2_400L)
    }
}
