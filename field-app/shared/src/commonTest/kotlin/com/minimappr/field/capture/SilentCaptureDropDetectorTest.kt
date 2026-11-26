package com.minimappr.field.capture

import com.minimappr.field.playback.FrameTimeAnchor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class SilentCaptureDropDetectorTest {
    private val detector = SilentCaptureDropDetector(sampleRateHz = 48_000, stepThresholdNanoseconds = 15_000_000)

    @Test
    fun theTimestampStepOfASilentDropIsReportedWithItsSize() {
        // Pixel 10 Pro XL, 2026-11-26: 1.0 s of monotonic time, 22 848 frames.
        assertNull(detector.observe(FrameTimeAnchor(framePosition = 2_215_872, nanoTime = 134_989_044_754_055)))
        val report = assertNotNull(detector.observe(FrameTimeAnchor(framePosition = 2_238_720, nanoTime = 134_990_047_122_103)))
        assertEquals(2_238_720, report.gotFrame)
        assertEquals(25_265, report.missingFrames)
    }

    @Test
    fun oneHalBurstOfTimestampWanderIsNotADrop() {
        assertNull(detector.observe(FrameTimeAnchor(0, 1_000_000_000)))
        assertNull(detector.observe(FrameTimeAnchor(48_000, 2_005_333_333)))
        assertNull(detector.observe(FrameTimeAnchor(96_000, 3_000_000_000)))
    }

    @Test
    fun aCaptureRestartIsNotADrop() {
        assertNull(detector.observe(FrameTimeAnchor(480_000, 10_000_000_000)))
        assertNull(detector.observe(FrameTimeAnchor(960, 11_000_000_000)))
    }
}
