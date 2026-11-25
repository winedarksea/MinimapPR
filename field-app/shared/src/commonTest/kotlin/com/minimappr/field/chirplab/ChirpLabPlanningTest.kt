package com.minimappr.field.chirplab

import com.minimappr.field.capture.CaptureBlock
import com.minimappr.field.capture.CaptureRingBuffer
import com.minimappr.field.capture.pcm16LittleEndianToShortArray
import com.minimappr.field.ffi.FieldChirpDetector
import com.minimappr.field.ffi.FieldChirpWaveformKind
import com.minimappr.field.ffi.FieldChirpWaveformSpec
import com.minimappr.field.ffi.FieldStereoChannel
import com.minimappr.field.ffi.chirpPlaybackPcm16
import com.minimappr.field.playback.FrameTimeAnchor
import com.minimappr.field.session.FieldSessionIdentity
import com.minimappr.field.session.parseSessionIdHex
import com.minimappr.field.testsupport.deterministicNoise
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ChirpLabPlanningTest {
    private val periodFrames = 192_000L // 4 s

    @Test
    fun roleBIsHalfAPeriodAfterRoleAOnItsOwnTimeline() {
        val roleA = ChirpLabEmissionPlanner(periodFrames, (periodFrames * ChirpLabRole.A.periodOffsetFraction).toLong())
        val roleB = ChirpLabEmissionPlanner(periodFrames, (periodFrames * ChirpLabRole.B.periodOffsetFraction).toLong())
        assertEquals(3 * periodFrames, roleA.outputFrameFor(3))
        assertEquals(3 * periodFrames + periodFrames / 2, roleB.outputFrameFor(3))
    }

    @Test
    fun firstEmissionIsTheFirstPeriodSlotPastTheLead() {
        val planner = ChirpLabEmissionPlanner(periodFrames, 96_000)
        assertEquals(PlannedChirpLabEmission(0, 96_000), planner.firstEmissionAtOrAfter(14_400))
        assertEquals(PlannedChirpLabEmission(1, 288_000), planner.firstEmissionAtOrAfter(96_001))
        assertEquals(PlannedChirpLabEmission(2, 480_000), planner.next(PlannedChirpLabEmission(1, 288_000)))
        assertFailsWith<IllegalArgumentException> { ChirpLabEmissionPlanner(periodFrames, periodFrames) }
    }

    @Test
    fun ringBufferKeepsTheDeviceTimelineAcrossGapsAndForgetsOldAudio() {
        val ring = CaptureRingBuffer(capacitySamples = 10)
        ring.append(CaptureBlock(0, shortArrayOf(1, 2, 3)))
        ring.append(CaptureBlock(5, shortArrayOf(6, 7)))
        assertContentEquals(shortArrayOf(2, 3, 0, 0, 6), ring.copyWindow(1, 6))
        ring.append(CaptureBlock(7, ShortArray(8) { 9 }))
        assertNull(ring.copyWindow(1, 6), "evicted")
        assertNull(ring.copyWindow(10, 16), "not captured yet")
    }

    @Test
    fun selfChirpMonitorFindsTheChirpAtThePredictedFrame() {
        val spec = FieldChirpWaveformSpec(FieldChirpWaveformKind.UP_DOWN_HYPERBOLIC, 13_000.0, 16_000.0, 0.05)
        val stereo = chirpPlaybackPcm16(spec, FieldStereoChannel.RIGHT, -6.0).pcm16LittleEndianToShortArray()
        val readouts = mutableListOf<SelfChirpReadout>()
        val monitor = SelfChirpMonitor(FieldChirpDetector(spec, 343.0), 48_000, 2, 100, { it() }) { readouts += it }
        assertFalse(monitor.expectEmission(0, 0), "no capture anchor yet")
        monitor.onCaptureAnchor(FrameTimeAnchor(framePosition = 0, nanoTime = 1_000_000_000))
        // Predicted at 1.5 s after the anchor = frame 24 000; the real chirp lands 100 frames late.
        assertTrue(monitor.expectEmission(7, 1_500_000_000))
        val capture = deterministicNoise(48_000, amplitudeLsb = 30)
        for (frame in 0 until 2_400) capture[24_100 + frame] = (capture[24_100 + frame] + stereo[frame * 2 + 1]).toShort()
        for (block in 0 until 50) monitor.onCaptureBlock(CaptureBlock(block * 960L, capture.copyOfRange(block * 960, block * 960 + 960)))
        val readout = readouts.single()
        assertEquals(7, readout.emissionIndex)
        val detected = assertNotNull(readout.detectedCaptureFrame)
        assertTrue(abs(detected - 24_100) < 0.5, "detected at $detected")
        assertTrue(readout.detection!!.peakFraction in 0.4f..0.6f)
    }

    @Test
    fun sessionIdentitiesAreRandomAndJoinersAdoptOrExpectASessionId() {
        val first = FieldSessionIdentity.newHost()
        val second = FieldSessionIdentity.newHost()
        assertNotEquals(first.nodeId, second.nodeId)
        assertFalse(first.sessionId.contentEquals(second.sessionId))
        assertContentEquals(first.sessionId, parseSessionIdHex(first.sessionIdHex))
        assertNull(parseSessionIdHex("abc"))
        assertTrue(FieldSessionIdentity.newJoiner(null).sessionId.all { it == 0.toByte() })
        assertContentEquals(first.sessionId, FieldSessionIdentity.newJoiner(first.sessionId).sessionId)
    }
}
