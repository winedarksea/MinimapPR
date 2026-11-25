package com.minimappr.field.ffi

import com.minimappr.field.capture.pcm16LittleEndianToShortArray
import com.minimappr.field.capture.toPcm16LittleEndianBytes
import com.minimappr.field.session.FieldSessionIdentity
import com.minimappr.field.testsupport.deterministicNoise
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/** Exercises the real Rust library (FFI surface v3) through the Gobley bindings. */
class MinimapprFfiBindingsTest {
    private val joinRequestKindId = 1
    private val oneSecondAt48KilohertzSamples = 48_000
    private val expectedSurfaceVersion = 3u
    private val chirpSpec = FieldChirpWaveformSpec(FieldChirpWaveformKind.UP_DOWN_HYPERBOLIC, 13_000.0, 16_000.0, 0.05)
    private val chirpFrames = 2_400

    @Test
    fun surfaceVersionMatchesTheShellExpectation() {
        assertEquals(expectedSurfaceVersion, fieldFfiSurfaceVersion())
        assertTrue(knownMessageKindIds().contains(joinRequestKindId.toUShort()))
    }

    @Test
    fun stage1DetectorStaysClosedOnSilence() {
        FieldStage1Detector(FieldStage1Mode.BIRDS).use { detector ->
            detector.processPcm16Block(0uL, ShortArray(oneSecondAt48KilohertzSamples).toPcm16LittleEndianBytes())
            assertTrue(detector.counters().framesEvaluated > 0uL)
            assertEquals(0uL, detector.counters().triggersOpened)
        }
    }

    @Test
    fun clipEncodeDecodeKeepsTheDeviceClockAnchor() {
        val tonePeriodSamples = 48
        val toneAmplitude = 8_000.0
        val rollingBuffer = ShortArray(oneSecondAt48KilohertzSamples) { index ->
            (toneAmplitude * sin(2 * PI * (index % tonePeriodSamples) / tonePeriodSamples)).toInt().toShort()
        }
        val windowFirstSampleIndex = 12_000uL
        val clip = encodeClipFromRollingBuffer(
            rollingBuffer.toPcm16LittleEndianBytes(), 0uL, windowFirstSampleIndex, 24_000u,
            FieldDecimationFilter.BIRD_BAND_WITH_RUMBLE_CUT,
        )
        val decoded = decodeClip(clip)
        assertEquals(windowFirstSampleIndex, decoded.firstSampleIndexDeviceClock)
        assertEquals(24_000u, decoded.outputSampleRateHz)
        assertEquals(12_000, decoded.pcm16LittleEndianBytesOutputRate.pcm16LittleEndianToShortArray().size)
    }

    @Test
    fun malformedFrameAndEnvelopeRaiseTheTypedError() {
        assertFailsWith<FieldFfiException.ProtocolFrame> { describeFrame(byteArrayOf()) }
        assertFailsWith<FieldFfiException.ProtocolFrame> { readLanEnvelopeRouting(byteArrayOf(1, 2, 3)) }
    }

    @Test
    fun chirpPlaybackBufferIsInterleavedStereoWithASilentUnusedChannel() {
        assertEquals(chirpFrames.toUInt(), chirpFrameCount(chirpSpec))
        val stereo = chirpPlaybackPcm16(chirpSpec, FieldStereoChannel.LEFT, -6.0).pcm16LittleEndianToShortArray()
        assertEquals(chirpFrames * 2, stereo.size)
        assertTrue((0 until chirpFrames).all { stereo[it * 2 + 1] == 0.toShort() }, "right channel silent")
        assertTrue((0 until chirpFrames).any { stereo[it * 2] != 0.toShort() }, "left channel carries the chirp")
    }

    @Test
    fun detectorRecoversAChirpInsertedFromThePlaybackBuffer() {
        val stereo = chirpPlaybackPcm16(chirpSpec, FieldStereoChannel.LEFT, -6.0).pcm16LittleEndianToShortArray()
        val insertAt = 3_217
        val recording = deterministicNoise(12_000, amplitudeLsb = 30)
        for (frame in 0 until chirpFrames) recording[insertAt + frame] = (recording[insertAt + frame] + stereo[frame * 2]).toShort()
        val detection = FieldChirpDetector(chirpSpec, speedOfSoundForTemperature(20.0))
            .detect(recording.toPcm16LittleEndianBytes(), 0u, 8_000u, false, null)
        assertNotNull(detection)
        assertTrue(abs(detection.sampleIndex - insertAt) < 0.5, "picked ${detection.sampleIndex}")
    }

    @Test
    fun defaultTuningIsPeriodicAndSoundSpeedFollowsTemperature() {
        val tuning = defaultFieldMeshTuning()
        assertEquals(FieldChirpScheduleMode.PERIODIC, tuning.scheduleMode)
        assertTrue(abs(speedOfSoundForTemperature(20.0) - 343.2) < 0.2)
    }

    @Test
    fun invalidDeviceSetupsAreTypedErrorsNotAllocations() {
        val host = FieldSessionIdentity.newHost()
        assertFailsWith<FieldFfiException.InvalidArgument> {
            FieldMeshDevice(host.deviceSetup("jvm-test", "0", rollingBufferSeconds = 0), defaultFieldMeshTuning(), 0L)
        }
        val hostWithoutSession = host.copy(sessionId = ByteArray(16))
        assertFailsWith<FieldFfiException.InvalidArgument> {
            FieldMeshDevice(hostWithoutSession.deviceSetup("jvm-test", "0", rollingBufferSeconds = 2), defaultFieldMeshTuning(), 0L)
        }
    }

    @Test
    fun aJoinerSendsAHelloOnceItsLinkComesUp() {
        val joiner = FieldSessionIdentity.newJoiner(null)
        FieldMeshDevice(joiner.deviceSetup("jvm-test", "0", rollingBufferSeconds = 2), defaultFieldMeshTuning(), 0L).use { device ->
            assertTrue(device.tick(1_000_000L).none { it is FieldDeviceAction.SendEnvelope }, "no link, no Hello")
            device.setTransportLinkUp(2_000_000L, true)
            val hello = device.tick(3_000_000L).filterIsInstance<FieldDeviceAction.SendEnvelope>().single()
            val routing = readLanEnvelopeRouting(hello.envelope)
            assertTrue(routing.isHello)
            assertEquals(joiner.nodeId, routing.senderNodeId)
            assertEquals(null, routing.recipientNodeId)
            assertEquals(FieldDeviceRole.JOINER, device.syncStatus().role)
        }
    }
}
