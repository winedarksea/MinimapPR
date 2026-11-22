package com.minimappr.field.ffi

import com.minimappr.field.FieldNodeConfiguration
import com.minimappr.field.capture.CaptureBlock
import com.minimappr.field.capture.pcm16LittleEndianToShortArray
import com.minimappr.field.capture.toPcm16LittleEndianBytes
import com.minimappr.field.node.FieldNodeRuntime
import com.minimappr.field.node.FieldOutboundFrameTransport
import com.minimappr.field.node.skeletonSoloNodeSetup
import kotlin.math.PI
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** Exercises the real Rust library through the Gobley bindings. */
class MinimapprFfiBindingsTest {
    private val joinRequestKindId = 1
    private val captureBlockSamples = 960
    private val oneSecondAt48KilohertzSamples = 48_000

    @Test
    fun surfaceVersionMatchesTheShellExpectation() {
        assertEquals(1u, fieldFfiSurfaceVersion())
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
        val windowSampleCount = 24_000u
        val clip = encodeClipFromRollingBuffer(
            rollingBuffer.toPcm16LittleEndianBytes(),
            0uL,
            windowFirstSampleIndex,
            windowSampleCount,
            FieldDecimationFilter.BIRD_BAND_WITH_RUMBLE_CUT,
        )
        val decoded = decodeClip(clip)
        assertEquals(windowFirstSampleIndex, decoded.firstSampleIndexDeviceClock)
        assertEquals(24_000u, decoded.outputSampleRateHz)
        assertEquals(12_000, decoded.pcm16LittleEndianBytesOutputRate.pcm16LittleEndianToShortArray().size)
    }

    @Test
    fun malformedFrameRaisesTheTypedError() {
        assertFailsWith<FieldFfiException.ProtocolFrame> { describeFrame(byteArrayOf()) }
    }

    @Test
    fun runtimeFirstTickSendsAJoinRequestThroughTheTransport() {
        val sentKinds = mutableListOf<Int>()
        val transport = FieldOutboundFrameTransport { _, frame, _ ->
            sentKinds += describeFrame(frame).messageKindId.toInt()
        }
        val nodeConfiguration = FieldNodeConfiguration(rollingBufferSeconds = 2)
        FieldNodeRuntime(skeletonSoloNodeSetup(nodeConfiguration, "jvm-test", "0"), nodeConfiguration, transport).use { runtime ->
            runtime.onCaptureBlock(CaptureBlock(0, ShortArray(captureBlockSamples)))
            runtime.tick()
            assertEquals(listOf(joinRequestKindId), sentKinds)
            assertEquals(captureBlockSamples.toLong(), runtime.status.value.capturedSampleCount)
            assertEquals(1, runtime.status.value.outboundFramesSent)
        }
    }
}
