package com.minimappr.field.capture

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CaptureBackpressureTest {
    @Test
    fun gateDropsAndCountsWhileAWindowIsInFlight() {
        val gate = DropWhileBusyBackpressureGate()
        assertTrue(gate.tryAcquire())
        assertFalse(gate.tryAcquire())
        assertFalse(gate.tryAcquire())
        gate.release()
        assertTrue(gate.tryAcquire())
        gate.release()
        assertEquals(2, gate.acceptedWindowCount)
        assertEquals(2, gate.droppedWindowCount)
    }

    @Test
    fun releaseWithoutAcquireIsAProgrammingError() {
        assertFailsWith<IllegalStateException> { DropWhileBusyBackpressureGate().release() }
    }

    @Test
    fun sampleCounterAdvancesByReadsAndMarksOneGapPerFailureRun() {
        val counter = DeviceSampleIndexCounter()
        assertEquals(0, counter.claimIndicesForRead(960))
        assertEquals(960, counter.claimIndicesForRead(960))
        counter.recordFailedRead()
        counter.recordFailedRead()
        assertEquals(1, counter.captureGapCount)
        // The jump makes the next block discontinuous for the rolling buffer.
        assertEquals(1_921, counter.claimIndicesForRead(960))
    }

    @Test
    fun readResultsDeliverFlagOrStopWithoutSpinningOnErrors() {
        assertEquals(CaptureReadOutcome.DELIVER_BLOCK, classifyCaptureReadResult(960))
        assertEquals(CaptureReadOutcome.FLAG_GAP_AND_CONTINUE, classifyCaptureReadResult(0))
        // AudioRecord.ERROR_INVALID_OPERATION (-3) and ERROR_DEAD_OBJECT (-6)
        // never clear on retry: the loop must stop, not spin.
        assertEquals(CaptureReadOutcome.STOP_CAPTURE_WITH_FAILURE, classifyCaptureReadResult(-3))
        assertEquals(CaptureReadOutcome.STOP_CAPTURE_WITH_FAILURE, classifyCaptureReadResult(-6))
    }

    @Test
    fun pcm16PackingRoundTripsExtremes() {
        val samples = shortArrayOf(Short.MIN_VALUE, -1, 0, 1, Short.MAX_VALUE)
        val packed = samples.toPcm16LittleEndianBytes()
        assertEquals(samples.size * 2, packed.size)
        assertContentEquals(samples, packed.pcm16LittleEndianToShortArray())
    }
}
