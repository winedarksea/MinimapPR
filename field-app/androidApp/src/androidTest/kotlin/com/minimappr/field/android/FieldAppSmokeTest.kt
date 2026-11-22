package com.minimappr.field.android

import android.Manifest
import android.os.SystemClock
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.minimappr.field.ffi.describeFrame
import com.minimappr.field.ffi.fieldFfiSurfaceVersion
import com.minimappr.field.node.FieldCaptureServiceState
import com.minimappr.field.node.FieldServiceStatus
import com.minimappr.field.node.FieldServiceStatusBoard
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Emulator smoke test: app launches, the Rust library answers, the
 * microphone foreground service starts and the node runtime advances.
 * Not a microphone validation: the emulator mic is virtual.
 */
@RunWith(AndroidJUnit4::class)
class FieldAppSmokeTest {
    private val serviceStartTimeoutMilliseconds = 10_000L
    private val pollIntervalMilliseconds = 100L
    private val expectedFfiSurfaceVersion = 1u
    private val joinRequestKindId = 1

    @get:Rule
    val recordAudioPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.RECORD_AUDIO)

    private fun awaitStatus(predicate: (FieldServiceStatus) -> Boolean): FieldServiceStatus {
        val deadline = SystemClock.elapsedRealtime() + serviceStartTimeoutMilliseconds
        while (SystemClock.elapsedRealtime() < deadline) {
            val status = FieldServiceStatusBoard.status.value
            if (predicate(status)) return status
            SystemClock.sleep(pollIntervalMilliseconds)
        }
        return FieldServiceStatusBoard.status.value
    }

    @Test
    fun appLaunchesServiceStartsAndFfiAnswers() {
        assertEquals(expectedFfiSurfaceVersion, fieldFfiSurfaceVersion())
        ActivityScenario.launch(FieldMainActivity::class.java).use { scenario ->
            scenario.onActivity { activity -> FieldCaptureForegroundService.start(activity) }
            val running = awaitStatus {
                it.serviceState == FieldCaptureServiceState.RUNNING && it.node.capturedSampleCount > 0 &&
                    it.node.outboundFramesSent > 0
            }
            assertEquals(running.failureReason, FieldCaptureServiceState.RUNNING, running.serviceState)
            assertTrue("capture advanced", running.node.capturedSampleCount > 0)
            assertEquals("first frame is a join request", joinRequestKindId, running.node.lastOutboundMessageKindId)

            FieldCaptureForegroundService.stop(InstrumentationRegistry.getInstrumentation().targetContext)
            val stopped = awaitStatus { it.serviceState == FieldCaptureServiceState.STOPPED }
            assertEquals(FieldCaptureServiceState.STOPPED, stopped.serviceState)
        }
        // A malformed frame must surface as the typed FFI error, not a crash.
        assertTrue(runCatching { describeFrame(byteArrayOf()) }.isFailure)
    }
}
