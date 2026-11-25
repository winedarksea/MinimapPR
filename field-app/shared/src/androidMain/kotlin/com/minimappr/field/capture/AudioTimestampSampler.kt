package com.minimappr.field.capture

import android.media.AudioRecord
import android.media.AudioTimestamp
import com.minimappr.field.playback.FrameTimeAnchor
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull

/** One capture timestamp plus the wall clock sampled right after it (coarse cross-phone reference). */
data class CaptureTimestampSample(val anchor: FrameTimeAnchor, val wallClockMilliseconds: Long)

/**
 * Polls `AudioRecord.getTimestamp(TIMEBASE_MONOTONIC)` (~1 Hz) and fans each
 * valid anchor out to the device (`reportCaptureTimestamp`), the self-chirp
 * monitor and the run log. MONOTONIC = CLOCK_MONOTONIC = System.nanoTime,
 * the clock every device call uses. A zero nanoTime means "not yet valid"
 * on some HALs and is skipped.
 */
class AudioTimestampSampler(
    private val recordProvider: () -> AudioRecord?,
    private val intervalMilliseconds: Long,
    private val onSample: (CaptureTimestampSample) -> Unit,
) {
    private val timestamp = AudioTimestamp()
    private var job: Job? = null

    /** One read; null when the recorder has no valid timestamp yet. */
    fun sampleOnce(): CaptureTimestampSample? {
        val record = recordProvider() ?: return null
        val status = runCatching { record.getTimestamp(timestamp, AudioTimestamp.TIMEBASE_MONOTONIC) }.getOrNull()
        if (status != AudioRecord.SUCCESS || timestamp.nanoTime == 0L) return null
        return CaptureTimestampSample(FrameTimeAnchor(timestamp.framePosition, timestamp.nanoTime), System.currentTimeMillis())
    }

    /** Duplex start step 1: wait for the first valid timestamp (polled fast). */
    suspend fun awaitFirstValid(timeoutMilliseconds: Long, pollMilliseconds: Long = FIRST_TIMESTAMP_POLL_MILLISECONDS): CaptureTimestampSample? =
        withTimeoutOrNull(timeoutMilliseconds) {
            var sample = sampleOnce()
            while (sample == null) {
                delay(pollMilliseconds)
                sample = sampleOnce()
            }
            sample
        }

    fun start(scope: CoroutineScope) {
        if (job != null) return
        job = scope.launch(Dispatchers.Default) {
            while (isActive) {
                sampleOnce()?.let(onSample)
                delay(intervalMilliseconds)
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
    }

    private companion object {
        const val FIRST_TIMESTAMP_POLL_MILLISECONDS = 20L
    }
}
