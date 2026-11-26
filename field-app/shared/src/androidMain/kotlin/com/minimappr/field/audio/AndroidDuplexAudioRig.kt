package com.minimappr.field.audio

import android.Manifest
import android.content.Context
import android.media.MicrophoneInfo
import androidx.annotation.RequiresPermission
import com.minimappr.field.FieldAppConfiguration
import com.minimappr.field.capture.AndroidMicrophoneCaptureLoop
import com.minimappr.field.capture.AudioTimestampSampler
import com.minimappr.field.capture.CaptureBlockSink
import com.minimappr.field.capture.CaptureGapReport
import com.minimappr.field.capture.CaptureTimestampSample
import com.minimappr.field.capture.SilentCaptureDropDetector
import com.minimappr.field.chirplab.ChirpLabAudioPathFacts
import com.minimappr.field.chirplab.describeAudioDevice
import com.minimappr.field.playback.AndroidChirpPlaybackEngine
import com.minimappr.field.recording.FieldRunEventLog
import com.minimappr.field.recording.FieldRunEventTypes
import kotlinx.coroutines.CoroutineScope

/** Why a duplex start failed; the service shows it and stops. */
class DuplexAudioStartException(message: String) : RuntimeException(message)

/**
 * Capture + capture timestamps + continuous playback, started in the plan's
 * duplex order (A.1): AudioRecord running → first valid capture timestamp →
 * AudioTrack writing silence → first valid playback timestamp. Shared by
 * chirp lab and the LAN session so both measure on the same audio paths.
 *
 * Logs `capture_timestamp` (with `wall_ms`, the coarse cross-phone time
 * reference) and `capture_gap` events, from failed reads and from silent
 * drops the capture timestamps reveal (`source`); playback logs its own
 * emissions and timestamps.
 */
class AndroidDuplexAudioRig(
    private val context: Context,
    private val configuration: FieldAppConfiguration,
    captureSink: CaptureBlockSink,
    private val eventLog: FieldRunEventLog,
    forcedAudioSource: Int?,
    emissionChannelLabel: String,
    emissionLevelDbfs: Double,
    private val onCaptureTimestamp: (CaptureTimestampSample) -> Unit,
    private val onCaptureGap: (CaptureGapReport) -> Unit,
    /** False for a listen-only rig (CAPTURE mode): no AudioTrack is opened. */
    private val withPlayback: Boolean = true,
) {
    val capture = AndroidMicrophoneCaptureLoop(context, configuration.capture, captureSink, { report -> handleCaptureGap(report, GAP_SOURCE_FAILED_READ) }, forcedAudioSource)
    private val silentDropDetector = SilentCaptureDropDetector(
        configuration.capture.captureSampleRateHz,
        configuration.capture.silentCaptureDropThresholdMilliseconds * NANOSECONDS_PER_MILLISECOND,
    )
    val playback = AndroidChirpPlaybackEngine(context, configuration.playback, eventLog, emissionChannelLabel, emissionLevelDbfs)
    private val timestampSampler = AudioTimestampSampler({ capture.audioRecord }, configuration.capture.captureTimestampIntervalMilliseconds, ::handleCaptureTimestamp)

    var startedNanoTime: Long = 0
        private set
    var startedWallClockMilliseconds: Long = 0
        private set

    private fun handleCaptureTimestamp(sample: CaptureTimestampSample) {
        eventLog.append(FieldRunEventTypes.CAPTURE_TIMESTAMP) {
            put("frame_position", sample.anchor.framePosition)
            put("nano_time", sample.anchor.nanoTime)
            put("wall_ms", sample.wallClockMilliseconds)
            put("frames_read", capture.framesRead)
        }
        silentDropDetector.observe(sample.anchor)?.let { report -> handleCaptureGap(report, GAP_SOURCE_TIMESTAMP_STEP) }
        onCaptureTimestamp(sample)
    }

    private fun handleCaptureGap(report: CaptureGapReport, source: String) {
        eventLog.append(FieldRunEventTypes.CAPTURE_GAP) {
            put("source", source)
            put("expected_frame", report.expectedFrame)
            put("got_frame", report.gotFrame)
            put("missing_frames", report.missingFrames)
        }
        onCaptureGap(report)
    }

    /** Blocking (waits for timestamps); call off the main thread. Throws [DuplexAudioStartException]. */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    suspend fun start(scope: CoroutineScope) {
        startedNanoTime = System.nanoTime()
        startedWallClockMilliseconds = System.currentTimeMillis()
        if (!capture.start(scope)) throw DuplexAudioStartException("no audio source could be opened")
        val firstCaptureTimestamp = timestampSampler.awaitFirstValid(configuration.capture.firstCaptureTimestampTimeoutMilliseconds)
            ?: throw DuplexAudioStartException("AudioRecord never produced a valid MONOTONIC timestamp")
        handleCaptureTimestamp(firstCaptureTimestamp)
        timestampSampler.start(scope)
        if (!withPlayback) return
        if (!playback.start()) throw DuplexAudioStartException("AudioTrack could not be opened")
        playback.awaitFirstTimestamp(configuration.playback.firstPlaybackTimestampTimeoutMilliseconds)
            ?: throw DuplexAudioStartException("AudioTrack never produced a valid timestamp")
    }

    fun activeMicrophones(): List<MicrophoneInfo> = runCatching { capture.audioRecord?.activeMicrophones }.getOrNull().orEmpty()

    fun routedInputDescription(): String? = describeAudioDevice(capture.audioRecord?.routedDevice)

    fun routedOutputDescription(): String? = describeAudioDevice(playback.routedDevice)

    fun audioPathFacts(requestedSource: String) = ChirpLabAudioPathFacts(
        requestedSource = requestedSource,
        actualSource = capture.activeAudioSourceName,
        captureBufferSizeFrames = capture.bufferSizeFrames,
        captureSampleRateHz = configuration.capture.captureSampleRateHz,
        playbackBufferSizeFrames = playback.bufferSizeFrames,
        playbackPerformanceMode = playback.performanceModeName,
        lowLatencyRequested = playback.lowLatencyRequested,
        routedInput = routedInputDescription(),
        routedOutput = routedOutputDescription(),
    )

    /** Playback first (its outcomes may still reach the device), then capture. */
    fun stop() {
        timestampSampler.stop()
        playback.stop()
        capture.stop()
    }

    private companion object {
        const val NANOSECONDS_PER_MILLISECOND = 1_000_000L
        const val GAP_SOURCE_FAILED_READ = "failed_read"
        const val GAP_SOURCE_TIMESTAMP_STEP = "timestamp_step"
    }
}
