package com.minimappr.field.capture

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.os.Process
import android.util.Log
import androidx.annotation.RequiresPermission
import com.minimappr.field.FieldCaptureConfiguration
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import java.util.concurrent.Executors
import kotlin.math.max

private const val LOG_TAG = "FieldCaptureLoop"
private const val PCM16_MONO_BYTES_PER_FRAME = 2

/**
 * 48 kHz mono PCM16 capture on a blocking read loop.
 *
 * Adapted from the On Device Audio App (MIT, (c) 2026 Colin Catlin):
 * `MicrophoneAudioInputSource` (AudioRecord.Builder setup and buffer
 * sizing from getMinBufferSize) and `AudioRecordingSession` (blocking
 * READ_BLOCKING loop on Dispatchers.IO). Not copied: WAV writing, RMS and
 * waveform UI flows. Added: sample-count timing and gap flagging (plan,
 * Sample continuity), the source fallback chain and a forced source for
 * chirp lab's capture-path sweep (A7).
 *
 * Timeline: device sample index 0 is the first frame read after
 * startRecording, which is also AudioRecord.getTimestamp's frame 0 for a
 * freshly built recorder, so timestamps and indices share one timeline.
 *
 * The loop does no heavy work: each block goes straight to [captureBlockSink]
 * (device rolling buffer, WAV queue), so the OS buffer never backs up.
 */
class AndroidMicrophoneCaptureLoop(
    private val context: Context,
    private val captureConfiguration: FieldCaptureConfiguration,
    private val captureBlockSink: CaptureBlockSink,
    private val onCaptureGap: (CaptureGapReport) -> Unit,
    /** A MediaRecorder.AudioSource to use exclusively; null = preference chain. */
    private val forcedAudioSource: Int? = null,
) {
    private val sampleIndexCounter = DeviceSampleIndexCounter()
    private var captureJob: Job? = null
    /** Dedicated thread: URGENT_AUDIO priority must not leak into a shared pool. */
    private var captureThreadDispatcher: ExecutorCoroutineDispatcher? = null

    @Volatile
    var audioRecord: AudioRecord? = null
        private set

    var activeAudioSource: Int? = null
        private set
    val activeAudioSourceName: String? get() = activeAudioSource?.let(::audioSourceDisplayName)

    var bufferSizeFrames: Int = 0
        private set

    val captureGapCount: Long get() = sampleIndexCounter.captureGapCount
    val framesRead: Long get() = sampleIndexCounter.nextDeviceSampleIndex

    fun hasRecordAudioPermission(): Boolean =
        context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    /** Returns false when no source could be opened or started (reason logged). */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun start(scope: CoroutineScope): Boolean {
        if (captureJob != null) return true
        val recorder = openFirstWorkingAudioRecord() ?: return false
        val recordingStarted = runCatching { recorder.startRecording() }.isSuccess &&
            recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING
        if (!recordingStarted) {
            // e.g. another app holds the mic exclusively: do not leak the recorder.
            Log.w(LOG_TAG, "AudioRecord.startRecording did not enter RECORDING")
            recorder.release()
            return false
        }
        audioRecord = recorder
        val readBuffer = ShortArray(captureConfiguration.captureBlockSampleCount)
        val dispatcher = Executors.newSingleThreadExecutor { runnable -> Thread(runnable, "field-capture") }.asCoroutineDispatcher()
        captureThreadDispatcher = dispatcher
        captureJob = scope.launch(dispatcher) {
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
            while (isActive) {
                val samplesReadOrErrorCode = recorder.read(readBuffer, 0, readBuffer.size, AudioRecord.READ_BLOCKING)
                when (classifyCaptureReadResult(samplesReadOrErrorCode)) {
                    CaptureReadOutcome.DELIVER_BLOCK -> {
                        val firstDeviceSampleIndex = sampleIndexCounter.claimIndicesForRead(samplesReadOrErrorCode)
                        captureBlockSink.onCaptureBlock(
                            CaptureBlock(firstDeviceSampleIndex, readBuffer.copyOf(samplesReadOrErrorCode))
                        )
                    }
                    CaptureReadOutcome.FLAG_GAP_AND_CONTINUE -> {
                        // stop() makes a blocking read return early; that is not a gap.
                        if (!isActive) break
                        Log.w(LOG_TAG, "AudioRecord.read returned no samples")
                        val expectedFrame = sampleIndexCounter.nextDeviceSampleIndex
                        val gapsBefore = sampleIndexCounter.captureGapCount
                        sampleIndexCounter.recordFailedRead()
                        if (sampleIndexCounter.captureGapCount != gapsBefore) {
                            val gotFrame = sampleIndexCounter.nextDeviceSampleIndex
                            onCaptureGap(CaptureGapReport(expectedFrame, gotFrame, gotFrame - expectedFrame))
                        }
                    }
                    CaptureReadOutcome.STOP_CAPTURE_WITH_FAILURE -> {
                        if (!isActive) break
                        throw CaptureReadFailedException(samplesReadOrErrorCode)
                    }
                }
            }
        }
        return true
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private fun openFirstWorkingAudioRecord(): AudioRecord? {
        val sampleRateHz = captureConfiguration.captureSampleRateHz
        val channelMask = AudioFormat.CHANNEL_IN_MONO
        val encoding = AudioFormat.ENCODING_PCM_16BIT
        val minimumBufferBytes = AudioRecord.getMinBufferSize(sampleRateHz, channelMask, encoding)
        val requestedBufferBytes = max(
            minimumBufferBytes,
            captureConfiguration.captureBlockSampleCount * PCM16_MONO_BYTES_PER_FRAME *
                captureConfiguration.audioRecordBufferBlockCount,
        )
        val candidates = forcedAudioSource?.let(::listOf) ?: fieldAudioSourceCandidates(context)
        for (audioSource in candidates) {
            val recorder = runCatching {
                AudioRecord.Builder()
                    .setAudioSource(audioSource)
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setSampleRate(sampleRateHz)
                            .setChannelMask(channelMask)
                            .setEncoding(encoding)
                            .build()
                    )
                    .setBufferSizeInBytes(requestedBufferBytes)
                    .build()
            }.getOrNull()
            if (recorder != null && recorder.state == AudioRecord.STATE_INITIALIZED) {
                activeAudioSource = audioSource
                bufferSizeFrames = recorder.bufferSizeInFrames
                Log.i(LOG_TAG, "capturing from $activeAudioSourceName, buffer=$bufferSizeFrames frames")
                return recorder
            }
            recorder?.release()
            Log.w(LOG_TAG, "audio source ${audioSourceDisplayName(audioSource)} failed to initialize")
        }
        return null
    }

    /**
     * Stops capture and releases the recorder. Order matters: cancel, then
     * AudioRecord.stop() to unblock a READ_BLOCKING read, then wait for the
     * loop to exit, and only then release. Releasing while the capture
     * thread is inside read() frees the native recorder under it, and the
     * loop could also push a block into an already-closed Rust device.
     */
    fun stop() {
        val job = captureJob
        captureJob = null
        val recorder = audioRecord
        audioRecord = null
        job?.cancel()
        recorder?.let { runCatching { if (it.recordingState == AudioRecord.RECORDSTATE_RECORDING) it.stop() } }
        if (job != null) {
            val loopExited = runBlocking {
                withTimeoutOrNull(captureConfiguration.captureStopJoinTimeoutMilliseconds) { job.join() } != null
            }
            if (!loopExited) Log.w(LOG_TAG, "capture loop did not exit within the stop timeout")
        }
        recorder?.release()
        captureThreadDispatcher?.close()
        captureThreadDispatcher = null
    }
}
