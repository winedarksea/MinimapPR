package com.minimappr.field.capture

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.util.Log
import androidx.annotation.RequiresPermission
import com.minimappr.field.FieldCaptureConfiguration
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
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
 * Sample continuity) and the source fallback chain.
 *
 * The loop does no heavy work: each block goes straight to [captureBlockSink]
 * (the Rust rolling buffer + Stage 1), so the OS buffer never backs up.
 */
class AndroidMicrophoneCaptureLoop(
    private val context: Context,
    private val captureConfiguration: FieldCaptureConfiguration,
    private val captureBlockSink: CaptureBlockSink,
    private val onCaptureGap: () -> Unit,
) {
    private val sampleIndexCounter = DeviceSampleIndexCounter()
    private var audioRecord: AudioRecord? = null
    private var captureJob: Job? = null

    var activeAudioSourceName: String? = null
        private set

    val captureGapCount: Long get() = sampleIndexCounter.captureGapCount

    fun hasRecordAudioPermission(): Boolean =
        context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    /** Returns false when no source could be opened (reason logged). */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun start(scope: CoroutineScope): Boolean {
        if (captureJob != null) return true
        val recorder = openFirstWorkingAudioRecord() ?: return false
        audioRecord = recorder
        recorder.startRecording()
        val readBuffer = ShortArray(captureConfiguration.captureBlockSampleCount)
        captureJob = scope.launch(Dispatchers.IO) {
            while (isActive) {
                val samplesRead = recorder.read(readBuffer, 0, readBuffer.size, AudioRecord.READ_BLOCKING)
                if (samplesRead > 0) {
                    val firstDeviceSampleIndex = sampleIndexCounter.claimIndicesForRead(samplesRead)
                    captureBlockSink.onCaptureBlock(
                        CaptureBlock(firstDeviceSampleIndex, readBuffer.copyOf(samplesRead))
                    )
                } else {
                    // Negative codes (ERROR_INVALID_OPERATION, ERROR_DEAD_OBJECT)
                    // or a zero read: audio may be lost, so flag a gap.
                    Log.w(LOG_TAG, "AudioRecord.read returned $samplesRead")
                    sampleIndexCounter.recordFailedRead()
                    onCaptureGap()
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
        for (audioSource in fieldAudioSourceCandidates(context)) {
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
                activeAudioSourceName = audioSourceDisplayName(audioSource)
                Log.i(LOG_TAG, "capturing from $activeAudioSourceName, buffer=$requestedBufferBytes B")
                return recorder
            }
            recorder?.release()
            Log.w(LOG_TAG, "audio source ${audioSourceDisplayName(audioSource)} failed to initialize")
        }
        return null
    }

    fun stop() {
        captureJob?.cancel()
        captureJob = null
        audioRecord?.run {
            if (recordingState == AudioRecord.RECORDSTATE_RECORDING) stop()
            release()
        }
        audioRecord = null
    }
}
