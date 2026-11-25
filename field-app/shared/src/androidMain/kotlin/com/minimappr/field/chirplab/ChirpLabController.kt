package com.minimappr.field.chirplab

import android.Manifest
import android.content.Context
import androidx.annotation.RequiresPermission
import com.minimappr.field.FieldAppConfiguration
import com.minimappr.field.audio.AndroidDuplexAudioRig
import com.minimappr.field.capture.CaptureBlockSink
import com.minimappr.field.capture.FanOutCaptureBlockSink
import com.minimappr.field.capture.toForcedAudioSource
import com.minimappr.field.ffi.FieldChirpDetector
import com.minimappr.field.ffi.chirpPlaybackPcm16
import com.minimappr.field.ffi.speedOfSoundForTemperature
import com.minimappr.field.playback.FrameTimeline
import com.minimappr.field.recording.FieldRunEventTypes
import com.minimappr.field.recording.FieldRunRecording
import kotlinx.atomicfu.atomic
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File

/** Self-chirp templates do not depend on air temperature; any valid speed works. */
private const val SELF_CHIRP_DETECTOR_TEMPERATURE_C = 20.0
private const val STATUS_REFRESH_INTERVAL_MILLISECONDS = 250L

/**
 * One Stage A chirp lab run on one phone (plan A.1/A.2): duplex start, then
 * chirps at k·P (+P/2 for role B) on the phone's own AudioTrack write-frame
 * timeline, continuous capture to capture.wav, events.jsonl and
 * session.json under `<outputRoot>/<run_id>/`, plus live self-SNR.
 *
 * No transport: the two phones never talk. Start skew only moves where the
 * other phone's chirps land in the gap; the offline analyser measures it.
 */
class ChirpLabController(
    private val context: Context,
    private val configuration: FieldAppConfiguration,
    val parameters: ChirpLabRunParameters,
    outputRootDirectory: File,
) {
    private val runDirectory = File(outputRootDirectory, parameters.runId)
    private val sampleRateHz = configuration.capture.captureSampleRateHz
    private val timeline = FrameTimeline(sampleRateHz)
    private val chirpPcm = chirpPlaybackPcm16(parameters.waveform, parameters.channel, parameters.levelDbfs)
    private val detector = FieldChirpDetector(parameters.waveform, speedOfSoundForTemperature(SELF_CHIRP_DETECTOR_TEMPERATURE_C))
    private val planner = ChirpLabEmissionPlanner(
        timeline.framesForMilliseconds(parameters.periodMilliseconds.toLong()),
        timeline.framesForMilliseconds(parameters.periodOffsetMilliseconds.toLong()),
    )
    private val mutableStatus = MutableStateFlow(
        ChirpLabStatus(runId = parameters.runId, role = parameters.role.wireName, outputDirectory = runDirectory.absolutePath)
    )
    val status: StateFlow<ChirpLabStatus> = mutableStatus.asStateFlow()
    private val emissionsPlayed = atomic(0L)
    private val emissionsSkipped = atomic(0L)
    private val captureGaps = atomic(0L)
    @Volatile private var stopRequested = false
    private var recording: FieldRunRecording? = null
    private var rig: AndroidDuplexAudioRig? = null
    private var conditionMonitor: AndroidDeviceConditionMonitor? = null
    private var statusJob: Job? = null

    /** Suspends through the duplex start; throws DuplexAudioStartException on failure. */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    suspend fun start(scope: CoroutineScope) {
        val runRecording = FieldRunRecording(runDirectory, sampleRateHz, configuration.chirpLab, scope)
        recording = runRecording
        val selfMonitor = SelfChirpMonitor(
            detector, sampleRateHz, configuration.chirpLab.selfChirpCaptureRingSeconds,
            configuration.chirpLab.selfChirpSearchMarginMilliseconds,
            detectionRunner = { work -> scope.launch(Dispatchers.Default) { work() } },
            onReadout = ::recordSelfChirp,
        )
        val sinks: List<CaptureBlockSink> = listOf(runRecording, selfMonitor)
        val duplexRig = AndroidDuplexAudioRig(
            context, configuration, FanOutCaptureBlockSink(sinks), runRecording,
            parameters.captureSource.toForcedAudioSource(),
            ChirpLabWireNames.channelName(parameters.channel), parameters.levelDbfs,
            onCaptureTimestamp = { selfMonitor.onCaptureAnchor(it.anchor) },
            onCaptureGap = { captureGaps.incrementAndGet() },
        )
        rig = duplexRig
        duplexRig.start(scope)
        runRecording.writeSessionJson(sessionJson(duplexRig))
        conditionMonitor = AndroidDeviceConditionMonitor(
            context, configuration.chirpLab, runRecording, duplexRig::routedInputDescription, duplexRig::routedOutputDescription,
        ).also { it.start(scope) }
        val leadFrames = timeline.framesForMilliseconds(parameters.leadMilliseconds.toLong())
        scheduleEmission(duplexRig, selfMonitor, planner.firstEmissionAtOrAfter(duplexRig.playback.framesWritten + leadFrames), leadFrames)
        mutableStatus.update {
            it.copy(
                running = true,
                captureSourceName = duplexRig.capture.activeAudioSourceName,
                playbackPerformanceMode = duplexRig.playback.performanceModeName,
            )
        }
        statusJob = scope.launch {
            while (isActive) {
                refreshCounters(duplexRig, runRecording)
                delay(STATUS_REFRESH_INTERVAL_MILLISECONDS)
            }
        }
    }

    /**
     * Queue one emission; its outcome (on the playback thread) queues the
     * next. A skipped slot re-plans from the current write position, keeping
     * k a true period index.
     */
    private fun scheduleEmission(duplexRig: AndroidDuplexAudioRig, selfMonitor: SelfChirpMonitor, emission: PlannedChirpLabEmission, leadFrames: Long) {
        duplexRig.playback.scheduleChirpAtOutputFrame(emission.emissionIndex, emission.outputFrameIndex, chirpPcm) { played, _, predictedNs ->
            if (played) {
                emissionsPlayed.incrementAndGet()
                if (predictedNs != null) selfMonitor.expectEmission(emission.emissionIndex, predictedNs)
            } else {
                emissionsSkipped.incrementAndGet()
            }
            if (stopRequested || !duplexRig.playback.isRunning) return@scheduleChirpAtOutputFrame
            val next = if (played) planner.next(emission) else planner.firstEmissionAtOrAfter(duplexRig.playback.framesWritten + leadFrames)
            scheduleEmission(duplexRig, selfMonitor, next, leadFrames)
        }
    }

    private fun recordSelfChirp(readout: SelfChirpReadout) {
        val detection = readout.detection
        recording?.append(FieldRunEventTypes.SELF_CHIRP) {
            put("emission_index", readout.emissionIndex)
            put("detected", detection != null)
            put("snr_db", detection?.snrDb)
            put("peak_fraction", detection?.peakFraction)
            put("predicted_capture_frame", readout.predictedCaptureFrame)
            put("detected_capture_frame", readout.detectedCaptureFrame)
        }
        mutableStatus.update { current ->
            if (detection == null) {
                current.copy(selfChirpsMissed = current.selfChirpsMissed + 1)
            } else {
                current.copy(
                    selfChirpsDetected = current.selfChirpsDetected + 1,
                    lastSelfSnrDb = detection.snrDb,
                    lastSelfPeakFraction = detection.peakFraction,
                    lastSelfArrivalErrorFrames = readout.detectedCaptureFrame?.minus(readout.predictedCaptureFrame),
                )
            }
        }
    }

    private fun refreshCounters(duplexRig: AndroidDuplexAudioRig, runRecording: FieldRunRecording) {
        mutableStatus.update {
            it.copy(
                emissions = emissionsPlayed.value,
                emissionsSkipped = emissionsSkipped.value,
                captureFrames = duplexRig.capture.framesRead,
                captureGaps = captureGaps.value,
                writerOverflows = runRecording.writerOverflowCount,
                playbackUnderruns = duplexRig.playback.underrunCount,
            )
        }
    }

    private fun sessionJson(duplexRig: AndroidDuplexAudioRig): String = AndroidDeviceSurvey(context).sessionJson(
        runId = parameters.runId,
        role = parameters.role.wireName,
        mode = "chirp_lab",
        schedule = {
            put("period_ms", parameters.periodMilliseconds)
            put("offset_ms", parameters.periodOffsetMilliseconds)
            put("lead_ms", parameters.leadMilliseconds)
            putObject("waveform") {
                put("kind", ChirpLabWireNames.waveformKindName(parameters.waveform.kind))
                put("band_low_hz", parameters.waveform.bandLowHz)
                put("band_high_hz", parameters.waveform.bandHighHz)
                put("duration_s", parameters.waveform.durationS)
            }
            put("channel", ChirpLabWireNames.channelName(parameters.channel))
            put("level_dbfs", parameters.levelDbfs)
        },
        paths = duplexRig.audioPathFacts(parameters.captureSource.wireName),
        activeMicrophones = duplexRig.activeMicrophones(),
        startedNanoTime = duplexRig.startedNanoTime,
        startedWallClockMilliseconds = duplexRig.startedWallClockMilliseconds,
    )

    fun mark(label: String) {
        recording?.append(FieldRunEventTypes.MARKER) { put("label", label) }
    }

    /** Idempotent. Order: stop scheduling, audio, then flush the files. */
    fun stop() {
        if (stopRequested) return
        stopRequested = true
        statusJob?.cancel()
        conditionMonitor?.stop()
        val duplexRig = rig
        val runRecording = recording
        duplexRig?.stop()
        if (duplexRig != null && runRecording != null) refreshCounters(duplexRig, runRecording)
        runRecording?.close()
        detector.close()
        mutableStatus.update { it.copy(running = false) }
    }

    fun fail(reason: String) = mutableStatus.update { it.copy(running = false, failure = reason) }
}
