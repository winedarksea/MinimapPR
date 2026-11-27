package com.minimappr.field.playback

import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTimestamp
import android.media.AudioTrack
import android.os.Process
import android.os.SystemClock
import android.util.Log
import com.minimappr.field.FieldPlaybackConfiguration
import com.minimappr.field.capture.pcm16LittleEndianToShortArray
import com.minimappr.field.recording.DiscardingFieldRunEventLog
import com.minimappr.field.recording.FieldRunEventLog
import com.minimappr.field.recording.FieldRunEventTypes
import kotlin.math.max

private const val LOG_TAG = "FieldChirpPlayback"
private const val STEREO_CHANNEL_COUNT = 2
private const val PCM16_BYTES_PER_SAMPLE = 2
private const val MILLISECONDS_PER_SECOND = 1_000

/** Called once per splice: `predictedPresentationNs` null when no anchor existed. */
fun interface ChirpSpliceOutcomeListener {
    fun onSpliceOutcome(played: Boolean, outputFrameIndex: Long, predictedPresentationNs: Long?)
}

private class PendingChirpSplice(
    val emissionIndex: Long,
    val startFrame: Long,
    val interleavedStereo: ShortArray,
    val listener: ChirpSpliceOutcomeListener,
    /** Logged on this splice's `emission` event (chirp lab channel cycles). */
    val channelLabel: String,
) {
    val frameCount: Long get() = (interleavedStereo.size / STEREO_CHANNEL_COUNT).toLong()
    /** Set by the writer thread (under the engine lock) once its first frame is mixed. */
    var wasStarted = false
}

/**
 * Continuous stereo PCM16 48 kHz AudioTrack that writes silence and splices
 * chirps in at exact output frames (plan A.1 duplex start, step 2).
 *
 * Why continuous: an open stream keeps the speaker amplifier awake, so the
 * chirp onset is never eaten by an amp ramp, and the write-frame counter
 * gives a stable output timeline (frame 0 = first written frame = the
 * AudioTrack timestamp's frame 0). A dedicated URGENT_AUDIO writer thread
 * blocks in write(); [scheduleChirpAtOutputFrame] and the [ChirpPlaybackPort]
 * method only queue a splice under [lock].
 *
 * Every splice produces exactly one outcome: played when the block holding
 * its first frame was written, not played when the writer had already
 * passed it or the engine stopped. Emissions are logged to [eventLog].
 */
class AndroidChirpPlaybackEngine(
    private val context: Context,
    private val configuration: FieldPlaybackConfiguration,
    private val eventLog: FieldRunEventLog = DiscardingFieldRunEventLog,
    private val emissionChannelLabel: String,
    private val emissionLevelDbfs: Double,
) : ChirpPlaybackPort {
    private val lock = Any()
    private val scheduler = PlaybackFrameScheduler(configuration.playbackSampleRateHz, configuration.schedulingSafetyMarginMilliseconds)
    private val pending = mutableListOf<PendingChirpSplice>()
    private var track: AudioTrack? = null
    private var writerThread: Thread? = null
    @Volatile private var running = false
    val isRunning: Boolean get() = running

    @Volatile var framesWritten: Long = 0
        private set
    var lowLatencyRequested: Boolean = false
        private set
    var bufferSizeFrames: Int = 0
        private set
    val performanceModeName: String get() = track?.let { performanceModeName(it.performanceMode) } ?: "NOT_STARTED"
    val underrunCount: Int get() = track?.underrunCount ?: 0
    val latestAnchor: FrameTimeAnchor? get() = synchronized(lock) { scheduler.latestAnchor }
    val routedDevice: AudioDeviceInfo? get() = track?.routedDevice

    /** Opens and starts the stream; false (logged) when the track cannot be built. */
    fun start(): Boolean {
        if (running) return true
        val built = runCatching { buildTrack() }.onFailure { Log.e(LOG_TAG, "AudioTrack build failed", it) }.getOrNull() ?: return false
        if (built.state != AudioTrack.STATE_INITIALIZED) {
            built.release()
            return false
        }
        builtInSpeaker()?.let { built.preferredDevice = it }
        built.play()
        track = built
        bufferSizeFrames = built.bufferSizeInFrames
        running = true
        writerThread = Thread({ writeLoop(built) }, "field-chirp-playback").apply { start() }
        Log.i(LOG_TAG, "playback running: mode=$performanceModeName buffer=$bufferSizeFrames frames")
        return true
    }

    private fun buildTrack(): AudioTrack {
        val sampleRateHz = configuration.playbackSampleRateHz
        val minimumBufferBytes = AudioTrack.getMinBufferSize(sampleRateHz, AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_16BIT)
        val floorBytes = sampleRateHz * configuration.audioTrackMinimumBufferMilliseconds / MILLISECONDS_PER_SECOND *
            STEREO_CHANNEL_COUNT * PCM16_BYTES_PER_SAMPLE
        val builder = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRateHz)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .build()
            )
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(max(minimumBufferBytes * configuration.audioTrackBufferMinimumSizeMultiplier, floorBytes))
        lowLatencyRequested = context.packageManager.hasSystemFeature(PackageManager.FEATURE_AUDIO_LOW_LATENCY)
        if (lowLatencyRequested) builder.setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
        return builder.build()
    }

    private fun builtInSpeaker(): AudioDeviceInfo? {
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        return audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS).firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER }
    }

    private fun writeLoop(audioTrack: AudioTrack) {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
        val blockFrames = configuration.writeBlockFrameCount
        val block = ShortArray(blockFrames * STEREO_CHANNEL_COUNT)
        val timestamp = AudioTimestamp()
        var lastTimestampPollMs = 0L
        var timestampPollCount = 0L
        while (running) {
            block.fill(0)
            val blockStart = framesWritten
            val started = mutableListOf<PendingChirpSplice>()
            val late = mutableListOf<PendingChirpSplice>()
            synchronized(lock) { mixPendingInto(block, blockStart, blockStart + blockFrames, started, late) }
            late.forEach { notifyOutcome(it, played = false) }
            val writtenShorts = audioTrack.write(block, 0, block.size, AudioTrack.WRITE_BLOCKING)
            if (writtenShorts < 0) {
                Log.e(LOG_TAG, "AudioTrack.write failed: $writtenShorts")
                break
            }
            framesWritten = blockStart + writtenShorts / STEREO_CHANNEL_COUNT
            started.forEach { notifyOutcome(it, played = true) }
            val nowMs = SystemClock.elapsedRealtime()
            if (nowMs - lastTimestampPollMs >= configuration.playbackTimestampPollIntervalMilliseconds) {
                lastTimestampPollMs = nowMs
                if (audioTrack.getTimestamp(timestamp) && timestamp.nanoTime != 0L) {
                    val anchor = FrameTimeAnchor(timestamp.framePosition, timestamp.nanoTime)
                    synchronized(lock) { scheduler.updateAnchor(anchor) }
                    if (timestampPollCount++ % configuration.playbackTimestampLogEveryNthPoll == 0L) {
                        eventLog.append(FieldRunEventTypes.PLAYBACK_TIMESTAMP) {
                            put("frame_position", anchor.framePosition)
                            put("nano_time", anchor.nanoTime)
                            put("frames_written", framesWritten)
                        }
                    }
                }
            }
        }
        // Stopped (or failed): every queued splice still gets its one outcome.
        val leftovers = synchronized(lock) { pending.toList().also { pending.clear() } }
        leftovers.forEach { notifyOutcome(it, played = false) }
    }

    /** Copy overlapping chirp frames into the block; collect starts and late splices. */
    private fun mixPendingInto(
        block: ShortArray,
        blockStart: Long,
        blockEnd: Long,
        started: MutableList<PendingChirpSplice>,
        late: MutableList<PendingChirpSplice>,
    ) {
        val iterator = pending.iterator()
        while (iterator.hasNext()) {
            val splice = iterator.next()
            val spliceEnd = splice.startFrame + splice.frameCount
            if (splice.startFrame >= blockEnd) continue
            if (!splice.wasStarted && splice.startFrame < blockStart) {
                // The writer already passed its first frame: a partial chirp
                // would be a corrupt measurement, so it is not played at all.
                iterator.remove()
                late += splice
                continue
            }
            val firstFrame = max(splice.startFrame, blockStart)
            val lastFrameExclusive = minOf(spliceEnd, blockEnd)
            for (frame in firstFrame until lastFrameExclusive) {
                val sourceOffset = ((frame - splice.startFrame) * STEREO_CHANNEL_COUNT).toInt()
                val targetOffset = ((frame - blockStart) * STEREO_CHANNEL_COUNT).toInt()
                block[targetOffset] = splice.interleavedStereo[sourceOffset]
                block[targetOffset + 1] = splice.interleavedStereo[sourceOffset + 1]
            }
            if (!splice.wasStarted) {
                splice.wasStarted = true
                started += splice
            }
            if (spliceEnd <= blockEnd) iterator.remove()
        }
    }

    private fun notifyOutcome(splice: PendingChirpSplice, played: Boolean) {
        val predictedNs = synchronized(lock) { scheduler.predictPresentationNs(splice.startFrame) }
        eventLog.append(FieldRunEventTypes.EMISSION) {
            put("emission_index", splice.emissionIndex)
            put("output_frame_index", splice.startFrame)
            put("target_ns", predictedNs)
            put("played", played)
            put("channel", splice.channelLabel)
            put("level_dbfs", emissionLevelDbfs)
        }
        splice.listener.onSpliceOutcome(played, splice.startFrame, predictedNs)
    }

    /**
     * Chirp lab: splice at an exact write-frame index (k·P + offset).
     * `channelLabel` names the channel this PCM drives; it defaults to the
     * engine's single channel.
     */
    fun scheduleChirpAtOutputFrame(
        emissionIndex: Long,
        outputFrameIndex: Long,
        pcm16StereoLittleEndian: ByteArray,
        channelLabel: String = emissionChannelLabel,
        listener: ChirpSpliceOutcomeListener,
    ) {
        val splice = PendingChirpSplice(emissionIndex, outputFrameIndex, pcm16StereoLittleEndian.pcm16LittleEndianToShortArray(), listener, channelLabel)
        val accepted = synchronized(lock) {
            val acceptable = running && outputFrameIndex >= framesWritten + safetyMarginFrames()
            if (acceptable) pending += splice
            acceptable
        }
        if (!accepted) notifyOutcome(splice, played = false)
    }

    private fun safetyMarginFrames(): Long =
        configuration.schedulingSafetyMarginMilliseconds * configuration.playbackSampleRateHz / MILLISECONDS_PER_SECOND

    /** Session: map a local-ns target through the latest timestamp anchor. */
    override fun scheduleChirpAtLocalNs(
        emissionId: Long,
        targetPresentationNs: Long,
        pcm16StereoLittleEndian: ByteArray,
        outcomeListener: ChirpEmissionOutcomeListener,
    ) {
        val decision = synchronized(lock) { if (running) scheduler.decide(targetPresentationNs, framesWritten) else null }
        if (decision !is PlaybackScheduleDecision.Scheduled) {
            Log.w(LOG_TAG, "emission $emissionId not schedulable: $decision")
            val targetFrame = (decision as? PlaybackScheduleDecision.RejectedTooLate)?.targetFrame ?: -1L
            eventLog.append(FieldRunEventTypes.EMISSION) {
                put("emission_index", emissionId)
                put("output_frame_index", targetFrame)
                put("target_ns", targetPresentationNs)
                put("played", false)
                put("channel", emissionChannelLabel)
                put("level_dbfs", emissionLevelDbfs)
                put("reject_reason", decision?.toString() ?: "engine_stopped")
            }
            outcomeListener.onChirpEmissionOutcome(false, targetPresentationNs)
            return
        }
        scheduleChirpAtOutputFrame(emissionId, decision.outputFrameIndex, pcm16StereoLittleEndian) { played, _, predictedNs ->
            outcomeListener.onChirpEmissionOutcome(played, predictedNs ?: decision.predictedPresentationNs)
        }
    }

    /** Duplex start step 2: block until the first valid AudioTrack timestamp. */
    fun awaitFirstTimestamp(timeoutMilliseconds: Long): FrameTimeAnchor? {
        val deadline = SystemClock.elapsedRealtime() + timeoutMilliseconds
        while (SystemClock.elapsedRealtime() < deadline) {
            latestAnchor?.let { return it }
            SystemClock.sleep(FIRST_TIMESTAMP_POLL_MILLISECONDS)
        }
        return latestAnchor
    }

    fun stop() {
        if (!running) return
        running = false
        val audioTrack = track
        // pause+flush unblocks a WRITE_BLOCKING write so the writer can exit.
        runCatching { audioTrack?.pause(); audioTrack?.flush() }
        writerThread?.join(configuration.writerThreadJoinTimeoutMilliseconds)
        writerThread = null
        runCatching { audioTrack?.stop() }
        audioTrack?.release()
        track = null
    }

    private companion object {
        const val FIRST_TIMESTAMP_POLL_MILLISECONDS = 20L

        fun performanceModeName(mode: Int): String = when (mode) {
            AudioTrack.PERFORMANCE_MODE_LOW_LATENCY -> "LOW_LATENCY"
            AudioTrack.PERFORMANCE_MODE_POWER_SAVING -> "POWER_SAVING"
            AudioTrack.PERFORMANCE_MODE_NONE -> "NONE"
            else -> "mode#$mode"
        }
    }
}
