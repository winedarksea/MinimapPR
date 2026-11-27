package com.minimappr.field.chirplab

import com.minimappr.field.FieldChirpLabConfiguration
import com.minimappr.field.ffi.FieldChirpWaveformKind
import com.minimappr.field.ffi.FieldChirpWaveformSpec
import com.minimappr.field.ffi.FieldStereoChannel

/** A chirps at k·P, B at k·P + P/2 on its own write-frame timeline (plan A.1). */
enum class ChirpLabRole(val wireName: String, val periodOffsetFraction: Double) {
    A("A", 0.0),
    B("B", 0.5),
    ;

    companion object {
        fun fromWireName(name: String?): ChirpLabRole? = entries.firstOrNull { it.wireName.equals(name, ignoreCase = true) }
    }
}

/** Capture source request; DEFAULT = the app's normal preference chain. */
enum class ChirpLabCaptureSource(val wireName: String) {
    DEFAULT("default"),
    UNPROCESSED("unprocessed"),
    VOICE_RECOGNITION("voice_recognition"),
    MIC("mic"),
    CAMCORDER("camcorder"),
    ;

    companion object {
        fun fromWireName(name: String?): ChirpLabCaptureSource? = entries.firstOrNull { it.wireName.equals(name, ignoreCase = true) }
    }
}

/** Launch-contract names for waveform kinds and channels (am start extras). */
object ChirpLabWireNames {
    fun waveformKind(name: String?): FieldChirpWaveformKind? = when (name?.lowercase()) {
        "updown_hfm" -> FieldChirpWaveformKind.UP_DOWN_HYPERBOLIC
        "hfm" -> FieldChirpWaveformKind.HYPERBOLIC_FM
        "lfm" -> FieldChirpWaveformKind.LINEAR_FM
        else -> null
    }

    fun waveformKindName(kind: FieldChirpWaveformKind): String = when (kind) {
        FieldChirpWaveformKind.UP_DOWN_HYPERBOLIC -> "updown_hfm"
        FieldChirpWaveformKind.HYPERBOLIC_FM -> "hfm"
        FieldChirpWaveformKind.LINEAR_FM -> "lfm"
    }

    fun channel(name: String?): FieldStereoChannel? = when (name?.lowercase()) {
        "left" -> FieldStereoChannel.LEFT
        "right" -> FieldStereoChannel.RIGHT
        "both" -> FieldStereoChannel.BOTH
        else -> null
    }

    fun channelName(channel: FieldStereoChannel): String = channel.name.lowercase()

    /** `left,right,both` → channels in order; null when any name is unknown or the list is empty. */
    fun channelCycle(names: String?): List<FieldStereoChannel>? {
        val parts = names?.split(',')?.map { it.trim() }?.filter { it.isNotEmpty() }.orEmpty()
        if (parts.isEmpty()) return null
        val channels = parts.map { channel(it) ?: return null }
        return channels
    }

    fun channelCycleName(channels: List<FieldStereoChannel>): String = channels.joinToString(",") { channelName(it) }
}

data class ChirpLabRunParameters(
    val runId: String,
    val role: ChirpLabRole,
    val periodMilliseconds: Int,
    val leadMilliseconds: Int,
    val waveform: FieldChirpWaveformSpec,
    val channel: FieldStereoChannel,
    val captureSource: ChirpLabCaptureSource,
    val levelDbfs: Double,
    /**
     * Per-emission channel rotation inside one stream: emission k plays
     * `channelCycle[k % n]`. Every channel then shares one output latency,
     * so the analyser can compare speakers to the sample (stream restarts
     * move self latency by thousands of samples). Empty = always [channel].
     */
    val channelCycle: List<FieldStereoChannel> = emptyList(),
) {
    val periodOffsetMilliseconds: Int get() = (periodMilliseconds * role.periodOffsetFraction).toInt()

    /** Distinct channels this run plays, in first-use order. */
    val emissionChannels: List<FieldStereoChannel> get() = channelCycle.ifEmpty { listOf(channel) }.distinct()

    /**
     * Keyed by the period index k (not a play count), so a skipped slot
     * never shifts the rotation and the analyser can trust each emission's
     * logged channel against the schedule.
     */
    fun channelForEmission(emissionIndex: Long): FieldStereoChannel {
        val cycle = channelCycle.ifEmpty { return channel }
        return cycle[(emissionIndex % cycle.size).toInt()]
    }

    companion object {
        fun defaults(configuration: FieldChirpLabConfiguration, runId: String, role: ChirpLabRole = ChirpLabRole.A) =
            ChirpLabRunParameters(
                runId = runId,
                role = role,
                periodMilliseconds = configuration.defaultPeriodMilliseconds,
                leadMilliseconds = configuration.defaultLeadMilliseconds,
                waveform = FieldChirpWaveformSpec(
                    kind = FieldChirpWaveformKind.UP_DOWN_HYPERBOLIC,
                    bandLowHz = configuration.defaultBandLowHz,
                    bandHighHz = configuration.defaultBandHighHz,
                    durationS = configuration.defaultChirpDurationSeconds,
                ),
                channel = FieldStereoChannel.LEFT,
                captureSource = ChirpLabCaptureSource.DEFAULT,
                levelDbfs = configuration.defaultLevelDbfs,
            )
    }
}

/** One planned emission on the write-frame timeline. */
data class PlannedChirpLabEmission(val emissionIndex: Long, val outputFrameIndex: Long)

/**
 * Pure emission schedule: emission k starts at output frame k·P + offset.
 * The first emission is the smallest k at least `leadFrames` beyond what is
 * already written; later ones keep k+1 unless the writer overtook them
 * (then they are skipped and logged, never shifted, so k stays a clean
 * period index for the analyser).
 */
class ChirpLabEmissionPlanner(
    private val periodFrames: Long,
    private val offsetFrames: Long,
) {
    init {
        require(periodFrames > 0) { "period must be positive" }
        require(offsetFrames in 0 until periodFrames) { "offset must lie inside one period" }
    }

    fun outputFrameFor(emissionIndex: Long): Long = emissionIndex * periodFrames + offsetFrames

    fun firstEmissionAtOrAfter(earliestFrame: Long): PlannedChirpLabEmission {
        val shifted = earliestFrame - offsetFrames
        val index = if (shifted <= 0) 0 else (shifted + periodFrames - 1) / periodFrames
        return PlannedChirpLabEmission(index, outputFrameFor(index))
    }

    fun next(after: PlannedChirpLabEmission): PlannedChirpLabEmission =
        PlannedChirpLabEmission(after.emissionIndex + 1, outputFrameFor(after.emissionIndex + 1))
}
