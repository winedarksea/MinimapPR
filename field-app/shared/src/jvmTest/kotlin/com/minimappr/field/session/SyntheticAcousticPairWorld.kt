package com.minimappr.field.session

import com.minimappr.field.capture.CaptureBlock
import com.minimappr.field.capture.pcm16LittleEndianToShortArray
import com.minimappr.field.playback.ChirpPlaybackPort
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.math.roundToLong

private const val NOMINAL_SAMPLE_RATE_HZ = 48_000.0
private const val NANOSECONDS_PER_SECOND = 1e9
private const val PCM16_FULL_SCALE = 32_767.0
/** Emissions older than this cannot overlap new capture any more. */
private const val EMISSION_RETENTION_NS = 2_000_000_000L

/** One simulated phone: its monotonic epoch and when its capture started (true time). */
class SyntheticPhone(val monotonicOffsetNs: Long, val captureStartTrueNs: Long) {
    var nextSampleIndex = 0L
    lateinit var controller: FieldSessionController

    fun localNs(trueNs: Long) = trueNs + monotonicOffsetNs
}

private class SyntheticEmission(val emitterIndex: Int, val trueStartNs: Long, val template: FloatArray)

/**
 * Real-time acoustic world for the JVM loopback test, a simplified Kotlin
 * mirror of minimappr-ffi `tests/support/synthetic_acoustic_pair.rs`:
 * true time is the test's System.nanoTime, each phone sees it through its
 * own monotonic offset, and every chirp a device asks to play is rendered
 * into both capture streams (self path for the emitter, the separation for
 * the other phone) at the nearest 48 kHz sample. Audio crystals are ideal
 * here; the Rust test covers ppm drift.
 */
class SyntheticAcousticPairWorld(
    val phones: List<SyntheticPhone>,
    private val distanceM: Double,
    private val selfPathM: Double,
    private val soundSpeedMps: Double,
    private val trueNowNs: () -> Long,
) {
    private val emissions = CopyOnWriteArrayList<SyntheticEmission>()
    private var noiseState = 0x2545F4914F6CDD1DL

    fun playbackPort(phoneIndex: Int) = object : ChirpPlaybackPort {
        override fun scheduleChirpAtLocalNs(
            emissionId: Long,
            targetPresentationNs: Long,
            pcm16StereoLittleEndian: ByteArray,
            outcomeListener: com.minimappr.field.playback.ChirpEmissionOutcomeListener,
        ) {
            val trueStart = targetPresentationNs - phones[phoneIndex].monotonicOffsetNs
            if (trueStart <= trueNowNs()) {
                outcomeListener.onChirpEmissionOutcome(false, targetPresentationNs)
                return
            }
            emissions += SyntheticEmission(phoneIndex, trueStart, loudestChannel(pcm16StereoLittleEndian))
            outcomeListener.onChirpEmissionOutcome(true, targetPresentationNs)
        }
    }

    private fun loudestChannel(stereoBytes: ByteArray): FloatArray {
        val stereo = stereoBytes.pcm16LittleEndianToShortArray()
        val frames = stereo.size / 2
        val leftEnergy = (0 until frames).sumOf { stereo[it * 2].toDouble() * stereo[it * 2] }
        val rightEnergy = (0 until frames).sumOf { stereo[it * 2 + 1].toDouble() * stereo[it * 2 + 1] }
        val channel = if (leftEnergy >= rightEnergy) 0 else 1
        return FloatArray(frames) { (stereo[it * 2 + channel] / PCM16_FULL_SCALE).toFloat() }
    }

    private fun noise(): Double {
        noiseState = noiseState xor (noiseState shl 13)
        noiseState = noiseState xor (noiseState ushr 7)
        noiseState = noiseState xor (noiseState shl 17)
        return ((noiseState ushr 40).toDouble() / (1L shl 24) - 0.5) * NOISE_PEAK_TO_PEAK
    }

    /** Capture every phone up to now, push the block and an exact capture timestamp. */
    fun captureUpToNow() {
        val now = trueNowNs()
        emissions.removeAll { now - it.trueStartNs > EMISSION_RETENTION_NS }
        for ((phoneIndex, phone) in phones.withIndex()) {
            if (now < phone.captureStartTrueNs) continue
            val endIndex = ((now - phone.captureStartTrueNs) * NOMINAL_SAMPLE_RATE_HZ / NANOSECONDS_PER_SECOND).toLong()
            val first = phone.nextSampleIndex
            if (endIndex <= first) continue
            val block = ShortArray((endIndex - first).toInt()) { offset ->
                val trueSampleNs = phone.captureStartTrueNs + ((first + offset) * NANOSECONDS_PER_SECOND / NOMINAL_SAMPLE_RATE_HZ).roundToLong()
                val value = renderSample(phoneIndex, trueSampleNs) + noise()
                (value.coerceIn(-1.0, 1.0) * PCM16_FULL_SCALE).toInt().toShort()
            }
            phone.controller.onCaptureBlock(CaptureBlock(first, block))
            phone.nextSampleIndex = endIndex
            val anchorTrueNs = phone.captureStartTrueNs + (endIndex * NANOSECONDS_PER_SECOND / NOMINAL_SAMPLE_RATE_HZ).roundToLong()
            phone.controller.onCaptureTimestamp(endIndex, phone.localNs(anchorTrueNs))
        }
    }

    private fun renderSample(phoneIndex: Int, trueSampleNs: Long): Double {
        var value = 0.0
        for (emission in emissions) {
            val isSelf = emission.emitterIndex == phoneIndex
            val pathM = if (isSelf) selfPathM else distanceM
            val offsetNs = trueSampleNs - emission.trueStartNs - pathM / soundSpeedMps * NANOSECONDS_PER_SECOND
            val templateIndex = (offsetNs * NOMINAL_SAMPLE_RATE_HZ / NANOSECONDS_PER_SECOND).roundToLong()
            if (templateIndex < 0 || templateIndex >= emission.template.size) continue
            value += (if (isSelf) SELF_PATH_GAIN else REMOTE_PATH_GAIN) * emission.template[templateIndex.toInt()]
        }
        return value
    }

    private companion object {
        const val SELF_PATH_GAIN = 0.5
        const val REMOTE_PATH_GAIN = 0.1
        const val NOISE_PEAK_TO_PEAK = 0.002
    }
}
