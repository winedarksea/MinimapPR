package com.minimappr.field.chirplab

import com.minimappr.field.capture.CaptureBlock
import com.minimappr.field.capture.CaptureBlockSink
import com.minimappr.field.capture.CaptureRingBuffer
import com.minimappr.field.capture.toPcm16LittleEndianBytes
import com.minimappr.field.ffi.FieldChirpDetection
import com.minimappr.field.ffi.FieldChirpDetector
import com.minimappr.field.playback.FrameTimeAnchor
import com.minimappr.field.playback.FrameTimeline
import kotlinx.atomicfu.locks.SynchronizedObject
import kotlinx.atomicfu.locks.synchronized
import kotlin.math.roundToLong

/** One self-chirp readout; `detection` null = not found in the window. */
data class SelfChirpReadout(
    val emissionIndex: Long,
    val predictedCaptureFrame: Long,
    val windowFirstDeviceSampleIndex: Long,
    val detection: FieldChirpDetection?,
) {
    /** Detected first-sample capture frame on the device timeline. */
    val detectedCaptureFrame: Double? get() = detection?.let { windowFirstDeviceSampleIndex + it.sampleIndex }
}

/**
 * Live self-SNR / clipping readout for chirp lab: for each emission, the
 * predicted presentation ns is mapped to a capture frame through the latest
 * AudioRecord timestamp anchor; once capture passes the window end, the
 * window is copied on the capture thread (cheap) and handed to
 * [detectionRunner] so the matched filter never runs on the capture thread.
 *
 * Not the measurement itself: the offline analyser re-detects from the WAV.
 */
class SelfChirpMonitor(
    private val detector: FieldChirpDetector,
    sampleRateHz: Int,
    ringSeconds: Int,
    private val searchMarginMilliseconds: Int,
    private val detectionRunner: (work: () -> Unit) -> Unit,
    private val onReadout: (SelfChirpReadout) -> Unit,
) : CaptureBlockSink {
    private val timeline = FrameTimeline(sampleRateHz)
    private val ring = CaptureRingBuffer(sampleRateHz * ringSeconds)
    private val chirpFrames = detector.templateFrameCount().toLong()
    private val marginFrames = timeline.framesForMilliseconds(searchMarginMilliseconds.toLong())
    private val lock = SynchronizedObject()
    private var latestCaptureAnchor: FrameTimeAnchor? = null
    private val pending = ArrayDeque<Pair<Long, Long>>()

    fun onCaptureAnchor(anchor: FrameTimeAnchor) = synchronized(lock) { latestCaptureAnchor = anchor }

    /**
     * Register an emission by predicted presentation ns. Returns false when no
     * capture anchor exists yet (nothing to map with; no blind fallback).
     */
    fun expectEmission(emissionIndex: Long, predictedPresentationNs: Long): Boolean = synchronized(lock) {
        val anchor = latestCaptureAnchor ?: return false
        val predictedFrame = timeline.frameAtNanoTime(anchor, predictedPresentationNs).roundToLong()
        pending.addLast(emissionIndex to predictedFrame)
        true
    }

    override fun onCaptureBlock(block: CaptureBlock) {
        ring.append(block)
        val ready = synchronized(lock) {
            val due = mutableListOf<Pair<Long, Long>>()
            while (pending.isNotEmpty() && pending.first().second + chirpFrames + marginFrames <= ring.endDeviceSampleIndex) {
                due += pending.removeFirst()
            }
            due
        }
        for ((emissionIndex, predictedFrame) in ready) {
            val windowFirst = predictedFrame - marginFrames
            val window = ring.copyWindow(windowFirst, predictedFrame + chirpFrames + marginFrames)
            if (window == null) {
                onReadout(SelfChirpReadout(emissionIndex, predictedFrame, windowFirst, null))
                continue
            }
            val searchLagCount = (window.size - chirpFrames + 1).toUInt()
            detectionRunner {
                val detection = detector.detect(window.toPcm16LittleEndianBytes(), 0u, searchLagCount, true, null)
                onReadout(SelfChirpReadout(emissionIndex, predictedFrame, windowFirst, detection))
            }
        }
    }
}
