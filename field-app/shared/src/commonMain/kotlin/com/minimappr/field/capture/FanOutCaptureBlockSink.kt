package com.minimappr.field.capture

/**
 * Hands every capture block to each sink in order (rolling buffer / device
 * first, then optional recorders). Each sink must itself be non-blocking:
 * the WAV writer enqueues, it never writes on the capture thread.
 */
class FanOutCaptureBlockSink(private val sinks: List<CaptureBlockSink>) : CaptureBlockSink {
    override fun onCaptureBlock(block: CaptureBlock) {
        for (sink in sinks) sink.onCaptureBlock(block)
    }
}

/**
 * A failed or short read. `expectedFrame` is where the next block should
 * have started; `gotFrame` where it now starts (the counter's jump);
 * `missingFrames` is a lower bound (the OS does not say how much it lost).
 */
data class CaptureGapReport(val expectedFrame: Long, val gotFrame: Long, val missingFrames: Long)
