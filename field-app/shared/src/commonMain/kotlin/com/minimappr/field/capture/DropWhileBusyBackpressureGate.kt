package com.minimappr.field.capture

import kotlinx.atomicfu.atomic

/**
 * Drop-a-window-and-count-it backpressure for expensive stages (classifier).
 *
 * Adapted from the On Device Audio App (`InferenceController.handleFrame`,
 * MIT, (c) 2026 Colin Catlin): while one window is being processed, newer
 * windows are dropped and counted instead of queued, so the stage never
 * falls behind real time. Only the in-flight flag and drop counter are taken;
 * the audio app's RMS energy gate is deliberately NOT copied (AGENTS.md:
 * faint long-range calls must not be blocked by absolute level gates; Stage 1
 * in minimappr-core is relative to the noise floor instead).
 *
 * Never put this in front of the rolling buffer: capture blocks must all
 * reach it or clip timing breaks. It guards only work that may be skipped.
 */
class DropWhileBusyBackpressureGate {
    private val windowInFlight = atomic(false)
    private val acceptedWindowCounter = atomic(0L)
    private val droppedWindowCounter = atomic(0L)

    val acceptedWindowCount: Long get() = acceptedWindowCounter.value
    val droppedWindowCount: Long get() = droppedWindowCounter.value

    /** True when the caller owns the stage and must call [release]. */
    fun tryAcquire(): Boolean {
        if (windowInFlight.compareAndSet(expect = false, update = true)) {
            acceptedWindowCounter.incrementAndGet()
            return true
        }
        droppedWindowCounter.incrementAndGet()
        return false
    }

    fun release() {
        check(windowInFlight.compareAndSet(expect = true, update = false)) { "release without acquire" }
    }
}
