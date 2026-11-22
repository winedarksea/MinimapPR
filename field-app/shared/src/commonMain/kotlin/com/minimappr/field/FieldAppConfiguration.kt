package com.minimappr.field

/**
 * Operational constants for the shell, in one place (AGENTS.md: no magic
 * numbers). Mesh/DSP tunables live in minimappr-core, not here.
 */
data class FieldCaptureConfiguration(
    /** Device clock rate every TDOA computation in minimappr-core assumes. */
    val captureSampleRateHz: Int = 48_000,
    /** 20 ms blocks: one Stage-1 hop per read. */
    val captureBlockSampleCount: Int = 960,
    /**
     * OS capture buffer, in blocks. Large enough to ride out a GC pause on the
     * capture thread without an overrun (which would be a TDOA gap).
     */
    val audioRecordBufferBlockCount: Int = 16,
    /**
     * How long stopping capture waits for the read thread to exit before
     * releasing the recorder anyway. AudioRecord.stop() unblocks a blocking
     * read, so this is normally one 20 ms block; the bound keeps a wedged
     * audio HAL from hanging the main thread into an ANR.
     */
    val captureStopJoinTimeoutMilliseconds: Long = 500,
)

data class FieldNodeConfiguration(
    /** Plan: about 30 s of PCM kept for clip requests. */
    val rollingBufferSeconds: Int = 30,
    /** Coarse session-clock tick driving heartbeats, joins and liveness. */
    val stateMachineTickIntervalMilliseconds: Long = 250,
)

data class FieldAppConfiguration(
    val capture: FieldCaptureConfiguration = FieldCaptureConfiguration(),
    val node: FieldNodeConfiguration = FieldNodeConfiguration(),
)
