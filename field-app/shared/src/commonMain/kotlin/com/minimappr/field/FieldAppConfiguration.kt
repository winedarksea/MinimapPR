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
)

data class FieldNodeConfiguration(
    /** Plan: about 30 s of PCM kept for clip requests. */
    val rollingBufferSeconds: Int = 30,
    /** Coarse session-clock tick driving heartbeats, joins and liveness. */
    val stateMachineTickIntervalMilliseconds: Long = 250,
    /** Recent frames kept for the status UI (debug only, bounded). */
    val recentOutboundFrameHistoryLimit: Int = 32,
)

data class FieldAppConfiguration(
    val capture: FieldCaptureConfiguration = FieldCaptureConfiguration(),
    val node: FieldNodeConfiguration = FieldNodeConfiguration(),
)
