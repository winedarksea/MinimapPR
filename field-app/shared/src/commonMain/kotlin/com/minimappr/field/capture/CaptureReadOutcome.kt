package com.minimappr.field.capture

/** What the capture loop does with one blocking read's return value. */
enum class CaptureReadOutcome {
    /** Samples arrived: claim indices and hand the block on. */
    DELIVER_BLOCK,

    /** Nothing arrived but the recorder is alive: flag a possible gap, keep reading. */
    FLAG_GAP_AND_CONTINUE,

    /**
     * A negative status (ERROR_DEAD_OBJECT, ERROR_INVALID_OPERATION, ...).
     * The recorder will keep returning it, so retrying would spin a core and
     * flood the node with gap reports; the loop stops and the service fails
     * visibly instead.
     */
    STOP_CAPTURE_WITH_FAILURE,
}

/** Classifies `AudioRecord.read` (or AAudio read) return values. */
fun classifyCaptureReadResult(samplesReadOrErrorCode: Int): CaptureReadOutcome = when {
    samplesReadOrErrorCode > 0 -> CaptureReadOutcome.DELIVER_BLOCK
    samplesReadOrErrorCode == 0 -> CaptureReadOutcome.FLAG_GAP_AND_CONTINUE
    else -> CaptureReadOutcome.STOP_CAPTURE_WITH_FAILURE
}

/** Thrown out of the capture coroutine so the service's handler fails it. */
class CaptureReadFailedException(val readErrorCode: Int) :
    RuntimeException("audio capture read failed with code $readErrorCode")
