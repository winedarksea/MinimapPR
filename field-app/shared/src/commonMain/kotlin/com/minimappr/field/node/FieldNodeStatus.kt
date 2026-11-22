package com.minimappr.field.node

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

/** Snapshot the UI renders; produced by [FieldNodeRuntime]. */
data class FieldNodeStatus(
    val ffiSurfaceVersion: Long = 0,
    val protocolVersionLabel: String = "",
    val capturedSampleCount: Long = 0,
    val captureGapCount: Long = 0,
    val stage1TriggersOpened: Long = 0,
    val stage1OnsetsDetected: Long = 0,
    val classifierWindowsAccepted: Long = 0,
    val classifierWindowsDropped: Long = 0,
    val outboundFramesSent: Long = 0,
    val lastOutboundMessageKindId: Int? = null,
    val lastUserNotice: String? = null,
)

enum class FieldCaptureServiceState { STOPPED, STARTING, RUNNING, FAILED }

data class FieldServiceStatus(
    val serviceState: FieldCaptureServiceState = FieldCaptureServiceState.STOPPED,
    val failureReason: String? = null,
    val node: FieldNodeStatus = FieldNodeStatus(),
)

/**
 * Process-wide status shared by the capture service (writer) and the UI
 * (reader). A StateFlow rather than a bound service: the UI only observes,
 * and the service outlives activity recreation.
 */
object FieldServiceStatusBoard {
    private val mutableStatus = MutableStateFlow(FieldServiceStatus())
    val status: StateFlow<FieldServiceStatus> = mutableStatus.asStateFlow()

    fun update(transform: (FieldServiceStatus) -> FieldServiceStatus) = mutableStatus.update(transform)
}
