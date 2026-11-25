package com.minimappr.field.node

import com.minimappr.field.chirplab.ChirpLabStatus
import com.minimappr.field.session.FieldSessionStatus
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

enum class FieldCaptureServiceState { STOPPED, STARTING, RUNNING, FAILED }

/**
 * What the foreground service is doing. CAPTURE = solo listening node (a
 * one-phone host with no transport and no scheduled chirps); CHIRP_LAB =
 * Stage A recording run; SESSION = Stage B LAN session.
 */
enum class FieldServiceMode(val wireName: String) {
    CAPTURE("capture"),
    CHIRP_LAB("chirp_lab"),
    SESSION("session"),
    ;

    companion object {
        fun fromWireName(name: String?): FieldServiceMode? = entries.firstOrNull { it.wireName.equals(name, ignoreCase = true) }
    }
}

data class FieldServiceStatus(
    val serviceState: FieldCaptureServiceState = FieldCaptureServiceState.STOPPED,
    val mode: FieldServiceMode? = null,
    val failureReason: String? = null,
    val captureSourceName: String? = null,
    val chirpLab: ChirpLabStatus? = null,
    val session: FieldSessionStatus? = null,
)

/**
 * Process-wide status shared by the foreground service (writer) and the UI
 * and dumpsys (readers). A StateFlow rather than a bound service: the UI
 * only observes, and the service outlives activity recreation.
 */
object FieldServiceStatusBoard {
    private val mutableStatus = MutableStateFlow(FieldServiceStatus())
    val status: StateFlow<FieldServiceStatus> = mutableStatus.asStateFlow()

    fun update(transform: (FieldServiceStatus) -> FieldServiceStatus) = mutableStatus.update(transform)
}
