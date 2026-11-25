package com.minimappr.field.session

import com.minimappr.field.ffi.FieldChirpScheduleMode
import com.minimappr.field.ffi.FieldMeshTuning
import com.minimappr.field.ffi.defaultFieldMeshTuning

enum class SessionLaunchRole(val wireName: String) {
    HOST("host"),
    JOIN("join"),

    /** CAPTURE mode: a one-phone host with no transport and no scheduled chirps. */
    SOLO_CAPTURE("solo"),
    ;

    companion object {
        fun fromWireName(name: String?): SessionLaunchRole? = entries.firstOrNull { it.wireName.equals(name, ignoreCase = true) }
    }
}

/**
 * Session launch (am start `--es mode session ...` or the Session screen).
 * Null overrides keep the FFI defaults (`default_field_mesh_tuning`); both
 * phones must use the same tuning because the schedule carries only the band.
 */
data class SessionLaunchParameters(
    val role: SessionLaunchRole,
    /** "ip:port" typed or from the launch intent; null = NSD discovery. */
    val hostAddress: String? = null,
    /** From NSD TXT `sid`: the joiner insists on this session id. */
    val expectedSessionIdHex: String? = null,
    val airTemperatureC: Double,
    val periodicRoundIntervalMilliseconds: Int? = null,
    val scheduleLeadTimeMilliseconds: Int? = null,
    val shadowRecording: Boolean = false,
    val runId: String,
) {
    fun meshTuning(): FieldMeshTuning {
        val defaults = defaultFieldMeshTuning()
        return defaults.copy(
            scheduleMode = if (role == SessionLaunchRole.SOLO_CAPTURE) FieldChirpScheduleMode.SILENT_UNTIL_NEEDED else defaults.scheduleMode,
            periodicRoundIntervalMs = periodicRoundIntervalMilliseconds?.toULong() ?: defaults.periodicRoundIntervalMs,
            scheduleLeadTimeMs = scheduleLeadTimeMilliseconds?.toULong() ?: defaults.scheduleLeadTimeMs,
            airTemperatureC = airTemperatureC,
        )
    }
}
