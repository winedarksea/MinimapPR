package com.minimappr.field.android

import android.os.Bundle
import com.minimappr.field.FieldAppConfiguration
import com.minimappr.field.chirplab.ChirpLabCaptureSource
import com.minimappr.field.chirplab.ChirpLabRole
import com.minimappr.field.chirplab.ChirpLabRunParameters
import com.minimappr.field.chirplab.ChirpLabWireNames
import com.minimappr.field.ffi.FieldChirpWaveformSpec
import com.minimappr.field.node.FieldServiceMode
import com.minimappr.field.session.SessionLaunchParameters
import com.minimappr.field.session.SessionLaunchRole
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Launch-intent extras: the contract the laptop scripts use with
 * `adb shell am start ... --es mode chirp_lab|session ...`. Missing extras
 * take defaults; an intent without `mode` opens the normal UI.
 */
object FieldLaunchExtras {
    const val MODE = "mode"
    const val ROLE = "role"
    const val RUN_ID = "run_id"
    const val PERIOD_MS = "period_ms"
    const val WAVEFORM = "waveform"
    const val BAND_LOW_HZ = "band_low_hz"
    const val BAND_HIGH_HZ = "band_high_hz"
    const val DURATION_S = "duration_s"
    const val CHANNEL = "channel"
    const val SOURCE = "source"
    const val LEVEL_DBFS = "level_dbfs"
    const val LEAD_MS = "lead_ms"
    const val SESSION_ROLE = "session_role"
    const val HOST_ADDRESS = "host_address"
    const val SESSION_ID = "session_id"
    const val TEMPERATURE_C = "temperature_c"
    const val PERIODIC_MS = "periodic_ms"
    const val SHADOW_WAV = "shadow_wav"
}

sealed interface FieldLaunchRequest {
    val mode: FieldServiceMode

    data object Capture : FieldLaunchRequest {
        override val mode = FieldServiceMode.CAPTURE
    }

    data class ChirpLab(val parameters: ChirpLabRunParameters) : FieldLaunchRequest {
        override val mode = FieldServiceMode.CHIRP_LAB
    }

    data class Session(val parameters: SessionLaunchParameters) : FieldLaunchRequest {
        override val mode = FieldServiceMode.SESSION
    }
}

private const val RUN_ID_TIMESTAMP_PATTERN = "yyyyMMdd-HHmmss"

fun defaultRunId(prefix: String): String =
    "$prefix-${SimpleDateFormat(RUN_ID_TIMESTAMP_PATTERN, Locale.US).format(Date())}"

/** Reads a numeric extra whether adb sent it as --ei, --el, --ef, --ed or --es. */
@Suppress("DEPRECATION") // typed getters would reject the other adb extra kinds
private fun Bundle.number(key: String): Double? = when (val value = get(key)) {
    // --ef 0.05 arrives as a Float; via its decimal string it stays 0.05, not 0.0500000007.
    is Float -> value.toString().toDouble()
    is Number -> value.toDouble()
    is String -> value.toDoubleOrNull()
    else -> null
}

@Suppress("DEPRECATION")
private fun Bundle.flag(key: String): Boolean? = when (val value = get(key)) {
    is Boolean -> value
    is String -> value.toBooleanStrictOrNull()
    else -> null
}

/** Null when the bundle has no (known) mode: show the normal UI only. */
fun parseFieldLaunchRequest(extras: Bundle?, configuration: FieldAppConfiguration): FieldLaunchRequest? {
    if (extras == null) return null
    return when (FieldServiceMode.fromWireName(extras.getString(FieldLaunchExtras.MODE))) {
        FieldServiceMode.CAPTURE -> FieldLaunchRequest.Capture
        FieldServiceMode.CHIRP_LAB -> FieldLaunchRequest.ChirpLab(parseChirpLab(extras, configuration))
        FieldServiceMode.SESSION -> FieldLaunchRequest.Session(parseSession(extras, configuration))
        null -> null
    }
}

private fun parseChirpLab(extras: Bundle, configuration: FieldAppConfiguration): ChirpLabRunParameters {
    val role = ChirpLabRole.fromWireName(extras.getString(FieldLaunchExtras.ROLE)) ?: ChirpLabRole.A
    val defaults = ChirpLabRunParameters.defaults(configuration.chirpLab, defaultRunId("chirp-${role.wireName}"), role)
    return defaults.copy(
        runId = extras.getString(FieldLaunchExtras.RUN_ID) ?: defaults.runId,
        periodMilliseconds = extras.number(FieldLaunchExtras.PERIOD_MS)?.toInt() ?: defaults.periodMilliseconds,
        leadMilliseconds = extras.number(FieldLaunchExtras.LEAD_MS)?.toInt() ?: defaults.leadMilliseconds,
        waveform = FieldChirpWaveformSpec(
            kind = ChirpLabWireNames.waveformKind(extras.getString(FieldLaunchExtras.WAVEFORM)) ?: defaults.waveform.kind,
            bandLowHz = extras.number(FieldLaunchExtras.BAND_LOW_HZ) ?: defaults.waveform.bandLowHz,
            bandHighHz = extras.number(FieldLaunchExtras.BAND_HIGH_HZ) ?: defaults.waveform.bandHighHz,
            durationS = extras.number(FieldLaunchExtras.DURATION_S) ?: defaults.waveform.durationS,
        ),
        channel = ChirpLabWireNames.channel(extras.getString(FieldLaunchExtras.CHANNEL)) ?: defaults.channel,
        captureSource = ChirpLabCaptureSource.fromWireName(extras.getString(FieldLaunchExtras.SOURCE)) ?: defaults.captureSource,
        levelDbfs = extras.number(FieldLaunchExtras.LEVEL_DBFS) ?: defaults.levelDbfs,
    )
}

private fun parseSession(extras: Bundle, configuration: FieldAppConfiguration): SessionLaunchParameters {
    val role = SessionLaunchRole.fromWireName(extras.getString(FieldLaunchExtras.SESSION_ROLE)) ?: SessionLaunchRole.HOST
    return SessionLaunchParameters(
        role = role,
        hostAddress = extras.getString(FieldLaunchExtras.HOST_ADDRESS)?.takeIf { it.isNotBlank() },
        expectedSessionIdHex = extras.getString(FieldLaunchExtras.SESSION_ID),
        airTemperatureC = extras.number(FieldLaunchExtras.TEMPERATURE_C) ?: configuration.session.defaultAirTemperatureC,
        periodicRoundIntervalMilliseconds = extras.number(FieldLaunchExtras.PERIODIC_MS)?.toInt(),
        scheduleLeadTimeMilliseconds = extras.number(FieldLaunchExtras.LEAD_MS)?.toInt(),
        shadowRecording = extras.flag(FieldLaunchExtras.SHADOW_WAV) ?: false,
        runId = extras.getString(FieldLaunchExtras.RUN_ID) ?: defaultRunId("session-${role.wireName}"),
    )
}

/** Inverse of the parser for UI-started runs, so the service has one input path. */
fun FieldLaunchRequest.toExtras(): Bundle = Bundle().apply {
    putString(FieldLaunchExtras.MODE, mode.wireName)
    when (val request = this@toExtras) {
        FieldLaunchRequest.Capture -> Unit
        is FieldLaunchRequest.ChirpLab -> request.parameters.let { parameters ->
            putString(FieldLaunchExtras.ROLE, parameters.role.wireName)
            putString(FieldLaunchExtras.RUN_ID, parameters.runId)
            putInt(FieldLaunchExtras.PERIOD_MS, parameters.periodMilliseconds)
            putInt(FieldLaunchExtras.LEAD_MS, parameters.leadMilliseconds)
            putString(FieldLaunchExtras.WAVEFORM, ChirpLabWireNames.waveformKindName(parameters.waveform.kind))
            putDouble(FieldLaunchExtras.BAND_LOW_HZ, parameters.waveform.bandLowHz)
            putDouble(FieldLaunchExtras.BAND_HIGH_HZ, parameters.waveform.bandHighHz)
            putDouble(FieldLaunchExtras.DURATION_S, parameters.waveform.durationS)
            putString(FieldLaunchExtras.CHANNEL, ChirpLabWireNames.channelName(parameters.channel))
            putString(FieldLaunchExtras.SOURCE, parameters.captureSource.wireName)
            putDouble(FieldLaunchExtras.LEVEL_DBFS, parameters.levelDbfs)
        }
        is FieldLaunchRequest.Session -> request.parameters.let { parameters ->
            putString(FieldLaunchExtras.SESSION_ROLE, parameters.role.wireName)
            parameters.hostAddress?.let { putString(FieldLaunchExtras.HOST_ADDRESS, it) }
            parameters.expectedSessionIdHex?.let { putString(FieldLaunchExtras.SESSION_ID, it) }
            putDouble(FieldLaunchExtras.TEMPERATURE_C, parameters.airTemperatureC)
            parameters.periodicRoundIntervalMilliseconds?.let { putInt(FieldLaunchExtras.PERIODIC_MS, it) }
            parameters.scheduleLeadTimeMilliseconds?.let { putInt(FieldLaunchExtras.LEAD_MS, it) }
            putBoolean(FieldLaunchExtras.SHADOW_WAV, parameters.shadowRecording)
            putString(FieldLaunchExtras.RUN_ID, parameters.runId)
        }
    }
}
