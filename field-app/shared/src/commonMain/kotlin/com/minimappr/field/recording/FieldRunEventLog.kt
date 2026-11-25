package com.minimappr.field.recording

import com.minimappr.field.json.JsonObjectBuilder

/**
 * Append-only run log (events.jsonl). Every line carries `t_ns`
 * (System.nanoTime at append) and `type`; the remaining keys are the
 * per-type contract the laptop analyser reads. Implementations must never
 * block the caller (capture and playback threads append).
 */
interface FieldRunEventLog {
    fun append(type: String, fields: JsonObjectBuilder.() -> Unit = {})
}

/** For runs without a recording (plain session, unit tests). */
object DiscardingFieldRunEventLog : FieldRunEventLog {
    override fun append(type: String, fields: JsonObjectBuilder.() -> Unit) = Unit
}

/** Event type names: one place, shared by writers and tests. */
object FieldRunEventTypes {
    const val EMISSION = "emission"
    const val CAPTURE_TIMESTAMP = "capture_timestamp"
    const val PLAYBACK_TIMESTAMP = "playback_timestamp"
    const val ATTITUDE = "attitude"
    const val SCREEN = "screen"
    const val ROUTE_CHANGE = "route_change"
    const val CAPTURE_GAP = "capture_gap"
    const val MARKER = "marker"
    const val THERMAL = "thermal"
    const val VOLUME = "volume"
    const val WRITER_OVERFLOW = "writer_overflow"
    const val SESSION_ROUND = "session_round"
    const val SELF_CHIRP = "self_chirp"
}
