package com.minimappr.field.audio

import android.media.MicrophoneInfo
import com.minimappr.field.json.JsonObjectBuilder

/** One MicrophoneInfo as session.json and `active_microphones` events record it. */
fun JsonObjectBuilder.putMicrophoneInfo(microphone: MicrophoneInfo) {
    put("id", microphone.id)
    put("description", microphone.description)
    put("type", microphone.type)
    put("address", microphone.address)
    put("location", microphone.location)
    put("directionality", microphone.directionality)
    val position = microphone.position
    putNumbers("position_m", listOf(position.x.toDouble(), position.y.toDouble(), position.z.toDouble()))
    val orientation = microphone.orientation
    putNumbers("orientation", listOf(orientation.x.toDouble(), orientation.y.toDouble(), orientation.z.toDouble()))
    put("sensitivity_dbfs_at_94db_spl", microphone.sensitivity)
}
