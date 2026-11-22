package com.minimappr.field.capture

import android.content.Context
import android.media.AudioManager
import android.media.MediaRecorder

/**
 * Raw-capture source preference, best first.
 *
 * Adapted from the On Device Audio App (`preferredDetectorAudioSource` in
 * `audio/source/AndroidAudioCaptureSource.kt`, MIT, (c) 2026 Colin Catlin),
 * extended with the plan's middle step (field_mode_phone_mesh_plan.md,
 * Platform specifics): UNPROCESSED when advertised; else VOICE_RECOGNITION,
 * which the compatibility definition requires to have noise reduction and
 * AGC off by default; MIC last, because its voice-tuned noise suppression
 * can attenuate tonal bird whistles.
 */
fun fieldAudioSourceCandidates(context: Context): List<Int> {
    val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    val unprocessedCaptureSupported =
        audioManager.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED)
            ?.toBooleanStrictOrNull() == true
    return buildList {
        if (unprocessedCaptureSupported) add(MediaRecorder.AudioSource.UNPROCESSED)
        add(MediaRecorder.AudioSource.VOICE_RECOGNITION)
        add(MediaRecorder.AudioSource.MIC)
    }
}

fun audioSourceDisplayName(audioSource: Int): String = when (audioSource) {
    MediaRecorder.AudioSource.UNPROCESSED -> "UNPROCESSED"
    MediaRecorder.AudioSource.VOICE_RECOGNITION -> "VOICE_RECOGNITION"
    MediaRecorder.AudioSource.MIC -> "MIC"
    else -> "source#$audioSource"
}
