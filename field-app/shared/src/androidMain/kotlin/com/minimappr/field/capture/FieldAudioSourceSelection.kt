package com.minimappr.field.capture

import android.content.Context
import android.media.AudioManager
import android.media.MediaRecorder
import com.minimappr.field.chirplab.ChirpLabCaptureSource

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
fun fieldAudioSourceCandidates(context: Context): List<Int> = buildList {
    if (unprocessedCaptureSupported(context)) add(MediaRecorder.AudioSource.UNPROCESSED)
    add(MediaRecorder.AudioSource.VOICE_RECOGNITION)
    add(MediaRecorder.AudioSource.MIC)
}

fun unprocessedCaptureSupported(context: Context): Boolean {
    val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    return audioManager.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED)?.toBooleanStrictOrNull() == true
}

/** Chirp lab `--es source`: null = the normal preference chain. */
fun ChirpLabCaptureSource.toForcedAudioSource(): Int? = when (this) {
    ChirpLabCaptureSource.DEFAULT -> null
    ChirpLabCaptureSource.UNPROCESSED -> MediaRecorder.AudioSource.UNPROCESSED
    ChirpLabCaptureSource.VOICE_RECOGNITION -> MediaRecorder.AudioSource.VOICE_RECOGNITION
    ChirpLabCaptureSource.MIC -> MediaRecorder.AudioSource.MIC
    ChirpLabCaptureSource.CAMCORDER -> MediaRecorder.AudioSource.CAMCORDER
}

fun audioSourceDisplayName(audioSource: Int): String = when (audioSource) {
    MediaRecorder.AudioSource.UNPROCESSED -> "UNPROCESSED"
    MediaRecorder.AudioSource.VOICE_RECOGNITION -> "VOICE_RECOGNITION"
    MediaRecorder.AudioSource.MIC -> "MIC"
    MediaRecorder.AudioSource.CAMCORDER -> "CAMCORDER"
    MediaRecorder.AudioSource.DEFAULT -> "DEFAULT"
    else -> "source#$audioSource"
}
