package com.minimappr.field.chirplab

import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.media.MicrophoneInfo
import android.os.Build
import com.minimappr.field.capture.unprocessedCaptureSupported
import com.minimappr.field.json.JsonObjectBuilder
import com.minimappr.field.json.jsonObject

/**
 * `platform_name` used by the per-model config tables: Build.MODEL
 * lowercased with spaces → underscores (`pixel_7a`, `sm-a037f`).
 */
fun fieldPlatformName(): String = Build.MODEL.lowercase().replace(' ', '_')

/** What the run knows about its capture and playback paths when it starts. */
data class ChirpLabAudioPathFacts(
    val requestedSource: String,
    val actualSource: String?,
    val captureBufferSizeFrames: Int,
    val captureSampleRateHz: Int,
    val playbackBufferSizeFrames: Int,
    val playbackPerformanceMode: String,
    val lowLatencyRequested: Boolean,
    val routedInput: String?,
    val routedOutput: String?,
)

/**
 * session.json for a run bundle (analyser contract): device identity,
 * schedule, capture/playback paths, active microphones, UNPROCESSED and
 * near-ultrasound support, routing and media volume.
 */
class AndroidDeviceSurvey(private val context: Context) {
    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

    fun musicVolumeIndex(): Int = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
    fun musicVolumeMaximum(): Int = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)

    fun sessionJson(
        runId: String,
        role: String,
        mode: String,
        schedule: (JsonObjectBuilder.() -> Unit)?,
        paths: ChirpLabAudioPathFacts,
        activeMicrophones: List<MicrophoneInfo>,
        startedNanoTime: Long,
        startedWallClockMilliseconds: Long,
    ): String = jsonObject {
        put("run_id", runId)
        put("role", role)
        put("mode", mode)
        put("platform_name", fieldPlatformName())
        putObject("device") {
            put("manufacturer", Build.MANUFACTURER)
            put("model", Build.MODEL)
            put("sdk_int", Build.VERSION.SDK_INT)
            put("fingerprint", Build.FINGERPRINT)
        }
        if (schedule != null) putObject("schedule", schedule) else put("schedule", null as String?)
        putObject("capture") {
            put("requested_source", paths.requestedSource)
            put("actual_source", paths.actualSource)
            put("buffer_size_frames", paths.captureBufferSizeFrames)
            put("sample_rate_hz", paths.captureSampleRateHz)
        }
        putObject("playback") {
            put("buffer_size_frames", paths.playbackBufferSizeFrames)
            put("performance_mode", paths.playbackPerformanceMode)
            put("low_latency", paths.lowLatencyRequested)
        }
        putObjects("active_microphones", activeMicrophones.map { microphone -> { microphoneJson(microphone) } })
        put("unprocessed_supported", unprocessedCaptureSupported(context))
        putObject("near_ultrasound") {
            put("mic", audioManager.getProperty(AudioManager.PROPERTY_SUPPORT_MIC_NEAR_ULTRASOUND)?.toBooleanStrictOrNull())
            put("speaker", audioManager.getProperty(AudioManager.PROPERTY_SUPPORT_SPEAKER_NEAR_ULTRASOUND)?.toBooleanStrictOrNull())
        }
        put("routed_input", paths.routedInput)
        put("routed_output", paths.routedOutput)
        putObject("volume") {
            put("music_index", musicVolumeIndex())
            put("music_max", musicVolumeMaximum())
        }
        put("started_ns", startedNanoTime)
        put("started_wall_ms", startedWallClockMilliseconds)
    }

    private fun JsonObjectBuilder.microphoneJson(microphone: MicrophoneInfo) {
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
}

fun describeAudioDevice(device: AudioDeviceInfo?): String? = device?.let { "${audioDeviceTypeName(it.type)}:${it.productName}" }

private fun audioDeviceTypeName(type: Int): String = when (type) {
    AudioDeviceInfo.TYPE_BUILTIN_MIC -> "builtin_mic"
    AudioDeviceInfo.TYPE_BUILTIN_SPEAKER -> "builtin_speaker"
    AudioDeviceInfo.TYPE_BUILTIN_EARPIECE -> "builtin_earpiece"
    AudioDeviceInfo.TYPE_WIRED_HEADSET -> "wired_headset"
    AudioDeviceInfo.TYPE_WIRED_HEADPHONES -> "wired_headphones"
    AudioDeviceInfo.TYPE_BLUETOOTH_A2DP -> "bluetooth_a2dp"
    AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> "bluetooth_sco"
    AudioDeviceInfo.TYPE_USB_DEVICE, AudioDeviceInfo.TYPE_USB_HEADSET -> "usb"
    else -> "type$type"
}
