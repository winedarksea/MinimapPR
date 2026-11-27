package com.minimappr.field.audio

import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioRecordingConfiguration
import android.media.audiofx.AudioEffect
import com.minimappr.field.chirplab.describeAudioDevice
import com.minimappr.field.json.JsonObjectBuilder
import com.minimappr.field.recording.FieldRunEventLog
import com.minimappr.field.recording.FieldRunEventTypes
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

private const val REASON_START = "start"
private const val REASON_CALLBACK = "callback"
private const val REASON_PERIODIC = "periodic"

/**
 * Logs which microphones and pre-processing effects AudioRecord actually
 * runs (`active_microphones`): at start, on every AudioRecordingCallback
 * and periodically. Tests the hypothesis that the Pixel 10 switches its
 * capture mic (or adds echo processing) while its bottom speaker plays,
 * which would put a second copy of the other phone's chirp into capture.
 *
 * Callbacks and the binder queries run on a private single thread, never
 * on the capture or playback threads.
 */
class AudioRecordingPathMonitor(
    private val audioRecordProvider: () -> AudioRecord?,
    private val eventLog: FieldRunEventLog,
    private val periodicIntervalMilliseconds: Long,
) {
    private var executor: ExecutorService? = null
    private var periodicJob: Job? = null
    private var registeredRecord: AudioRecord? = null

    private val callback = object : AudioManager.AudioRecordingCallback() {
        override fun onRecordingConfigChanged(configs: MutableList<AudioRecordingConfiguration>?) {
            log(REASON_CALLBACK, configs?.firstOrNull())
        }
    }

    fun start(scope: CoroutineScope) {
        val record = audioRecordProvider() ?: return
        val callbackExecutor = Executors.newSingleThreadExecutor { runnable -> Thread(runnable, "field-recording-path") }
        executor = callbackExecutor
        record.registerAudioRecordingCallback(callbackExecutor, callback)
        registeredRecord = record
        callbackExecutor.execute { log(REASON_START, record.activeRecordingConfiguration) }
        periodicJob = scope.launch(Dispatchers.Default) {
            while (isActive) {
                delay(periodicIntervalMilliseconds)
                executor?.execute { log(REASON_PERIODIC, audioRecordProvider()?.activeRecordingConfiguration) }
            }
        }
    }

    private fun log(reason: String, configuration: AudioRecordingConfiguration?) {
        val record = audioRecordProvider() ?: return
        val microphones = runCatching { record.activeMicrophones }.getOrNull().orEmpty()
        eventLog.append(FieldRunEventTypes.ACTIVE_MICROPHONES) {
            put("reason", reason)
            putObjects("microphones", microphones.map { microphone -> { putMicrophoneInfo(microphone) } })
            put("routed_input", describeAudioDevice(configuration?.audioDevice ?: record.routedDevice))
            put("client_audio_source", configuration?.clientAudioSource)
            put("client_silenced", configuration?.isClientSilenced)
            putObjects("client_effects", configuration?.clientEffects.orEmpty().map { effect -> { putEffect(effect) } })
            putObjects("device_effects", configuration?.effects.orEmpty().map { effect -> { putEffect(effect) } })
        }
    }

    private fun JsonObjectBuilder.putEffect(effect: AudioEffect.Descriptor) {
        put("name", effect.name)
        put("type", effect.type.toString())
        put("implementor", effect.implementor)
    }

    fun stop() {
        periodicJob?.cancel()
        periodicJob = null
        registeredRecord?.let { runCatching { it.unregisterAudioRecordingCallback(callback) } }
        registeredRecord = null
        executor?.shutdown()
        executor = null
    }
}
