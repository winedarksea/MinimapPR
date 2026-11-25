package com.minimappr.field.recording

import com.minimappr.field.FieldChirpLabConfiguration
import com.minimappr.field.capture.CaptureBlock
import com.minimappr.field.capture.CaptureBlockSink
import com.minimappr.field.json.JsonObjectBuilder
import kotlinx.coroutines.CoroutineScope
import java.io.File

/** File names of one run bundle (`<chirp_lab dir>/<run_id>/`): the analyser contract. */
object FieldRunBundleFiles {
    const val CAPTURE_WAV = "capture.wav"
    const val SESSION_JSON = "session.json"
    const val EVENTS_JSONL = "events.jsonl"
}

/**
 * One run bundle on disk: capture.wav + events.jsonl + session.json. Used by
 * chirp lab and by session mode's optional shadow recording.
 */
class FieldRunRecording(
    val runDirectory: File,
    sampleRateHz: Int,
    configuration: FieldChirpLabConfiguration,
    scope: CoroutineScope,
) : FieldRunEventLog, CaptureBlockSink, AutoCloseable {
    private val events = EventsJsonlWriter(File(runDirectory.apply { mkdirs() }, FieldRunBundleFiles.EVENTS_JSONL), configuration.eventsWriterQueueCapacity, scope)
    private val wav = BoundedChannelWavWriter(
        File(runDirectory, FieldRunBundleFiles.CAPTURE_WAV),
        sampleRateHz,
        configuration.wavWriterQueueCapacityBlocks,
        configuration.wavHeaderRefreshIntervalBlocks,
        scope,
    ) { droppedSamples -> events.append(FieldRunEventTypes.WRITER_OVERFLOW) { put("dropped_samples", droppedSamples) } }

    val writerOverflowCount: Long get() = wav.overflowCount

    fun writeSessionJson(json: String) = File(runDirectory, FieldRunBundleFiles.SESSION_JSON).writeText(json)

    override fun append(type: String, fields: JsonObjectBuilder.() -> Unit) = events.append(type, fields)

    override fun onCaptureBlock(block: CaptureBlock) = wav.onCaptureBlock(block)

    override fun close() {
        wav.close()
        events.close()
    }
}
