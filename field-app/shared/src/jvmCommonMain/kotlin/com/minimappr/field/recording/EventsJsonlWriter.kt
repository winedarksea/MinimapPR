package com.minimappr.field.recording

import com.minimappr.field.json.JsonObjectBuilder
import kotlinx.atomicfu.atomic
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File

/**
 * events.jsonl writer: lines are built on the caller's thread (cheap) and
 * queued; an IO coroutine appends and flushes each line, so a killed run
 * loses at most what was still queued. Full queue = drop + count
 * (never block the capture or playback thread).
 */
class EventsJsonlWriter(
    file: File,
    queueCapacityLines: Int,
    scope: CoroutineScope,
    private val monotonicNanoTime: () -> Long = System::nanoTime,
) : FieldRunEventLog, AutoCloseable {
    private val writer = file.bufferedWriter()
    private val queue = Channel<String>(queueCapacityLines)
    private val droppedLineCounter = atomic(0L)
    private val writerJob: Job

    val droppedLineCount: Long get() = droppedLineCounter.value

    init {
        writerJob = scope.launch(Dispatchers.IO) {
            for (line in queue) {
                writer.write(line)
                writer.newLine()
                // Event rate is a few lines per second: flushing each keeps a
                // killed run's log complete at negligible cost.
                writer.flush()
            }
            writer.flush()
            writer.close()
        }
    }

    override fun append(type: String, fields: JsonObjectBuilder.() -> Unit) {
        val line = JsonObjectBuilder().apply {
            put("t_ns", monotonicNanoTime())
            put("type", type)
            fields()
        }.build()
        if (queue.trySend(line).isFailure) droppedLineCounter.incrementAndGet()
    }

    fun closeAndAwait(timeoutMilliseconds: Long) {
        queue.close()
        runBlocking { withTimeoutOrNull(timeoutMilliseconds) { writerJob.join() } }
    }

    override fun close() = closeAndAwait(DEFAULT_CLOSE_TIMEOUT_MILLISECONDS)

    private companion object {
        const val DEFAULT_CLOSE_TIMEOUT_MILLISECONDS = 5_000L
    }
}
