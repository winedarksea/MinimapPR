package com.minimappr.field.recording

import com.minimappr.field.capture.CaptureBlock
import com.minimappr.field.capture.CaptureBlockSink
import kotlinx.atomicfu.atomic
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File
import java.io.RandomAccessFile

private const val WAV_HEADER_LENGTH_BYTES = 44
private const val PCM16_BYTES_PER_SAMPLE = 2
private const val MONO_CHANNEL_COUNT = 1
private const val PCM_FORMAT_TAG = 1
private const val FMT_CHUNK_LENGTH_BYTES = 16
private const val BITS_PER_SAMPLE = 16
private const val RIFF_SIZE_FIELD_OFFSET = 4
private const val DATA_SIZE_FIELD_OFFSET = 40
private const val RIFF_SIZE_EXCLUDED_BYTES = 8
private const val BYTE_MASK = 0xFF
private const val BITS_PER_BYTE = 8

/**
 * Mono PCM16 WAV recorder fed from the capture thread through a bounded
 * channel; an IO coroutine does the disk writes, so capture never blocks.
 *
 * Timeline contract (the analyser relies on it): sample n of the file is
 * device sample index n (AudioRecord frame n since startRecording). A block
 * dropped on overflow, or a hole from a failed read, is written as zeros,
 * so later samples keep their index. Overflow is reported through
 * [onOverflow] (a `writer_overflow` event).
 *
 * The header is patched every [headerRefreshIntervalBlocks] and on close;
 * a killed process leaves a slightly short header, so the analyser trusts
 * the file length.
 */
class BoundedChannelWavWriter(
    file: File,
    private val sampleRateHz: Int,
    queueCapacityBlocks: Int,
    private val headerRefreshIntervalBlocks: Int,
    scope: CoroutineScope,
    private val onOverflow: (droppedSamples: Long) -> Unit,
) : CaptureBlockSink, AutoCloseable {
    private val output = RandomAccessFile(file, "rw").apply { setLength(0) }
    private val queue = Channel<CaptureBlock>(queueCapacityBlocks)
    private val droppedSampleCounter = atomic(0L)
    private val overflowEventCounter = atomic(0L)
    private var nextFileSampleIndex = 0L
    private var blocksSinceHeaderRefresh = 0
    private val writerJob: Job

    val droppedSampleCount: Long get() = droppedSampleCounter.value
    val overflowCount: Long get() = overflowEventCounter.value

    init {
        output.write(wavHeader(dataBytes = 0))
        writerJob = scope.launch(Dispatchers.IO) {
            for (block in queue) writeBlock(block)
            patchHeader()
            output.close()
        }
    }

    override fun onCaptureBlock(block: CaptureBlock) {
        if (queue.trySend(block).isFailure) {
            droppedSampleCounter.addAndGet(block.samples.size.toLong())
            overflowEventCounter.incrementAndGet()
            onOverflow(block.samples.size.toLong())
        }
    }

    private fun writeBlock(block: CaptureBlock) {
        val gapSamples = block.firstDeviceSampleIndex - nextFileSampleIndex
        if (gapSamples > 0) {
            output.write(ByteArray((gapSamples * PCM16_BYTES_PER_SAMPLE).toInt()))
            nextFileSampleIndex += gapSamples
        }
        // An overlapping block (never expected) is trimmed, not rewritten.
        val skip = (nextFileSampleIndex - block.firstDeviceSampleIndex).coerceAtLeast(0).toInt()
        if (skip >= block.samples.size) return
        val bytes = ByteArray((block.samples.size - skip) * PCM16_BYTES_PER_SAMPLE)
        for (index in skip until block.samples.size) {
            val sample = block.samples[index].toInt()
            val byteOffset = (index - skip) * PCM16_BYTES_PER_SAMPLE
            bytes[byteOffset] = (sample and BYTE_MASK).toByte()
            bytes[byteOffset + 1] = ((sample shr BITS_PER_BYTE) and BYTE_MASK).toByte()
        }
        output.write(bytes)
        nextFileSampleIndex += block.samples.size - skip
        blocksSinceHeaderRefresh += 1
        if (blocksSinceHeaderRefresh >= headerRefreshIntervalBlocks) {
            blocksSinceHeaderRefresh = 0
            patchHeader()
        }
    }

    private fun patchHeader() {
        val dataBytes = nextFileSampleIndex * PCM16_BYTES_PER_SAMPLE
        val end = output.filePointer
        output.seek(RIFF_SIZE_FIELD_OFFSET.toLong())
        output.write(littleEndianInt((dataBytes + WAV_HEADER_LENGTH_BYTES - RIFF_SIZE_EXCLUDED_BYTES).toInt()))
        output.seek(DATA_SIZE_FIELD_OFFSET.toLong())
        output.write(littleEndianInt(dataBytes.toInt()))
        output.seek(end)
    }

    private fun wavHeader(dataBytes: Int): ByteArray {
        val byteRate = sampleRateHz * MONO_CHANNEL_COUNT * PCM16_BYTES_PER_SAMPLE
        return "RIFF".encodeToByteArray() + littleEndianInt(dataBytes + WAV_HEADER_LENGTH_BYTES - RIFF_SIZE_EXCLUDED_BYTES) +
            "WAVE".encodeToByteArray() + "fmt ".encodeToByteArray() + littleEndianInt(FMT_CHUNK_LENGTH_BYTES) +
            littleEndianShort(PCM_FORMAT_TAG) + littleEndianShort(MONO_CHANNEL_COUNT) + littleEndianInt(sampleRateHz) +
            littleEndianInt(byteRate) + littleEndianShort(MONO_CHANNEL_COUNT * PCM16_BYTES_PER_SAMPLE) +
            littleEndianShort(BITS_PER_SAMPLE) + "data".encodeToByteArray() + littleEndianInt(dataBytes)
    }

    /** Drains what is queued, finalises the header; bounded wait. */
    fun closeAndAwait(timeoutMilliseconds: Long) {
        queue.close()
        runBlocking { withTimeoutOrNull(timeoutMilliseconds) { writerJob.join() } }
    }

    override fun close() = closeAndAwait(DEFAULT_CLOSE_TIMEOUT_MILLISECONDS)

    private companion object {
        const val DEFAULT_CLOSE_TIMEOUT_MILLISECONDS = 5_000L
    }
}

private fun littleEndianInt(value: Int) = byteArrayOf(
    (value and BYTE_MASK).toByte(),
    ((value shr BITS_PER_BYTE) and BYTE_MASK).toByte(),
    ((value shr (2 * BITS_PER_BYTE)) and BYTE_MASK).toByte(),
    ((value shr (3 * BITS_PER_BYTE)) and BYTE_MASK).toByte(),
)

private fun littleEndianShort(value: Int) = byteArrayOf((value and BYTE_MASK).toByte(), ((value shr BITS_PER_BYTE) and BYTE_MASK).toByte())
