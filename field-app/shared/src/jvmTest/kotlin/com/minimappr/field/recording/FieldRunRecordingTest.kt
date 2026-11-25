package com.minimappr.field.recording

import com.minimappr.field.FieldChirpLabConfiguration
import com.minimappr.field.capture.CaptureBlock
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FieldRunRecordingTest {
    private val sampleRateHz = 48_000
    private val wavHeaderBytes = 44

    @Test
    fun wavKeepsTheDeviceTimelineByZeroFillingHolesAndFinalisesTheHeader() {
        val directory = Files.createTempDirectory("chirp_lab_test").toFile()
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val recording = FieldRunRecording(File(directory, "run1"), sampleRateHz, FieldChirpLabConfiguration(), scope)
        recording.onCaptureBlock(CaptureBlock(0, shortArrayOf(1, 2, 3)))
        // A failed read jumped the counter by 2: samples 3 and 4 are zeros in the file.
        recording.onCaptureBlock(CaptureBlock(5, shortArrayOf(-7, 8)))
        recording.append(FieldRunEventTypes.MARKER) { put("label", "foam \"left\"") }
        recording.writeSessionJson("{\"run_id\":\"run1\"}")
        recording.close()
        scope.cancel()

        val wav = File(directory, "run1/${FieldRunBundleFiles.CAPTURE_WAV}").readBytes()
        assertEquals(wavHeaderBytes + 7 * 2, wav.size)
        val header = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals("RIFF", String(wav, 0, 4))
        assertEquals(sampleRateHz, header.getInt(24))
        assertEquals(14, header.getInt(40), "data chunk size")
        val samples = (0 until 7).map { header.getShort(wavHeaderBytes + it * 2).toInt() }
        assertEquals(listOf(1, 2, 3, 0, 0, -7, 8), samples)

        val line = File(directory, "run1/${FieldRunBundleFiles.EVENTS_JSONL}").readLines().single()
        assertTrue(line.startsWith("{\"t_ns\":"), line)
        assertTrue(line.contains("\"type\":\"marker\",\"label\":\"foam \\\"left\\\"\""), line)
        assertTrue(File(directory, "run1/${FieldRunBundleFiles.SESSION_JSON}").readText().contains("run1"))
        directory.deleteRecursively()
    }
}
