package com.minimappr.field.android

import android.Manifest
import android.content.Context
import android.media.AudioManager
import android.os.SystemClock
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.minimappr.field.FieldAppConfiguration
import com.minimappr.field.chirplab.ChirpLabController
import com.minimappr.field.chirplab.ChirpLabRole
import com.minimappr.field.chirplab.ChirpLabRunParameters
import com.minimappr.field.chirplab.ChirpLabStatus
import com.minimappr.field.recording.FieldRunBundleFiles
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * One phone, real speaker → real mic: a short chirp lab run must hear its
 * own chirps with at least the configured self SNR and without clipping.
 * Media volume is set to a fixed fraction for the test and restored.
 */
@RunWith(AndroidJUnit4::class)
class ChirpLoopbackInstrumentedTest {
    private val configuration = FieldAppConfiguration()
    private val testPeriodMilliseconds = 1_000
    private val requiredSelfChirps = 4L
    private val runTimeoutMilliseconds = 15_000L
    private val pollIntervalMilliseconds = 100L
    private val testVolumeFraction = 0.6
    private val wavHeaderBytes = 44L
    private val pcm16BytesPerSample = 2L

    @get:Rule
    val recordAudioPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.RECORD_AUDIO)

    @Test
    fun selfChirpsAreHeardAboveTheMinimumSnrWithoutClipping() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val originalVolume = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
        val testVolume = (audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC) * testVolumeFraction).toInt()
        audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, testVolume, 0)
        val outputRoot = File(context.cacheDir, "chirp_loopback_test").apply { deleteRecursively() }
        val parameters = ChirpLabRunParameters.defaults(configuration.chirpLab, "loopback", ChirpLabRole.A)
            .copy(periodMilliseconds = testPeriodMilliseconds)
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        // A visible activity keeps the app foreground: while-in-use mic access.
        ActivityScenario.launch(FieldMainActivity::class.java).use {
            val controller = ChirpLabController(context, configuration, parameters, outputRoot)
            var status: ChirpLabStatus
            try {
                runBlocking { controller.start(scope) }
                val deadline = SystemClock.elapsedRealtime() + runTimeoutMilliseconds
                do {
                    SystemClock.sleep(pollIntervalMilliseconds)
                    status = controller.status.value
                } while (status.selfChirpsDetected < requiredSelfChirps && SystemClock.elapsedRealtime() < deadline)
            } finally {
                controller.stop()
                scope.cancel()
                audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, originalVolume, 0)
            }
            status = controller.status.value
            println("loopback status: ${status.toJson()}")
            assertTrue("heard ${status.selfChirpsDetected} self chirps: ${status.toJson()}", status.selfChirpsDetected >= requiredSelfChirps)
            val snr = status.lastSelfSnrDb ?: Double.NaN
            assertTrue("self SNR $snr dB below ${configuration.chirpLab.minimumAcceptableSelfSnrDb}", snr >= configuration.chirpLab.minimumAcceptableSelfSnrDb)
            val peak = status.lastSelfPeakFraction ?: 1f
            assertTrue("self chirp clipped (peak $peak FS)", peak < configuration.chirpLab.clippingPeakFraction)
            val wav = File(outputRoot, "loopback/${FieldRunBundleFiles.CAPTURE_WAV}")
            val wavSamples = (wav.length() - wavHeaderBytes) / pcm16BytesPerSample
            assertTrue("WAV holds $wavSamples samples vs ${status.captureFrames} captured", wavSamples >= status.captureFrames - configuration.capture.captureBlockSampleCount)
            assertTrue(File(outputRoot, "loopback/${FieldRunBundleFiles.EVENTS_JSONL}").readText().contains("\"type\":\"emission\""))
        }
    }
}
