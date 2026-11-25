package com.minimappr.field.android

import android.Manifest
import android.annotation.SuppressLint
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.net.wifi.WifiManager
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.util.Log
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.minimappr.field.FieldAppConfiguration
import com.minimappr.field.chirplab.ChirpLabController
import com.minimappr.field.node.FieldCaptureServiceState
import com.minimappr.field.node.FieldServiceMode
import com.minimappr.field.node.FieldServiceStatusBoard
import com.minimappr.field.session.SessionLaunchParameters
import com.minimappr.field.session.SessionLaunchRole
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.FileDescriptor
import java.io.PrintWriter

private const val LOG_TAG = "FieldCaptureService"
private const val WAKE_LOCK_TAG = "minimappr:field-run"
private const val WIFI_LOCK_TAG = "minimappr:field-session"

/**
 * Foreground service for every Field mode (CAPTURE | CHIRP_LAB | SESSION).
 *
 * Lifecycle: started from the foreground activity (Android 14+ refuses a
 * microphone FGS started from the background), promoted with
 * startForeground immediately, torn down on the Stop action / stopService.
 * A new start request replaces the running run. START_NOT_STICKY: an OS
 * restart without a visible activity could not reopen the mic anyway.
 *
 * Screen-locked runs: FGS types microphone|mediaPlayback plus a partial wake
 * lock keep capture and the continuous AudioTrack alive; SESSION also holds
 * Wi-Fi locks against power-save latency spikes.
 *
 * `dumpsys activity service .../.FieldCaptureForegroundService` prints
 * `CHIRP_LAB_STATUS {json}` / `SYNC_STATUS {json}` for the laptop scripts.
 */
class FieldCaptureForegroundService : Service() {
    private val appConfiguration = FieldAppConfiguration()
    private var serviceScope: CoroutineScope? = null
    private var activeRun: FieldServiceRun? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private val wifiLocks = mutableListOf<WifiManager.WifiLock>()
    private val mainThreadHandler = Handler(Looper.getMainLooper())

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        activeInstance = this
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP_CAPTURE) {
            stopSelf()
            return START_NOT_STICKY
        }
        val request = parseFieldLaunchRequest(intent?.extras, appConfiguration) ?: FieldLaunchRequest.Capture
        if (activeRun != null) stopActiveRun()
        FieldServiceStatusBoard.update {
            it.copy(serviceState = FieldCaptureServiceState.STARTING, mode = request.mode, failureReason = null, chirpLab = null, session = null)
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            failAndStop("RECORD_AUDIO not granted")
            return START_NOT_STICKY
        }
        FieldCaptureNotification.ensureChannel(this)
        try {
            ServiceCompat.startForeground(this, FieldCaptureNotification.NOTIFICATION_ID, FieldCaptureNotification.build(this), foregroundTypeFor(request.mode))
        } catch (securityOrStateException: RuntimeException) {
            failAndStop("startForeground refused: ${securityOrStateException.message}")
            return START_NOT_STICKY
        }
        startRun(request)
        return START_NOT_STICKY
    }

    private fun foregroundTypeFor(mode: FieldServiceMode): Int = when (mode) {
        FieldServiceMode.CAPTURE -> ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
        FieldServiceMode.CHIRP_LAB, FieldServiceMode.SESSION ->
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE or ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK
    }

    private fun createRun(request: FieldLaunchRequest): FieldServiceRun = when (request) {
        is FieldLaunchRequest.ChirpLab ->
            ChirpLabServiceRun(ChirpLabController(this, appConfiguration, request.parameters, chirpLabRootDirectory(this)))
        is FieldLaunchRequest.Session -> SessionServiceRun(this, appConfiguration, request.parameters, BuildConfig.VERSION_NAME)
        FieldLaunchRequest.Capture -> SessionServiceRun(
            this, appConfiguration,
            SessionLaunchParameters(
                role = SessionLaunchRole.SOLO_CAPTURE,
                airTemperatureC = appConfiguration.session.defaultAirTemperatureC,
                runId = defaultRunId("capture"),
            ),
            BuildConfig.VERSION_NAME,
        )
    }

    private fun startRun(request: FieldLaunchRequest) {
        // Any failure inside the run (capture read, FFI error, tick) must stop
        // the service visibly, not crash the process from a worker thread.
        lateinit var scope: CoroutineScope
        val failureHandler = CoroutineExceptionHandler { _, failure ->
            Log.e(LOG_TAG, "run coroutine failed", failure)
            mainThreadHandler.post { if (serviceScope === scope) failAndStop("${request.mode.wireName} failed: $failure") }
        }
        scope = CoroutineScope(SupervisorJob() + Dispatchers.Default + failureHandler)
        serviceScope = scope
        val run = try {
            createRun(request)
        } catch (setupFailure: RuntimeException) {
            failAndStop("setup rejected: ${setupFailure.message}")
            return
        }
        activeRun = run
        acquireLocks(request.mode)
        scope.launch {
            startWithCheckedPermission(run, scope)
            Log.i(LOG_TAG, "${request.mode.wireName} running")
            FieldServiceStatusBoard.update { it.copy(serviceState = FieldCaptureServiceState.RUNNING) }
        }
    }

    @SuppressLint("MissingPermission") // RECORD_AUDIO checked in onStartCommand
    private suspend fun startWithCheckedPermission(run: FieldServiceRun, scope: CoroutineScope) = run.start(scope)

    private fun acquireLocks(mode: FieldServiceMode) {
        if (mode == FieldServiceMode.CAPTURE) return
        val powerManager = getSystemService(PowerManager::class.java)
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_LOCK_TAG).apply {
            setReferenceCounted(false)
            acquire()
        }
        if (mode != FieldServiceMode.SESSION) return
        val wifiManager = applicationContext.getSystemService(WifiManager::class.java)
        // LOW_LATENCY only applies while foreground + screen on; HIGH_PERF
        // still disables power save on pre-34 releases (the A03s) when locked.
        @Suppress("DEPRECATION")
        val modes = listOf(WifiManager.WIFI_MODE_FULL_LOW_LATENCY, WifiManager.WIFI_MODE_FULL_HIGH_PERF)
        modes.forEach { lockMode ->
            wifiLocks += wifiManager.createWifiLock(lockMode, WIFI_LOCK_TAG).apply {
                setReferenceCounted(false)
                acquire()
            }
        }
    }

    private fun releaseLocks() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
        wifiLocks.forEach { if (it.isHeld) it.release() }
        wifiLocks.clear()
    }

    private fun stopActiveRun() {
        // Order: the run stops audio before its device / files close.
        activeRun?.stop()
        activeRun = null
        serviceScope?.cancel()
        serviceScope = null
        releaseLocks()
    }

    fun markRun(label: String) {
        activeRun?.mark(label)
    }

    private fun failAndStop(reason: String) {
        Log.e(LOG_TAG, reason)
        FieldServiceStatusBoard.update { current ->
            current.copy(
                serviceState = FieldCaptureServiceState.FAILED,
                failureReason = reason,
                chirpLab = current.chirpLab?.copy(running = false, failure = reason),
            )
        }
        stopSelf()
    }

    override fun dump(fd: FileDescriptor?, writer: PrintWriter, args: Array<out String>?) {
        val status = FieldServiceStatusBoard.status.value
        writer.println("FIELD_SERVICE {\"state\":\"${status.serviceState}\",\"mode\":\"${status.mode?.wireName}\"}")
        activeRun?.dumpLines()?.forEach(writer::println)
        if (activeRun == null) status.chirpLab?.let { writer.println("${FieldDumpPrefixes.CHIRP_LAB_STATUS} ${it.toJson()}") }
    }

    override fun onDestroy() {
        stopActiveRun()
        activeInstance = null
        FieldServiceStatusBoard.update { current ->
            if (current.serviceState == FieldCaptureServiceState.FAILED) current
            else current.copy(serviceState = FieldCaptureServiceState.STOPPED)
        }
        super.onDestroy()
    }

    companion object {
        private const val ACTION_STOP_CAPTURE = "com.minimappr.field.action.STOP_CAPTURE"

        /** Same-process handle for the debug receiver's MARK command. */
        @Volatile
        var activeInstance: FieldCaptureForegroundService? = null
            private set

        fun start(context: Context, launchExtras: Bundle? = null) {
            val intent = Intent(context, FieldCaptureForegroundService::class.java)
            launchExtras?.let(intent::putExtras)
            ContextCompat.startForegroundService(context, intent)
        }

        fun stopIntent(context: Context): Intent =
            Intent(context, FieldCaptureForegroundService::class.java).setAction(ACTION_STOP_CAPTURE)

        fun stop(context: Context) {
            context.stopService(Intent(context, FieldCaptureForegroundService::class.java))
        }
    }
}
