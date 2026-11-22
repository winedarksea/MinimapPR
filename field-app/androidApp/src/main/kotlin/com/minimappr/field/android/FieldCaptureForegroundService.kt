package com.minimappr.field.android

import android.Manifest
import android.annotation.SuppressLint
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.minimappr.field.FieldAppConfiguration
import com.minimappr.field.capture.AndroidMicrophoneCaptureLoop
import com.minimappr.field.node.FieldCaptureServiceState
import com.minimappr.field.node.FieldNodeRuntime
import com.minimappr.field.node.FieldServiceStatusBoard
import com.minimappr.field.node.skeletonSoloNodeSetup
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

private const val LOG_TAG = "FieldCaptureService"

/**
 * Microphone foreground service: owns capture, the Rust node runtime and the
 * state-machine tick loop for as long as Field Mode is listening.
 *
 * Lifecycle: started from the foreground activity (Android 14+ refuses a
 * microphone FGS started from the background), promoted with
 * startForeground immediately, torn down on the Stop action or stopService.
 * START_NOT_STICKY: an OS restart without a visible activity would not be
 * allowed to reopen the mic anyway, so silently "restarting" would lie.
 */
class FieldCaptureForegroundService : Service() {
    private val appConfiguration = FieldAppConfiguration()
    private var serviceScope: CoroutineScope? = null
    private var nodeRuntime: FieldNodeRuntime? = null
    private var captureLoop: AndroidMicrophoneCaptureLoop? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP_CAPTURE) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (serviceScope != null) return START_NOT_STICKY
        FieldServiceStatusBoard.update { it.copy(serviceState = FieldCaptureServiceState.STARTING, failureReason = null) }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            failAndStop("RECORD_AUDIO not granted")
            return START_NOT_STICKY
        }
        FieldCaptureNotification.ensureChannel(this)
        try {
            ServiceCompat.startForeground(
                this,
                FieldCaptureNotification.NOTIFICATION_ID,
                FieldCaptureNotification.build(this),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
            )
        } catch (securityOrStateException: RuntimeException) {
            failAndStop("startForeground refused: ${securityOrStateException.message}")
            return START_NOT_STICKY
        }
        startNodeAndCapture()
        return START_NOT_STICKY
    }

    private fun startNodeAndCapture() {
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        serviceScope = scope
        val runtime = FieldNodeRuntime(
            skeletonSoloNodeSetup(appConfiguration.node, "android-${Build.MODEL}", BuildConfig.VERSION_NAME),
            appConfiguration.node,
        )
        nodeRuntime = runtime
        scope.launch {
            runtime.status.collect { nodeStatus -> FieldServiceStatusBoard.update { it.copy(node = nodeStatus) } }
        }
        scope.launch { runtime.runTickLoop() }

        val loop = AndroidMicrophoneCaptureLoop(
            context = this,
            captureConfiguration = appConfiguration.capture,
            captureBlockSink = runtime,
            onCaptureGap = runtime::onCaptureGap,
        )
        captureLoop = loop
        val captureStarted = loop.hasRecordAudioPermission() && startCaptureWithCheckedPermission(loop, scope)
        if (!captureStarted) {
            failAndStop("no audio source could be opened")
            return
        }
        Log.i(LOG_TAG, "capture running from ${loop.activeAudioSourceName}")
        FieldServiceStatusBoard.update { it.copy(serviceState = FieldCaptureServiceState.RUNNING) }
    }

    @SuppressLint("MissingPermission") // RECORD_AUDIO checked by the caller
    private fun startCaptureWithCheckedPermission(loop: AndroidMicrophoneCaptureLoop, scope: CoroutineScope): Boolean =
        loop.start(scope)

    private fun failAndStop(reason: String) {
        Log.e(LOG_TAG, reason)
        FieldServiceStatusBoard.update { it.copy(serviceState = FieldCaptureServiceState.FAILED, failureReason = reason) }
        stopSelf()
    }

    override fun onDestroy() {
        captureLoop?.stop()
        captureLoop = null
        serviceScope?.cancel()
        serviceScope = null
        nodeRuntime?.close()
        nodeRuntime = null
        FieldServiceStatusBoard.update { current ->
            if (current.serviceState == FieldCaptureServiceState.FAILED) current
            else current.copy(serviceState = FieldCaptureServiceState.STOPPED)
        }
        super.onDestroy()
    }

    companion object {
        private const val ACTION_STOP_CAPTURE = "com.minimappr.field.action.STOP_CAPTURE"

        fun start(context: Context) {
            ContextCompat.startForegroundService(context, Intent(context, FieldCaptureForegroundService::class.java))
        }

        fun stopIntent(context: Context): Intent =
            Intent(context, FieldCaptureForegroundService::class.java).setAction(ACTION_STOP_CAPTURE)

        fun stop(context: Context) {
            context.stopService(Intent(context, FieldCaptureForegroundService::class.java))
        }
    }
}
