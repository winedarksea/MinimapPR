package com.minimappr.field.android

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import com.minimappr.field.ffi.fieldFfiSurfaceVersion
import com.minimappr.field.node.FieldServiceStatusBoard
import com.minimappr.field.ui.FieldApp

/**
 * Hosts the shared Compose UI and asks for RECORD_AUDIO + notifications
 * before starting the capture service (it must start while we are visible).
 */
class FieldMainActivity : ComponentActivity() {
    private val requiredRuntimePermissions = arrayOf(
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.POST_NOTIFICATIONS,
    )

    private val permissionRequest =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
            // Notifications are optional (the FGS still runs); audio is not.
            if (grants[Manifest.permission.RECORD_AUDIO] == true) FieldCaptureForegroundService.start(this)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Loading the Rust library here surfaces a missing/mismatched .so at
        // launch instead of at the first capture block.
        val ffiSurfaceVersion = fieldFfiSurfaceVersion().toLong()
        setContent {
            val serviceStatus by FieldServiceStatusBoard.status.collectAsState()
            FieldApp(
                serviceStatus = serviceStatus,
                ffiSurfaceVersion = ffiSurfaceVersion,
                onStartCapture = ::startCaptureWhenPermitted,
                onStopCapture = { FieldCaptureForegroundService.stop(this) },
            )
        }
    }

    private fun startCaptureWhenPermitted() {
        val recordAudioGranted =
            checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        if (recordAudioGranted) {
            FieldCaptureForegroundService.start(this)
        } else {
            permissionRequest.launch(requiredRuntimePermissions)
        }
    }
}
