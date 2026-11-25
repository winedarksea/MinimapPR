package com.minimappr.field

import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.window.ComposeUIViewController
import com.minimappr.field.ffi.fieldFfiSurfaceVersion
import com.minimappr.field.node.FieldServiceStatusBoard
import com.minimappr.field.ui.FieldApp
import com.minimappr.field.ui.FieldAppActions

/**
 * Entry point for a future native iOS shell (Xcode project not included:
 * needs full Xcode). Capture via AVAudioEngine and a background-audio mode
 * are not implemented; the buttons are inert on iOS for now.
 */
@Suppress("FunctionName", "unused")
fun MainViewController() = ComposeUIViewController {
    val serviceStatus by FieldServiceStatusBoard.status.collectAsState()
    FieldApp(
        serviceStatus = serviceStatus,
        ffiSurfaceVersion = fieldFfiSurfaceVersion().toLong(),
        discoveredHosts = emptyList(),
        actions = FieldAppActions(),
    )
}
