package com.minimappr.field.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import com.minimappr.field.node.FieldCaptureServiceState
import com.minimappr.field.node.FieldServiceMode
import com.minimappr.field.node.FieldServiceStatus
import com.minimappr.field.session.DiscoveredFieldHost

internal val SCREEN_PADDING = 16.dp
internal val ROW_SPACING = 8.dp

/** Test tags the instrumented tests look up. */
object FieldAppTestTags {
    const val SERVICE_STATE = "field-service-state"
    const val FFI_VERSION = "field-ffi-version"
    const val START_CAPTURE = "field-start-capture"
    const val STOP_CAPTURE = "field-stop-capture"
}

/** Root screen: header, service state, then the selected mode's screen. */
@Composable
fun FieldApp(
    serviceStatus: FieldServiceStatus,
    ffiSurfaceVersion: Long,
    discoveredHosts: List<DiscoveredFieldHost>,
    actions: FieldAppActions,
) {
    MaterialTheme {
        Surface(modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier.padding(SCREEN_PADDING).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(ROW_SPACING),
            ) {
                Text("MinimapPR Field", style = MaterialTheme.typography.headlineSmall)
                Text("FFI surface v$ffiSurfaceVersion", modifier = Modifier.testTag(FieldAppTestTags.FFI_VERSION))
                Text(
                    "Service: ${serviceStatus.serviceState}${serviceStatus.mode?.let { " (${it.wireName})" } ?: ""}",
                    modifier = Modifier.testTag(FieldAppTestTags.SERVICE_STATE),
                )
                serviceStatus.failureReason?.let { Text("Failure: $it", color = MaterialTheme.colorScheme.error) }
                FieldScreenNavigation(initialScreen = serviceStatus.mode.toScreen()) { screen ->
                    when (screen) {
                        FieldScreen.CAPTURE -> CaptureScreen(serviceStatus, actions)
                        FieldScreen.CHIRP_LAB -> ChirpLabScreen(serviceStatus, actions)
                        FieldScreen.SESSION -> SessionScreen(serviceStatus, discoveredHosts, actions)
                    }
                }
            }
        }
    }
}

private fun FieldServiceMode?.toScreen(): FieldScreen = when (this) {
    FieldServiceMode.CHIRP_LAB -> FieldScreen.CHIRP_LAB
    FieldServiceMode.SESSION -> FieldScreen.SESSION
    FieldServiceMode.CAPTURE, null -> FieldScreen.CAPTURE
}

@Composable
private fun CaptureScreen(serviceStatus: FieldServiceStatus, actions: FieldAppActions) {
    val running = serviceStatus.serviceState == FieldCaptureServiceState.RUNNING
    Row(horizontalArrangement = Arrangement.spacedBy(ROW_SPACING)) {
        Button(
            onClick = actions.onStartCapture,
            enabled = !running,
            modifier = Modifier.testTag(FieldAppTestTags.START_CAPTURE),
        ) { Text("Start capture") }
        OutlinedButton(onClick = actions.onStopService, modifier = Modifier.testTag(FieldAppTestTags.STOP_CAPTURE)) { Text("Stop") }
    }
    Text("Source: ${serviceStatus.captureSourceName ?: "-"}")
    serviceStatus.session?.takeIf { serviceStatus.mode == FieldServiceMode.CAPTURE }?.let { session ->
        Text("Samples captured: ${session.capturedSampleCount}")
        Text("Capture gaps: ${session.sync?.captureGapCount ?: 0uL}")
        Text("Stage-1 triggers: ${session.stage1TriggersOpened}, onsets: ${session.stage1OnsetsDetected}")
        Text("Last notice: ${session.noticeHistory.lastOrNull() ?: "-"}")
    }
}
