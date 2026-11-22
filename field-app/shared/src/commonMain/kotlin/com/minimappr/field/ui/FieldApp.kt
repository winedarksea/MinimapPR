package com.minimappr.field.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
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
import com.minimappr.field.node.FieldServiceStatus

private val SCREEN_PADDING = 16.dp
private val ROW_SPACING = 8.dp

/** Test tags the instrumented smoke test looks up. */
object FieldAppTestTags {
    const val SERVICE_STATE = "field-service-state"
    const val FFI_VERSION = "field-ffi-version"
    const val START_CAPTURE = "field-start-capture"
    const val STOP_CAPTURE = "field-stop-capture"
}

/** Skeleton status screen: capture service control plus node counters. */
@Composable
fun FieldApp(
    serviceStatus: FieldServiceStatus,
    ffiSurfaceVersion: Long,
    onStartCapture: () -> Unit,
    onStopCapture: () -> Unit,
) {
    MaterialTheme {
        Surface(modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier.padding(SCREEN_PADDING),
                verticalArrangement = Arrangement.spacedBy(ROW_SPACING),
            ) {
                Text("MinimapPR Field", style = MaterialTheme.typography.headlineSmall)
                Text(
                    "FFI surface v$ffiSurfaceVersion",
                    modifier = Modifier.testTag(FieldAppTestTags.FFI_VERSION),
                )
                Text(
                    "Capture service: ${serviceStatus.serviceState}",
                    modifier = Modifier.testTag(FieldAppTestTags.SERVICE_STATE),
                )
                serviceStatus.failureReason?.let { Text("Failure: $it", color = MaterialTheme.colorScheme.error) }
                Row(horizontalArrangement = Arrangement.spacedBy(ROW_SPACING)) {
                    Button(
                        onClick = onStartCapture,
                        enabled = serviceStatus.serviceState != FieldCaptureServiceState.RUNNING,
                        modifier = Modifier.testTag(FieldAppTestTags.START_CAPTURE),
                    ) { Text("Start capture") }
                    OutlinedButton(
                        onClick = onStopCapture,
                        modifier = Modifier.testTag(FieldAppTestTags.STOP_CAPTURE),
                    ) { Text("Stop") }
                }
                FieldNodeCounters(serviceStatus)
            }
        }
    }
}

@Composable
private fun FieldNodeCounters(serviceStatus: FieldServiceStatus) {
    val node = serviceStatus.node
    Text("Protocol ${node.protocolVersionLabel}")
    Text("Samples captured: ${node.capturedSampleCount}")
    Text("Capture gaps: ${node.captureGapCount}")
    Text("Stage-1 triggers: ${node.stage1TriggersOpened}, onsets: ${node.stage1OnsetsDetected}")
    Text("Classifier windows: ${node.classifierWindowsAccepted} run, ${node.classifierWindowsDropped} dropped")
    Text("Frames sent: ${node.outboundFramesSent} (last kind ${node.lastOutboundMessageKindId ?: "-"})")
    Text("Last notice: ${node.lastUserNotice ?: "-"}")
}
