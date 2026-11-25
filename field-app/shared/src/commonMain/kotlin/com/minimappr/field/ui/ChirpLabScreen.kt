package com.minimappr.field.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import com.minimappr.field.chirplab.ChirpLabRole
import com.minimappr.field.node.FieldCaptureServiceState
import com.minimappr.field.node.FieldServiceMode
import com.minimappr.field.node.FieldServiceStatus

private const val SNR_DISPLAY_DECIMALS = 1
private const val PEAK_DISPLAY_DECIMALS = 3

/**
 * Stage A debug screen: start a role with default schedule parameters (the
 * laptop script uses `am start` extras instead), watch live self SNR and
 * peak fraction (clipping) per emission.
 */
@Composable
fun ChirpLabScreen(serviceStatus: FieldServiceStatus, actions: FieldAppActions) {
    val running = serviceStatus.serviceState == FieldCaptureServiceState.RUNNING
    Row(horizontalArrangement = Arrangement.spacedBy(ROW_SPACING)) {
        ChirpLabRole.entries.forEach { role ->
            Button(onClick = { actions.onStartChirpLab(role) }, enabled = !running) { Text("Start role ${role.wireName}") }
        }
        OutlinedButton(onClick = actions.onStopService) { Text("Stop") }
    }
    val status = serviceStatus.chirpLab?.takeIf { serviceStatus.mode == FieldServiceMode.CHIRP_LAB } ?: run {
        Text("No chirp lab run.")
        return
    }
    Text("Run ${status.runId}, role ${status.role}${if (status.running) "" else " (stopped)"}")
    Text("Emissions: ${status.emissions} (skipped ${status.emissionsSkipped})")
    Text(
        "Self chirp: SNR ${status.lastSelfSnrDb?.formatDecimals(SNR_DISPLAY_DECIMALS) ?: "-"} dB, " +
            "peak ${status.lastSelfPeakFraction?.toDouble()?.formatDecimals(PEAK_DISPLAY_DECIMALS) ?: "-"} FS",
        style = MaterialTheme.typography.titleMedium,
    )
    Text("Self chirps found ${status.selfChirpsDetected}, missed ${status.selfChirpsMissed}")
    Text("Arrival vs prediction: ${status.lastSelfArrivalErrorFrames?.formatDecimals(SNR_DISPLAY_DECIMALS) ?: "-"} frames")
    Text("Capture frames ${status.captureFrames}, gaps ${status.captureGaps}, writer overflows ${status.writerOverflows}")
    Text("Playback ${status.playbackPerformanceMode ?: "-"}, underruns ${status.playbackUnderruns}; source ${status.captureSourceName ?: "-"}")
    status.outputDirectory?.let { Text("Output: $it") }
    status.failure?.let { Text("Failure: $it", color = MaterialTheme.colorScheme.error) }
}

internal fun Double.formatDecimals(decimals: Int): String {
    var scale = 1.0
    repeat(decimals) { scale *= 10 }
    val rounded = kotlin.math.round(this * scale) / scale
    return rounded.toString()
}
