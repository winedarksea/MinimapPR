package com.minimappr.field.android

import android.Manifest
import android.content.Context
import androidx.annotation.RequiresPermission
import com.minimappr.field.FieldAppConfiguration
import com.minimappr.field.chirplab.ChirpLabController
import com.minimappr.field.node.FieldServiceStatusBoard
import com.minimappr.field.session.AndroidFieldSessionRunner
import com.minimappr.field.session.SessionLaunchParameters
import com.minimappr.field.session.toSnakeCaseJson
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import java.io.File

/** dumpsys line prefixes: the laptop scripts grep for these. */
object FieldDumpPrefixes {
    const val CHIRP_LAB_STATUS = "CHIRP_LAB_STATUS"
    const val SYNC_STATUS = "SYNC_STATUS"
}

/** Root for run bundles: `getExternalFilesDir("chirp_lab")` (adb-pullable, app-private). */
fun chirpLabRootDirectory(context: Context): File =
    context.getExternalFilesDir(CHIRP_LAB_DIRECTORY_NAME) ?: File(context.filesDir, CHIRP_LAB_DIRECTORY_NAME)

private const val CHIRP_LAB_DIRECTORY_NAME = "chirp_lab"

/** One mode's work inside the foreground service. */
interface FieldServiceRun {
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    suspend fun start(scope: CoroutineScope)
    fun stop()
    fun dumpLines(): List<String>
    fun mark(label: String) = Unit
}

class ChirpLabServiceRun(private val controller: ChirpLabController) : FieldServiceRun {
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    override suspend fun start(scope: CoroutineScope) {
        scope.launch {
            controller.status.collect { chirpLab ->
                FieldServiceStatusBoard.update { it.copy(chirpLab = chirpLab, captureSourceName = chirpLab.captureSourceName) }
            }
        }
        controller.start(scope)
    }

    override fun stop() = controller.stop()

    override fun dumpLines() = listOf("${FieldDumpPrefixes.CHIRP_LAB_STATUS} ${controller.status.value.toJson()}")

    override fun mark(label: String) = controller.mark(label)
}

/** SESSION and CAPTURE (solo) modes. */
class SessionServiceRun(
    context: Context,
    configuration: FieldAppConfiguration,
    parameters: SessionLaunchParameters,
    appVersion: String,
) : FieldServiceRun {
    private val runner = AndroidFieldSessionRunner(context, configuration, parameters, appVersion, chirpLabRootDirectory(context))

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    override suspend fun start(scope: CoroutineScope) {
        scope.launch {
            runner.controller.collect { controller ->
                controller?.status?.collect { session ->
                    FieldServiceStatusBoard.update { it.copy(session = session, captureSourceName = runner.captureSourceName) }
                }
            }
        }
        runner.start(scope)
    }

    override fun stop() = runner.stop()

    override fun dumpLines(): List<String> {
        val controller = runner.controller.value ?: return emptyList()
        val session = controller.status.value
        val sync = runCatching { controller.device.syncStatus() }.getOrNull() ?: return emptyList()
        val json = sync.toSnakeCaseJson {
            put("session_role", runner.parameters.role.wireName)
            put("run_id", runner.parameters.runId)
            put("transport_endpoint", session.transportEndpoint)
            put("connected_peers", session.transport.connectedPeerCount)
            put("envelopes_sent", session.transport.envelopesSent)
            put("envelopes_received", session.transport.envelopesReceived)
            put("envelopes_dropped", session.transport.envelopesDroppedQueueFull + session.transport.envelopesDroppedUnknownRecipient)
            put("transport_last_error", session.transport.lastError)
            put("emissions_scheduled", session.emissionsScheduled)
            put("emissions_rejected", session.emissionsRejected)
            put("captured_samples", session.capturedSampleCount)
            put("capture_source", runner.captureSourceName)
            put("playback_performance_mode", runner.playbackPerformanceMode)
            putStrings("notices", session.noticeHistory)
        }
        return listOf("${FieldDumpPrefixes.SYNC_STATUS} $json")
    }

    override fun mark(label: String) = Unit
}
