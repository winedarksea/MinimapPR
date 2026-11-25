package com.minimappr.field.session

import android.Manifest
import android.content.Context
import android.util.Log
import androidx.annotation.RequiresPermission
import com.minimappr.field.FieldAppConfiguration
import com.minimappr.field.audio.AndroidDuplexAudioRig
import com.minimappr.field.capture.CaptureBlockSink
import com.minimappr.field.capture.FanOutCaptureBlockSink
import com.minimappr.field.chirplab.AndroidDeviceConditionMonitor
import com.minimappr.field.chirplab.AndroidDeviceSurvey
import com.minimappr.field.chirplab.ChirpLabWireNames
import com.minimappr.field.chirplab.fieldPlatformName
import com.minimappr.field.discovery.AndroidNsdHostAdvertiser
import com.minimappr.field.discovery.AndroidNsdHostBrowser
import com.minimappr.field.recording.DiscardingFieldRunEventLog
import com.minimappr.field.recording.FieldRunEventLog
import com.minimappr.field.recording.FieldRunRecording
import com.minimappr.field.transport.FieldLanTransport
import com.minimappr.field.transport.KtorHostWebSocketServer
import com.minimappr.field.transport.KtorJoinerWebSocketClient
import com.minimappr.field.transport.createAndroidWebSocketHttpClient
import com.minimappr.field.transport.parseHostAndPort
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File

private const val LOG_TAG = "FieldSessionRunner"
private const val NSD_SERVICE_NAME_PREFIX = "minimappr-"
private const val MODEL_DEFAULT_CHANNEL_LABEL = "model_default"

/**
 * Android wiring for one LAN session (or the solo CAPTURE node): identity,
 * NSD, Ktor transport, duplex audio rig and the platform-neutral
 * [FieldSessionController]. Start order matters: the controller (and its
 * transport) exist before capture starts, so no block reaches a missing
 * device; the duplex start then gives the device capture anchors and the
 * playback engine its first timestamp before any EmitChirp can arrive.
 */
class AndroidFieldSessionRunner(
    private val context: Context,
    private val configuration: FieldAppConfiguration,
    val parameters: SessionLaunchParameters,
    private val appVersion: String,
    private val chirpLabRootDirectory: File?,
) {
    private val sessionConfiguration = configuration.session
    private val mutableController = MutableStateFlow<FieldSessionController?>(null)
    val controller: StateFlow<FieldSessionController?> = mutableController.asStateFlow()
    private var rig: AndroidDuplexAudioRig? = null
    private var recording: FieldRunRecording? = null
    private var advertiser: AndroidNsdHostAdvertiser? = null
    private var browser: AndroidNsdHostBrowser? = null
    private var conditionMonitor: AndroidDeviceConditionMonitor? = null

    val captureSourceName: String? get() = rig?.capture?.activeAudioSourceName
    val playbackPerformanceMode: String? get() = rig?.playback?.performanceModeName

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    suspend fun start(scope: CoroutineScope) {
        val (identity, transport) = identityAndTransport(scope)
        val tuning = parameters.meshTuning()
        val runRecording = if (parameters.shadowRecording && chirpLabRootDirectory != null) {
            FieldRunRecording(File(chirpLabRootDirectory, parameters.runId), configuration.capture.captureSampleRateHz, configuration.chirpLab, scope)
        } else {
            null
        }
        recording = runRecording
        val eventLog: FieldRunEventLog = runRecording ?: DiscardingFieldRunEventLog
        lateinit var sessionController: FieldSessionController
        val deviceSink = CaptureBlockSink { block -> sessionController.onCaptureBlock(block) }
        val sinks = listOfNotNull(deviceSink, runRecording)
        val isSolo = parameters.role == SessionLaunchRole.SOLO_CAPTURE
        val duplexRig = AndroidDuplexAudioRig(
            context, configuration, FanOutCaptureBlockSink(sinks), eventLog, forcedAudioSource = null,
            emissionChannelLabel = tuning.chirpOutputChannel?.let(ChirpLabWireNames::channelName) ?: MODEL_DEFAULT_CHANNEL_LABEL,
            emissionLevelDbfs = tuning.chirpPeakDbfs,
            onCaptureTimestamp = { sample -> sessionController.onCaptureTimestamp(sample.anchor.framePosition, sample.anchor.nanoTime) },
            onCaptureGap = { sessionController.onCaptureGap() },
            withPlayback = !isSolo,
        )
        rig = duplexRig
        val setup = identity.deviceSetup(fieldPlatformName(), appVersion, configuration.node.rollingBufferSeconds, canEmitChirps = !isSolo)
        sessionController = FieldSessionController(
            setup, tuning, sessionConfiguration, transport, if (isSolo) null else duplexRig.playback, System::nanoTime, eventLog,
        )
        mutableController.value = sessionController
        sessionController.start(scope)
        // Throws DuplexAudioStartException; the service shows it and stops.
        duplexRig.start(scope)
        if (identity.role == com.minimappr.field.ffi.FieldDeviceRole.HOST && transport is KtorHostWebSocketServer) {
            advertiser = AndroidNsdHostAdvertiser(context, sessionConfiguration).also {
                it.register("$NSD_SERVICE_NAME_PREFIX${fieldPlatformName()}", transport.boundPort, identity.nodeId, identity.sessionIdHex)
            }
        }
        conditionMonitor = AndroidDeviceConditionMonitor(
            context, configuration.chirpLab, eventLog, duplexRig::routedInputDescription, duplexRig::routedOutputDescription,
            onThermalHeadroom = sessionController::onThermalHeadroom,
        ).also { it.start(scope) }
        runRecording?.writeSessionJson(
            AndroidDeviceSurvey(context).sessionJson(
                runId = parameters.runId,
                role = parameters.role.wireName,
                mode = "session",
                schedule = {
                    put("period_ms", tuning.periodicRoundIntervalMs.toLong())
                    put("lead_ms", tuning.scheduleLeadTimeMs.toLong())
                    put("slot_ms", tuning.slotDurationMs.toLong())
                    putObject("waveform") {
                        put("kind", ChirpLabWireNames.waveformKindName(tuning.waveform.kind))
                        put("band_low_hz", tuning.waveform.bandLowHz)
                        put("band_high_hz", tuning.waveform.bandHighHz)
                        put("duration_s", tuning.waveform.durationS)
                    }
                    put("channel", tuning.chirpOutputChannel?.let(ChirpLabWireNames::channelName) ?: MODEL_DEFAULT_CHANNEL_LABEL)
                    put("level_dbfs", tuning.chirpPeakDbfs)
                    put("air_temperature_c", tuning.airTemperatureC)
                    put("node_id", identity.nodeId.toLong())
                },
                paths = duplexRig.audioPathFacts("default"),
                activeMicrophones = duplexRig.activeMicrophones(),
                startedNanoTime = duplexRig.startedNanoTime,
                startedWallClockMilliseconds = duplexRig.startedWallClockMilliseconds,
            )
        )
    }

    private suspend fun identityAndTransport(scope: CoroutineScope): Pair<FieldSessionIdentity, FieldLanTransport?> =
        when (parameters.role) {
            SessionLaunchRole.SOLO_CAPTURE -> FieldSessionIdentity.newHost() to null
            SessionLaunchRole.HOST -> {
                val identity = FieldSessionIdentity.newHost()
                identity to KtorHostWebSocketServer(identity.nodeId, sessionConfiguration, diagnosticLog = { Log.i(LOG_TAG, it) })
            }
            SessionLaunchRole.JOIN -> {
                val (address, sessionIdHex) = parameters.hostAddress?.let { it to parameters.expectedSessionIdHex } ?: discoverFirstHost()
                val (hostName, port) = parseHostAndPort(address, sessionConfiguration.hostListenPort)
                    ?: throw IllegalArgumentException("host address '$address' is not ip:port")
                val identity = FieldSessionIdentity.newJoiner(parseSessionIdHex(sessionIdHex))
                identity to KtorJoinerWebSocketClient(
                    hostName, port, sessionConfiguration, { createAndroidWebSocketHttpClient(sessionConfiguration) }, scope,
                    diagnosticLog = { Log.i(LOG_TAG, it) },
                )
            }
        }

    /** Join without an address: the first NSD host found, insisting on its session id. */
    private suspend fun discoverFirstHost(): Pair<String, String?> {
        val found = CompletableDeferred<DiscoveredFieldHost>()
        val nsdBrowser = AndroidNsdHostBrowser(context, sessionConfiguration) { hosts -> hosts.firstOrNull()?.let(found::complete) }
        browser = nsdBrowser
        nsdBrowser.start()
        val host = withTimeoutOrNull(sessionConfiguration.nsdAutoJoinTimeoutMilliseconds) { found.await() }
        nsdBrowser.stop()
        browser = null
        host ?: throw IllegalStateException("no Field host found by NSD within ${sessionConfiguration.nsdAutoJoinTimeoutMilliseconds} ms")
        Log.i(LOG_TAG, "joining NSD host ${host.serviceName} at ${host.hostAddress}:${host.port}")
        return "${host.hostAddress}:${host.port}" to host.sessionIdHex
    }

    /** Idempotent. Audio first (playback outcomes still reach the device), then device and files. */
    fun stop() {
        browser?.stop()
        browser = null
        advertiser?.unregister()
        advertiser = null
        conditionMonitor?.stop()
        conditionMonitor = null
        rig?.stop()
        rig = null
        mutableController.value?.close()
        recording?.close()
        recording = null
    }
}
