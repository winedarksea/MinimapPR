package com.minimappr.field.android

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import com.minimappr.field.FieldAppConfiguration
import com.minimappr.field.chirplab.ChirpLabRunParameters
import com.minimappr.field.discovery.AndroidNsdHostBrowser
import com.minimappr.field.ffi.fieldFfiSurfaceVersion
import com.minimappr.field.node.FieldCaptureServiceState
import com.minimappr.field.node.FieldServiceStatusBoard
import com.minimappr.field.session.DiscoveredFieldHost
import com.minimappr.field.session.SessionLaunchParameters
import com.minimappr.field.session.SessionLaunchRole
import com.minimappr.field.ui.FieldApp
import com.minimappr.field.ui.FieldAppActions
import kotlinx.coroutines.flow.MutableStateFlow

/** Android 17 local-network permission; a string because compileSdk 36 has no constant. */
private const val ACCESS_LOCAL_NETWORK_PERMISSION = "android.permission.ACCESS_LOCAL_NETWORK"
private const val FIRST_SDK_WITH_LOCAL_NETWORK_PERMISSION = 37
private const val SCREEN_WAKE_LOCK_TAG = "minimappr:launch-wake"
private const val SCREEN_WAKE_HOLD_MILLISECONDS = 3_000L

/**
 * Hosts the shared Compose UI, asks for runtime permissions, and forwards
 * launch-intent extras (`am start --es mode ...`, the laptop script
 * contract) to the foreground service, which must start while we are
 * visible. Also runs an NSD browser for the Session screen's host list.
 */
class FieldMainActivity : ComponentActivity() {
    private val appConfiguration = FieldAppConfiguration()
    private val discoveredHosts = MutableStateFlow<List<DiscoveredFieldHost>>(emptyList())
    private var hostBrowser: AndroidNsdHostBrowser? = null
    private var pendingLaunch: FieldLaunchRequest? = null
    private var launchAwaitingFocus: FieldLaunchRequest? = null

    private val requiredRuntimePermissions: Array<String> get() = buildList {
        add(Manifest.permission.RECORD_AUDIO)
        add(Manifest.permission.POST_NOTIFICATIONS)
        if (Build.VERSION.SDK_INT >= FIRST_SDK_WITH_LOCAL_NETWORK_PERMISSION) add(ACCESS_LOCAL_NETWORK_PERMISSION)
    }.toTypedArray()

    private val permissionRequest =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { _ ->
            // Only the missing permissions were asked, so read the real state.
            // Audio is required; a denied local-network permission surfaces as
            // a visible transport/NSD error rather than a silent no-start.
            val launch = pendingLaunch
            pendingLaunch = null
            val recordAudioGranted = checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
            if (recordAudioGranted && launch != null) startService(launch)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Loading the Rust library here surfaces a missing/mismatched .so at
        // launch instead of at the first capture block.
        val ffiSurfaceVersion = fieldFfiSurfaceVersion().toLong()
        hostBrowser = AndroidNsdHostBrowser(this, appConfiguration.session) { hosts -> discoveredHosts.value = hosts }.also { it.start() }
        setContent {
            val serviceStatus by FieldServiceStatusBoard.status.collectAsState()
            val hosts by discoveredHosts.collectAsState()
            FieldApp(serviceStatus, ffiSurfaceVersion, hosts, uiActions())
        }
        if (savedInstanceState == null) handleLaunchIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleLaunchIntent(intent)
    }

    private fun handleLaunchIntent(intent: Intent?) {
        val request = parseFieldLaunchRequest(intent?.extras, appConfiguration) ?: return
        // Script launches often hit a dozing phone, and a microphone FGS may
        // only start while we are really visible ("top sleeping" is refused):
        // show over the keyguard, wake the screen, start on window focus.
        setShowWhenLocked(true)
        setTurnScreenOn(true)
        wakeScreenBriefly()
        launchAwaitingFocus = request
        if (hasWindowFocus() && getSystemService(PowerManager::class.java).isInteractive) startPendingLaunch()
    }

    @Suppress("DEPRECATION") // ACQUIRE_CAUSES_WAKEUP needs a screen lock level; no replacement for a one-shot wake.
    private fun wakeScreenBriefly() {
        val powerManager = getSystemService(PowerManager::class.java)
        if (powerManager.isInteractive) return
        powerManager.newWakeLock(PowerManager.SCREEN_DIM_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP, SCREEN_WAKE_LOCK_TAG)
            .acquire(SCREEN_WAKE_HOLD_MILLISECONDS)
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) startPendingLaunch()
    }

    private fun startPendingLaunch() {
        val request = launchAwaitingFocus ?: return
        launchAwaitingFocus = null
        startWhenPermitted(request)
    }

    private fun uiActions() = FieldAppActions(
        onStartCapture = { startWhenPermitted(FieldLaunchRequest.Capture) },
        onStopService = { FieldCaptureForegroundService.stop(this) },
        onStartChirpLab = { role ->
            val parameters = ChirpLabRunParameters.defaults(appConfiguration.chirpLab, defaultRunId("chirp-${role.wireName}"), role)
            startWhenPermitted(FieldLaunchRequest.ChirpLab(parameters))
        },
        onHostSession = { startWhenPermitted(sessionRequest(SessionLaunchRole.HOST)) },
        onJoinDiscoveredHost = { host ->
            startWhenPermitted(sessionRequest(SessionLaunchRole.JOIN, "${host.hostAddress}:${host.port}", host.sessionIdHex))
        },
        onJoinTypedAddress = { address -> startWhenPermitted(sessionRequest(SessionLaunchRole.JOIN, address.trim(), null)) },
    )

    private fun sessionRequest(role: SessionLaunchRole, hostAddress: String? = null, sessionIdHex: String? = null) =
        FieldLaunchRequest.Session(
            SessionLaunchParameters(
                role = role,
                hostAddress = hostAddress,
                expectedSessionIdHex = sessionIdHex,
                airTemperatureC = appConfiguration.session.defaultAirTemperatureC,
                runId = defaultRunId("session-${role.wireName}"),
            )
        )

    private fun startWhenPermitted(request: FieldLaunchRequest) {
        val missing = requiredRuntimePermissions.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        val recordAudioGranted = Manifest.permission.RECORD_AUDIO !in missing
        if (recordAudioGranted && (missing.isEmpty() || request !is FieldLaunchRequest.Session)) {
            startService(request)
        } else {
            pendingLaunch = request
            permissionRequest.launch(missing.toTypedArray())
        }
    }

    private fun startService(request: FieldLaunchRequest) {
        try {
            FieldCaptureForegroundService.start(this, request.toExtras())
        } catch (notAllowed: IllegalStateException) {
            // ForegroundServiceStartNotAllowedException: we were not visible
            // yet; retry on the next window focus.
            launchAwaitingFocus = request
            FieldServiceStatusBoard.update {
                it.copy(serviceState = FieldCaptureServiceState.FAILED, mode = request.mode, failureReason = "service start refused: ${notAllowed.message}")
            }
        }
    }

    override fun onDestroy() {
        hostBrowser?.stop()
        hostBrowser = null
        super.onDestroy()
    }
}
