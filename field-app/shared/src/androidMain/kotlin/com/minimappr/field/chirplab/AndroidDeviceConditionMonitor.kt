package com.minimappr.field.chirplab

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioDeviceCallback
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import androidx.core.content.ContextCompat
import com.minimappr.field.FieldChirpLabConfiguration
import com.minimappr.field.recording.FieldRunEventLog
import com.minimappr.field.recording.FieldRunEventTypes
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

private const val QUATERNION_COMPONENTS = 4
/** Rotation-vector `values[4]`: estimated heading accuracy, radians (−1 if unavailable). */
private const val HEADING_ACCURACY_VALUE_INDEX = 4
private const val MICROSECONDS_PER_MILLISECOND = 1_000
private const val NANOSECONDS_PER_MILLISECOND = 1_000_000L
/** getThermalHeadroom forecast horizon; 10 s matches the poll cadence. */
private const val THERMAL_FORECAST_SECONDS = 10

/**
 * Logs the run conditions the analyser needs to explain outliers:
 * attitude (rotation vector, 1 Hz; also forwarded to the session device
 * for the speaker–mic geometry correction), screen on/off, audio route
 * changes, thermal status/headroom (also forwarded) and media volume
 * changes. All callbacks only append to the non-blocking event log or hand
 * a small value to the session device; none touch the audio threads.
 *
 * Attitude convention: `SensorManager.getQuaternionFromVector` gives the
 * unit quaternion (w, x, y, z) rotating the device frame (x right, y up the
 * screen, z out of the screen) into east-north-up, with north = magnetic
 * north (no declination applied). Heading accuracy is the sensor's own
 * estimate (`values[4]`), null when the sensor does not report one.
 */
class AndroidDeviceConditionMonitor(
    private val context: Context,
    private val configuration: FieldChirpLabConfiguration,
    private val eventLog: FieldRunEventLog,
    private val routedInputDescription: () -> String?,
    private val routedOutputDescription: () -> String?,
    private val onThermalHeadroom: (Float) -> Unit = {},
    private val onAttitude: (quaternionWxyz: List<Float>, headingAccuracyRadians: Float?) -> Unit = { _, _ -> },
) {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    private val mainHandler = Handler(Looper.getMainLooper())
    private val jobs = mutableListOf<Job>()
    private val quaternion = FloatArray(QUATERNION_COMPONENTS)
    private var lastVolumeIndex: Int? = null
    private var lastAttitudeLoggedNs = Long.MIN_VALUE / 2
    @Volatile private var attitudeSensorName: String? = null

    private val attitudeListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            // The sampling period is only a hint (the Pixel delivers ~7 Hz): throttle here.
            if (event.timestamp - lastAttitudeLoggedNs < configuration.attitudeSampleIntervalMilliseconds * NANOSECONDS_PER_MILLISECOND) return
            lastAttitudeLoggedNs = event.timestamp
            SensorManager.getQuaternionFromVector(quaternion, event.values)
            val headingAccuracyRadians = event.values.getOrNull(HEADING_ACCURACY_VALUE_INDEX)?.takeIf { it.isFinite() && it >= 0f }
            eventLog.append(FieldRunEventTypes.ATTITUDE) {
                putNumbers("quaternion", quaternion.map { it.toDouble() })
                put("sensor", attitudeSensorName)
                put("heading_accuracy_rad", headingAccuracyRadians)
            }
            onAttitude(quaternion.toList(), headingAccuracyRadians)
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
    }

    private val screenReceiver = object : BroadcastReceiver() {
        override fun onReceive(receiverContext: Context, intent: Intent) {
            eventLog.append(FieldRunEventTypes.SCREEN) { put("on", intent.action == Intent.ACTION_SCREEN_ON) }
        }
    }

    private val routeCallback = object : AudioDeviceCallback() {
        override fun onAudioDevicesAdded(addedDevices: Array<out AudioDeviceInfo>?) = logRoute()
        override fun onAudioDevicesRemoved(removedDevices: Array<out AudioDeviceInfo>?) = logRoute()
    }

    private val thermalListener = PowerManager.OnThermalStatusChangedListener { logThermal() }

    fun start(scope: CoroutineScope) {
        // Budget phones (the A03s) have no gyroscope, hence no ROTATION_VECTOR;
        // the accelerometer + magnetometer variant still gives the pose.
        val attitudeSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_GEOMAGNETIC_ROTATION_VECTOR)
        attitudeSensorName = attitudeSensor?.stringType
        attitudeSensor?.let { sensor ->
            val periodMicroseconds = (configuration.attitudeSampleIntervalMilliseconds * MICROSECONDS_PER_MILLISECOND).toInt()
            sensorManager.registerListener(attitudeListener, sensor, periodMicroseconds, mainHandler)
        }
        val screenFilter = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_ON)
            addAction(Intent.ACTION_SCREEN_OFF)
        }
        ContextCompat.registerReceiver(context, screenReceiver, screenFilter, ContextCompat.RECEIVER_NOT_EXPORTED)
        eventLog.append(FieldRunEventTypes.SCREEN) { put("on", powerManager.isInteractive) }
        audioManager.registerAudioDeviceCallback(routeCallback, mainHandler)
        powerManager.addThermalStatusListener(context.mainExecutor, thermalListener)
        jobs += scope.launch(Dispatchers.Default) {
            while (isActive) {
                logThermal()
                delay(configuration.thermalPollIntervalMilliseconds)
            }
        }
        jobs += scope.launch(Dispatchers.Default) {
            while (isActive) {
                logVolumeIfChanged()
                delay(configuration.attitudeSampleIntervalMilliseconds)
            }
        }
    }

    private fun logRoute() {
        eventLog.append(FieldRunEventTypes.ROUTE_CHANGE) {
            put("input", routedInputDescription())
            put("output", routedOutputDescription())
        }
    }

    private fun logThermal() {
        val headroom = powerManager.getThermalHeadroom(THERMAL_FORECAST_SECONDS).takeIf { it.isFinite() }
        eventLog.append(FieldRunEventTypes.THERMAL) {
            put("status", powerManager.currentThermalStatus)
            put("headroom", headroom)
        }
        headroom?.let(onThermalHeadroom)
    }

    private fun logVolumeIfChanged() {
        val index = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
        if (index == lastVolumeIndex) return
        lastVolumeIndex = index
        eventLog.append(FieldRunEventTypes.VOLUME) {
            put("music_index", index)
            put("music_max", audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC))
        }
    }

    fun stop() {
        jobs.forEach { it.cancel() }
        jobs.clear()
        sensorManager.unregisterListener(attitudeListener)
        runCatching { context.unregisterReceiver(screenReceiver) }
        audioManager.unregisterAudioDeviceCallback(routeCallback)
        powerManager.removeThermalStatusListener(thermalListener)
    }
}
