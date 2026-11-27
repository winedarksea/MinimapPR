package com.minimappr.field.session

import com.minimappr.field.FieldSessionConfiguration
import com.minimappr.field.capture.CaptureBlock
import com.minimappr.field.capture.CaptureBlockSink
import com.minimappr.field.capture.toPcm16LittleEndianBytes
import com.minimappr.field.ffi.FieldDeviceAction
import com.minimappr.field.ffi.FieldDeviceSetup
import com.minimappr.field.ffi.FieldMeshDevice
import com.minimappr.field.ffi.FieldMeshTuning
import com.minimappr.field.ffi.FieldStage1Event
import com.minimappr.field.ffi.FieldSyncStatus
import com.minimappr.field.playback.ChirpPlaybackPort
import com.minimappr.field.recording.DiscardingFieldRunEventLog
import com.minimappr.field.recording.FieldRunEventLog
import com.minimappr.field.recording.FieldRunEventTypes
import com.minimappr.field.transport.FieldLanTransport
import com.minimappr.field.transport.FieldLanTransportCounters
import com.minimappr.field.transport.FieldLanTransportListener
import kotlinx.atomicfu.atomic
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/** Snapshot for the session screen and `SYNC_STATUS` dumpsys line. */
data class FieldSessionStatus(
    val sync: FieldSyncStatus? = null,
    val transport: FieldLanTransportCounters = FieldLanTransportCounters(),
    val transportEndpoint: String? = null,
    val capturedSampleCount: Long = 0,
    val stage1TriggersOpened: Long = 0,
    val stage1OnsetsDetected: Long = 0,
    val emissionsScheduled: Long = 0,
    val emissionsRejected: Long = 0,
    val noticeHistory: List<String> = emptyList(),
    val isHost: Boolean = false,
    val failure: String? = null,
)

private const val NOTICE_HISTORY_LENGTH = 8

/**
 * Owns one [FieldMeshDevice] and wires it to capture (pushed in), a LAN
 * transport, a chirp speaker and a ~20 ms tick loop. Platform-neutral: the
 * Android service and the JVM loopback test drive the same class.
 *
 * Time: every device call takes [monotonicNanoTime] (System.nanoTime, the
 * clock family of AudioTimestamp TIMEBASE_MONOTONIC), never wall time.
 *
 * Threading: the Rust device serialises calls with its own mutex, so
 * capture, transport and tick threads call it directly. Action dispatch is
 * thread-safe: transport send enqueues; playback scheduling locks inside
 * the engine; status goes through a StateFlow.
 */
class FieldSessionController(
    deviceSetup: FieldDeviceSetup,
    tuning: FieldMeshTuning,
    private val sessionConfiguration: FieldSessionConfiguration,
    private val transport: FieldLanTransport?,
    private val playback: ChirpPlaybackPort?,
    private val monotonicNanoTime: () -> Long,
    private val eventLog: FieldRunEventLog = DiscardingFieldRunEventLog,
) : CaptureBlockSink, AutoCloseable {
    val device = FieldMeshDevice(deviceSetup, tuning, monotonicNanoTime())
    private val mutableStatus = MutableStateFlow(FieldSessionStatus(isHost = deviceSetup.role == com.minimappr.field.ffi.FieldDeviceRole.HOST))
    val status: StateFlow<FieldSessionStatus> = mutableStatus.asStateFlow()
    private val emissionsScheduled = atomic(0L)
    private val emissionsRejected = atomic(0L)
    private val stage1TriggersOpened = atomic(0L)
    private val stage1OnsetsDetected = atomic(0L)
    private val closed = atomic(false)
    private var lastLoggedRoundId: UInt? = null
    private val jobs = mutableListOf<Job>()

    private val transportListener = object : FieldLanTransportListener {
        override fun onEnvelopeReceived(envelope: ByteArray) {
            if (closed.value) return
            val actions = runCatching { device.receiveEnvelope(monotonicNanoTime(), envelope) }
                .onFailure { failure -> recordNotice("bad envelope: ${failure.message}") }
                .getOrNull() ?: return
            dispatch(actions)
        }

        override fun onLinkStateChanged(linkUp: Boolean) {
            if (closed.value) return
            dispatch(device.setTransportLinkUp(monotonicNanoTime(), linkUp))
        }
    }

    fun start(scope: CoroutineScope) {
        transport?.start(transportListener)
        mutableStatus.update { it.copy(transportEndpoint = transport?.endpointDescription) }
        jobs += scope.launch {
            while (isActive && !closed.value) {
                dispatch(device.tick(monotonicNanoTime()))
                delay(sessionConfiguration.deviceTickIntervalMilliseconds)
            }
        }
        jobs += scope.launch {
            while (isActive && !closed.value) {
                refreshStatus()
                delay(sessionConfiguration.statusRefreshIntervalMilliseconds)
            }
        }
    }

    override fun onCaptureBlock(block: CaptureBlock) {
        if (closed.value) return
        val events = device.pushCaptureBlock(block.firstDeviceSampleIndex.toULong(), block.samples.toPcm16LittleEndianBytes())
        for (event in events) {
            when (event) {
                is FieldStage1Event.TriggerOpened -> stage1TriggersOpened.incrementAndGet()
                is FieldStage1Event.Onset -> stage1OnsetsDetected.incrementAndGet()
                is FieldStage1Event.TriggerClosed -> Unit
            }
        }
    }

    /** AudioRecord.getTimestamp on the device-sample-index timeline. */
    fun onCaptureTimestamp(framePosition: Long, monotonicNs: Long) {
        if (!closed.value) device.reportCaptureTimestamp(framePosition.toULong(), monotonicNs)
    }

    fun onCaptureGap() {
        if (!closed.value) dispatch(device.reportCaptureGap(monotonicNanoTime()))
    }

    fun onThermalHeadroom(headroomFraction: Float) {
        if (!closed.value) dispatch(device.reportThermalHeadroom(monotonicNanoTime(), headroomFraction))
    }

    /**
     * Rotation-vector attitude (body → ENU, magnetic north). The node attaches
     * the latest fresh one to its chirp-round reports so the host can correct
     * or bound the speaker–mic geometry of each ranged pair.
     */
    fun onAttitude(quaternionWxyz: List<Float>, headingAccuracyRadians: Float?) {
        if (!closed.value) dispatch(device.reportAttitude(monotonicNanoTime(), quaternionWxyz, headingAccuracyRadians, false))
    }

    private fun dispatch(actions: List<FieldDeviceAction>) {
        for (action in actions) {
            when (action) {
                is FieldDeviceAction.SendEnvelope -> transport?.send(action.recipientNodeId, action.envelope)
                is FieldDeviceAction.EmitChirp -> emitChirp(action)
                is FieldDeviceAction.UserNotice -> recordNotice(action.notice.name)
                is FieldDeviceAction.HostRoleChanged -> mutableStatus.update { it.copy(isHost = action.isHost) }
            }
        }
    }

    /** Exactly one outcome per EmitChirp, also without a speaker. */
    private fun emitChirp(action: FieldDeviceAction.EmitChirp) {
        val port = playback
        if (port == null || closed.value) {
            emissionsRejected.incrementAndGet()
            device.reportChirpEmissionOutcome(action.emissionId, false, action.emitAtLocalNs)
            return
        }
        port.scheduleChirpAtLocalNs(action.emissionId.toLong(), action.emitAtLocalNs, action.pcm16StereoLittleEndian) { played, firstFrameLocalNs ->
            if (played) emissionsScheduled.incrementAndGet() else emissionsRejected.incrementAndGet()
            if (!closed.value) device.reportChirpEmissionOutcome(action.emissionId, played, firstFrameLocalNs)
        }
    }

    private fun recordNotice(notice: String) {
        mutableStatus.update { it.copy(noticeHistory = (it.noticeHistory + notice).takeLast(NOTICE_HISTORY_LENGTH)) }
    }

    fun refreshStatus() {
        if (closed.value) return
        val sync = device.syncStatus()
        logNewLocalRound(sync)
        mutableStatus.update { current ->
            current.copy(
                sync = sync,
                transport = transport?.counters() ?: FieldLanTransportCounters(),
                capturedSampleCount = device.nextDeviceSampleIndex().toLong(),
                stage1TriggersOpened = stage1TriggersOpened.value,
                stage1OnsetsDetected = stage1OnsetsDetected.value,
                emissionsScheduled = emissionsScheduled.value,
                emissionsRejected = emissionsRejected.value,
            )
        }
    }

    private fun logNewLocalRound(sync: FieldSyncStatus) {
        val round = sync.lastLocalRound ?: return
        if (round.roundId == lastLoggedRoundId) return
        lastLoggedRoundId = round.roundId
        val selfSlot = round.slots.firstOrNull { it.isSelf }
        eventLog.append(FieldRunEventTypes.SESSION_ROUND) {
            put("round_id", round.roundId.toLong())
            put("self_snr_db", selfSlot?.snrDb)
            put("self_detected", selfSlot?.detected)
            put("own_emission_played", round.ownEmissionPlayed)
            put("detected_slot_count", round.slots.count { it.detected })
            put("slot_count", round.slots.size)
        }
    }

    override fun close() {
        if (!closed.compareAndSet(expect = false, update = true)) return
        jobs.forEach { it.cancel() }
        transport?.close()
        device.close()
    }
}
