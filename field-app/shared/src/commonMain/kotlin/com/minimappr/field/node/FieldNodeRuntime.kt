package com.minimappr.field.node

import com.minimappr.field.FieldNodeConfiguration
import com.minimappr.field.capture.CaptureBlock
import com.minimappr.field.capture.CaptureBlockSink
import com.minimappr.field.capture.DropWhileBusyBackpressureGate
import com.minimappr.field.capture.toPcm16LittleEndianBytes
import com.minimappr.field.ffi.FieldMeshNode
import com.minimappr.field.ffi.FieldNodeAction
import com.minimappr.field.ffi.FieldNodeSetup
import com.minimappr.field.ffi.FieldStage1Event
import com.minimappr.field.ffi.describeFrame
import com.minimappr.field.ffi.fieldFfiSurfaceVersion
import com.minimappr.field.ffi.fieldProtocolVersion
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlin.coroutines.coroutineContext
import kotlin.time.TimeSource

/** Where encoded frames go; the relay / hotspot transports implement this. */
fun interface FieldOutboundFrameTransport {
    fun sendFrame(recipientNodeId: UInt?, frame: ByteArray, isBulkTraffic: Boolean)
}

/** Skeleton transport: no relay or hotspot yet, frames are only counted. */
val discardingOutboundFrameTransport = FieldOutboundFrameTransport { _, _, _ -> }

/**
 * Glue between capture, the Rust mesh node and a transport.
 *
 * Every capture block reaches the Rust rolling buffer (no backpressure there:
 * a dropped block is a TDOA gap). Only the skippable classifier stage sits
 * behind [DropWhileBusyBackpressureGate]. The state machine is stepped by
 * [runTickLoop] on the coarse session clock; it never reads wall time.
 */
class FieldNodeRuntime(
    nodeSetup: FieldNodeSetup,
    private val nodeConfiguration: FieldNodeConfiguration,
    private val outboundFrameTransport: FieldOutboundFrameTransport = discardingOutboundFrameTransport,
) : CaptureBlockSink, AutoCloseable {
    private val sessionClockOrigin = TimeSource.Monotonic.markNow()
    private val meshNode = FieldMeshNode(nodeSetup)
    private val classifierBackpressureGate = DropWhileBusyBackpressureGate()

    private val mutableStatus = MutableStateFlow(
        FieldNodeStatus(
            ffiSurfaceVersion = fieldFfiSurfaceVersion().toLong(),
            protocolVersionLabel = fieldProtocolVersion().let { "${it.majorVersion}.${it.minorVersion}" },
        )
    )
    val status: StateFlow<FieldNodeStatus> = mutableStatus.asStateFlow()

    private fun sessionClockMilliseconds(): ULong =
        sessionClockOrigin.elapsedNow().inWholeMilliseconds.toULong()

    override fun onCaptureBlock(block: CaptureBlock) {
        val stage1Events = meshNode.pushCaptureBlock(
            block.firstDeviceSampleIndex.toULong(),
            block.samples.toPcm16LittleEndianBytes(),
        )
        var triggersOpened = 0L
        var onsetsDetected = 0L
        for (event in stage1Events) {
            when (event) {
                is FieldStage1Event.TriggerOpened -> {
                    triggersOpened += 1
                    offerWindowToClassifier()
                }
                is FieldStage1Event.Onset -> onsetsDetected += 1
                is FieldStage1Event.TriggerClosed -> Unit
            }
        }
        mutableStatus.update { current ->
            current.copy(
                capturedSampleCount = meshNode.nextDeviceSampleIndex().toLong(),
                captureGapCount = meshNode.rollingBufferGapCount().toLong(),
                stage1TriggersOpened = current.stage1TriggersOpened + triggersOpened,
                stage1OnsetsDetected = current.stage1OnsetsDetected + onsetsDetected,
                classifierWindowsAccepted = classifierBackpressureGate.acceptedWindowCount,
                classifierWindowsDropped = classifierBackpressureGate.droppedWindowCount,
            )
        }
    }

    /** Placeholder until BirdNET+ V3.0 lands: acquires and releases at once. */
    private fun offerWindowToClassifier() {
        if (classifierBackpressureGate.tryAcquire()) {
            classifierBackpressureGate.release()
        }
    }

    /** Called by the capture loop when the OS read failed (possible lost audio). */
    fun onCaptureGap() = dispatchActions(meshNode.reportCaptureGap(sessionClockMilliseconds()))

    fun tick() = dispatchActions(meshNode.tick(sessionClockMilliseconds()))

    fun onFrameReceived(fromNodeId: UInt, frame: ByteArray) =
        dispatchActions(meshNode.receiveFrame(sessionClockMilliseconds(), fromNodeId, frame))

    suspend fun runTickLoop() {
        while (coroutineContext.isActive) {
            tick()
            delay(nodeConfiguration.stateMachineTickIntervalMilliseconds)
        }
    }

    private fun dispatchActions(actions: List<FieldNodeAction>) {
        for (action in actions) {
            when (action) {
                is FieldNodeAction.SendFrame -> {
                    outboundFrameTransport.sendFrame(action.recipientNodeId, action.frame, action.isBulkTraffic)
                    val messageKindId = describeFrame(action.frame).messageKindId.toInt()
                    mutableStatus.update { current ->
                        current.copy(
                            outboundFramesSent = current.outboundFramesSent + 1,
                            lastOutboundMessageKindId = messageKindId,
                        )
                    }
                }
                is FieldNodeAction.UserNotice ->
                    mutableStatus.update { it.copy(lastUserNotice = action.notice.name) }
                // Chirp TX/RX and host takeover arrive with later workstreams.
                is FieldNodeAction.EmitChirp,
                is FieldNodeAction.ListenForChirpRound,
                is FieldNodeAction.BecomeHost -> Unit
            }
        }
    }

    override fun close() {
        meshNode.close()
    }
}
