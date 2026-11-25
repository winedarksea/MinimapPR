package com.minimappr.field.transport

import com.minimappr.field.FieldSessionConfiguration
import com.minimappr.field.ffi.readLanEnvelopeRouting
import io.ktor.server.application.install
import io.ktor.server.cio.CIO
import io.ktor.server.cio.CIOApplicationEngine
import io.ktor.server.engine.EmbeddedServer
import io.ktor.server.engine.embeddedServer
import io.ktor.server.routing.routing
import io.ktor.server.websocket.DefaultWebSocketServerSession
import io.ktor.server.websocket.WebSockets
import io.ktor.server.websocket.webSocket
import io.ktor.websocket.Frame
import io.ktor.websocket.readBytes
import kotlinx.atomicfu.atomic
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.util.concurrent.ConcurrentHashMap

private class JoinerConnection(val connectionId: Long, val outbound: Channel<ByteArray>) {
    /** Learned from the connection's Hello; null until then. */
    @Volatile
    var nodeId: UInt? = null
}

/**
 * Host side of the star: a Ktor CIO WebSocket server. Each joiner
 * connection is named by the sender of its first Hello. Inbound envelopes
 * addressed to the host (or broadcast) go to the host's device; unicast to
 * another joiner is relayed. Joiner broadcasts are NOT relayed to other
 * joiners: a joiner's Hello is a broadcast, and a second joiner that has
 * not found its host would otherwise adopt the first joiner as host.
 *
 * Unsealed development transport (UI badge): no key exchange yet; the
 * device itself rejects a Hello with the wrong session id. TODO: join-code
 * SPAKE2 sealing (`session_crypto/join_code_key_exchange.rs`).
 */
class KtorHostWebSocketServer(
    private val ownNodeId: UInt,
    private val configuration: FieldSessionConfiguration,
    private val bindAddress: String = ANY_ADDRESS,
    private val requestedPort: Int = configuration.hostListenPort,
    private val diagnosticLog: (String) -> Unit = {},
) : FieldLanTransport {
    private val connections = ConcurrentHashMap<Long, JoinerConnection>()
    private val nextConnectionId = atomic(0L)
    private val envelopesSent = atomic(0L)
    private val envelopesReceived = atomic(0L)
    private val droppedQueueFull = atomic(0L)
    private val droppedUnknownRecipient = atomic(0L)
    @Volatile private var lastError: String? = null
    @Volatile private var listener: FieldLanTransportListener? = null
    private var server: EmbeddedServer<CIOApplicationEngine, CIOApplicationEngine.Configuration>? = null

    /** Port actually bound (differs from the request when it was 0). */
    @Volatile var boundPort: Int = 0
        private set

    override val endpointDescription: String get() = "ws://$bindAddress:$boundPort${configuration.webSocketPath}"

    override fun start(listener: FieldLanTransportListener) {
        this.listener = listener
        val embedded = embeddedServer(CIO, port = requestedPort, host = bindAddress) {
            install(WebSockets) {
                pingPeriodMillis = configuration.webSocketPingIntervalMilliseconds
                timeoutMillis = configuration.webSocketTimeoutMilliseconds
            }
            routing {
                webSocket(configuration.webSocketPath) { serveJoiner(this) }
            }
        }
        embedded.start(wait = false)
        server = embedded
        boundPort = runBlocking { embedded.engine.resolvedConnectors().first().port }
        diagnosticLog("host WebSocket server listening on $endpointDescription")
    }

    private suspend fun serveJoiner(session: DefaultWebSocketServerSession) {
        val connection = JoinerConnection(nextConnectionId.incrementAndGet(), Channel(configuration.outboundQueueCapacityEnvelopes))
        connections[connection.connectionId] = connection
        val writer = session.launch {
            for (envelope in connection.outbound) {
                session.send(Frame.Binary(true, envelope))
                envelopesSent.incrementAndGet()
            }
        }
        try {
            for (frame in session.incoming) {
                if (frame is Frame.Binary) routeInbound(connection, frame.readBytes())
            }
        } catch (failure: Exception) {
            lastError = "joiner ${connection.nodeId}: ${failure.message}"
        } finally {
            connections.remove(connection.connectionId)
            connection.outbound.close()
            writer.cancel()
            diagnosticLog("joiner connection ${connection.connectionId} (node ${connection.nodeId}) closed")
        }
    }

    private fun routeInbound(connection: JoinerConnection, envelope: ByteArray) {
        val routing = runCatching { readLanEnvelopeRouting(envelope) }.getOrNull() ?: return
        envelopesReceived.incrementAndGet()
        if (routing.isHello) connection.nodeId = routing.senderNodeId
        val recipient = routing.recipientNodeId
        if (recipient == null || recipient == ownNodeId) {
            deliverToOwnDevice(envelope)
            return
        }
        val target = connections.values.firstOrNull { it.nodeId == recipient }
        if (target == null) droppedUnknownRecipient.incrementAndGet() else enqueue(target, envelope)
    }

    private fun deliverToOwnDevice(envelope: ByteArray) {
        try {
            listener?.onEnvelopeReceived(envelope)
        } catch (failure: RuntimeException) {
            // A closed device during shutdown must not kill the server session.
            lastError = "device rejected envelope: ${failure.message}"
        }
    }

    private fun enqueue(connection: JoinerConnection, envelope: ByteArray) {
        if (connection.outbound.trySend(envelope).isFailure) droppedQueueFull.incrementAndGet()
    }

    override fun send(recipientNodeId: UInt?, envelope: ByteArray) {
        if (recipientNodeId == null) {
            connections.values.forEach { enqueue(it, envelope) }
            return
        }
        val target = connections.values.firstOrNull { it.nodeId == recipientNodeId }
        if (target == null) droppedUnknownRecipient.incrementAndGet() else enqueue(target, envelope)
    }

    override fun counters() = FieldLanTransportCounters(
        connectedPeerCount = connections.values.count { it.nodeId != null },
        envelopesSent = envelopesSent.value,
        envelopesReceived = envelopesReceived.value,
        envelopesDroppedQueueFull = droppedQueueFull.value,
        envelopesDroppedUnknownRecipient = droppedUnknownRecipient.value,
        lastError = lastError,
    )

    override fun close() {
        listener = null
        server?.stop(SHUTDOWN_GRACE_MILLISECONDS, SHUTDOWN_TIMEOUT_MILLISECONDS)
        server = null
    }

    companion object {
        const val ANY_ADDRESS = "0.0.0.0"
        private const val SHUTDOWN_GRACE_MILLISECONDS = 100L
        private const val SHUTDOWN_TIMEOUT_MILLISECONDS = 1_000L
    }
}
