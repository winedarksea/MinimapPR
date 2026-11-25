package com.minimappr.field.transport

import com.minimappr.field.FieldSessionConfiguration
import io.ktor.client.HttpClient
import io.ktor.client.plugins.websocket.webSocket
import io.ktor.websocket.Frame
import io.ktor.websocket.readBytes
import kotlinx.atomicfu.atomic
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.coroutines.cancellation.CancellationException

/** "host:port" → parts; null when malformed. Bracketed IPv6 is not needed on this LAN. */
fun parseHostAndPort(hostAndPort: String, defaultPort: Int): Pair<String, Int>? {
    val trimmed = hostAndPort.trim()
    if (trimmed.isEmpty()) return null
    val separator = trimmed.lastIndexOf(':')
    if (separator < 0) return trimmed to defaultPort
    val port = trimmed.substring(separator + 1).toIntOrNull() ?: return null
    return trimmed.substring(0, separator) to port
}

/**
 * Joiner side of the star: one WebSocket to the host, reconnecting after
 * [FieldSessionConfiguration.joinerReconnectDelayMilliseconds]. The device
 * learns link state through [FieldLanTransportListener.onLinkStateChanged],
 * which drives its Hello / clock-probe sequence.
 *
 * `httpClientFactory` picks the engine: OkHttp on Android (it accepts a
 * `Network.socketFactory`, which LocalOnlyHotspot will need), CIO on the
 * host JVM tests. The joiner sends everything to the host regardless of
 * the recipient: the host relays.
 */
class KtorJoinerWebSocketClient(
    private val hostName: String,
    private val hostPort: Int,
    private val configuration: FieldSessionConfiguration,
    private val httpClientFactory: () -> HttpClient,
    private val scope: CoroutineScope,
    private val diagnosticLog: (String) -> Unit = {},
) : FieldLanTransport {
    @Volatile private var outbound: Channel<ByteArray>? = null
    private var connectionJob: Job? = null
    private var client: HttpClient? = null
    private val envelopesSent = atomic(0L)
    private val envelopesReceived = atomic(0L)
    private val droppedQueueFull = atomic(0L)
    private val droppedWhileDisconnected = atomic(0L)
    private val reconnectAttempts = atomic(0L)
    @Volatile private var lastError: String? = null

    override val endpointDescription: String get() = "ws://$hostName:$hostPort${configuration.webSocketPath}"

    override fun start(listener: FieldLanTransportListener) {
        val httpClient = httpClientFactory()
        client = httpClient
        connectionJob = scope.launch(Dispatchers.IO) {
            while (isActive) {
                var linkWasUp = false
                try {
                    httpClient.webSocket(host = hostName, port = hostPort, path = configuration.webSocketPath) {
                        val queue = Channel<ByteArray>(configuration.outboundQueueCapacityEnvelopes)
                        outbound = queue
                        val writer = launch {
                            for (envelope in queue) {
                                send(Frame.Binary(true, envelope))
                                envelopesSent.incrementAndGet()
                            }
                        }
                        diagnosticLog("connected to $endpointDescription")
                        linkWasUp = true
                        notifyLink(listener, true)
                        try {
                            for (frame in incoming) {
                                if (frame !is Frame.Binary) continue
                                envelopesReceived.incrementAndGet()
                                deliver(listener, frame.readBytes())
                            }
                        } finally {
                            outbound = null
                            queue.close()
                            writer.cancel()
                        }
                    }
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (failure: Exception) {
                    lastError = failure.message ?: failure::class.simpleName
                } finally {
                    if (linkWasUp) notifyLink(listener, false)
                }
                reconnectAttempts.incrementAndGet()
                delay(configuration.joinerReconnectDelayMilliseconds)
            }
        }
    }

    private fun notifyLink(listener: FieldLanTransportListener, linkUp: Boolean) {
        try {
            listener.onLinkStateChanged(linkUp)
        } catch (failure: RuntimeException) {
            lastError = "device rejected link change: ${failure.message}"
        }
    }

    private fun deliver(listener: FieldLanTransportListener, envelope: ByteArray) {
        try {
            listener.onEnvelopeReceived(envelope)
        } catch (failure: RuntimeException) {
            lastError = "device rejected envelope: ${failure.message}"
        }
    }

    override fun send(recipientNodeId: UInt?, envelope: ByteArray) {
        val queue = outbound
        if (queue == null) {
            droppedWhileDisconnected.incrementAndGet()
            return
        }
        if (queue.trySend(envelope).isFailure) droppedQueueFull.incrementAndGet()
    }

    override fun counters() = FieldLanTransportCounters(
        connectedPeerCount = if (outbound != null) 1 else 0,
        envelopesSent = envelopesSent.value,
        envelopesReceived = envelopesReceived.value,
        envelopesDroppedQueueFull = droppedQueueFull.value,
        envelopesDroppedUnknownRecipient = droppedWhileDisconnected.value,
        reconnectAttempts = reconnectAttempts.value,
        lastError = lastError,
    )

    override fun close() {
        connectionJob?.cancel()
        connectionJob = null
        client?.close()
        client = null
    }
}
