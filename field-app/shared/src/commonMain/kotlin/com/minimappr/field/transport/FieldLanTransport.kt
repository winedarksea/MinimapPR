package com.minimappr.field.transport

/** Callbacks from a LAN transport; may arrive on any thread. */
interface FieldLanTransportListener {
    fun onEnvelopeReceived(envelope: ByteArray)

    /** Joiner: the WebSocket to the host opened or closed. Hosts never call it. */
    fun onLinkStateChanged(linkUp: Boolean)
}

/** Counters for the status panel and dumpsys. */
data class FieldLanTransportCounters(
    val connectedPeerCount: Int = 0,
    val envelopesSent: Long = 0,
    val envelopesReceived: Long = 0,
    val envelopesDroppedQueueFull: Long = 0,
    val envelopesDroppedUnknownRecipient: Long = 0,
    val reconnectAttempts: Long = 0,
    val lastError: String? = null,
)

/**
 * Star-topology LAN transport carrying unsealed LAN envelopes (minimappr-ffi
 * `lan_envelope`). Host: a WebSocket server routing by recipient node id.
 * Joiner: one client connection to the host. [send] never blocks: it
 * enqueues or drops-and-counts, because it is called from the device tick.
 */
interface FieldLanTransport : AutoCloseable {
    fun start(listener: FieldLanTransportListener)

    /** `recipientNodeId` null = every connected phone (host) / the host (joiner). */
    fun send(recipientNodeId: UInt?, envelope: ByteArray)

    fun counters(): FieldLanTransportCounters

    /** Human-readable endpoint for the UI, e.g. "ws://192.168.1.30:47480". */
    val endpointDescription: String
}
