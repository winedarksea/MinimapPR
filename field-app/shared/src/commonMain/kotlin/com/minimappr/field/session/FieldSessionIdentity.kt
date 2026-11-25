package com.minimappr.field.session

import com.minimappr.field.ffi.FieldDeviceRole
import com.minimappr.field.ffi.FieldDeviceSetup
import com.minimappr.field.ffi.FieldStage1Mode
import kotlin.uuid.ExperimentalUuidApi
import kotlin.uuid.Uuid

/** `SessionId([u8; 16])` on the wire. */
const val FIELD_SESSION_ID_LENGTH_BYTES = 16

/** NodeId 0 is never assigned and u32::MAX is `NodeId::BROADCAST`. */
private val RESERVED_NODE_IDS = setOf(0u, UInt.MAX_VALUE)
private const val BYTES_PER_NODE_ID = 4
private const val BITS_PER_BYTE = 8
private const val BYTE_MASK = 0xFF

/**
 * Who this phone is in a LAN session. Replaces the hard-coded skeleton ids.
 *
 * Randomness: `Uuid.random()` is backed by the platform CSPRNG on every
 * Kotlin target (SecureRandom on the JVM/Android), so commonMain needs no
 * expect/actual. A v4 UUID's first 6 bytes are fully random, which is where
 * the node id is taken from; the session id keeps the 6 version/variant
 * bits, harmless for an identifier that only has to be unique.
 *
 * Role: the joiner never hosts in this pass (plan finding 8), so a host
 * silence cannot make two phones both claim the session.
 */
@OptIn(ExperimentalUuidApi::class)
data class FieldSessionIdentity(
    val nodeId: UInt,
    val sessionId: ByteArray,
    val role: FieldDeviceRole,
) {
    init {
        require(sessionId.size == FIELD_SESSION_ID_LENGTH_BYTES) { "session id must be 16 bytes" }
        require(nodeId !in RESERVED_NODE_IDS) { "node id $nodeId is reserved" }
    }

    val sessionIdHex: String get() = sessionId.toLowerHex()

    fun deviceSetup(
        platformName: String,
        appVersion: String,
        rollingBufferSeconds: Int,
        canEmitChirps: Boolean = true,
    ): FieldDeviceSetup = FieldDeviceSetup(
        nodeId = nodeId,
        sessionId = sessionId,
        role = role,
        platformName = platformName,
        appVersion = appVersion,
        canClassify = false,
        canEmitChirps = canEmitChirps,
        stage1Mode = FieldStage1Mode.BIRDS,
        rollingBufferSeconds = rollingBufferSeconds.toUInt(),
    )

    override fun equals(other: Any?): Boolean =
        other is FieldSessionIdentity && nodeId == other.nodeId && role == other.role && sessionId.contentEquals(other.sessionId)

    override fun hashCode(): Int = (nodeId.hashCode() * 31 + role.hashCode()) * 31 + sessionId.contentHashCode()

    companion object {
        fun newNodeId(): UInt {
            while (true) {
                val bytes = Uuid.random().toByteArray()
                var nodeId = 0u
                for (index in 0 until BYTES_PER_NODE_ID) {
                    nodeId = (nodeId shl BITS_PER_BYTE) or (bytes[index].toInt() and BYTE_MASK).toUInt()
                }
                if (nodeId !in RESERVED_NODE_IDS) return nodeId
            }
        }

        fun newHost(): FieldSessionIdentity =
            FieldSessionIdentity(newNodeId(), Uuid.random().toByteArray(), FieldDeviceRole.HOST)

        /**
         * `expectedSessionId` from discovery (NSD TXT `sid`), or null for a
         * typed IP:port join, which adopts whatever the host answers with.
         */
        fun newJoiner(expectedSessionId: ByteArray?): FieldSessionIdentity =
            FieldSessionIdentity(newNodeId(), expectedSessionId ?: ByteArray(FIELD_SESSION_ID_LENGTH_BYTES), FieldDeviceRole.JOINER)
    }
}

private const val HEX_RADIX = 16
private const val HEX_DIGITS_PER_BYTE = 2

fun ByteArray.toLowerHex(): String = joinToString("") { (it.toInt() and BYTE_MASK).toString(HEX_RADIX).padStart(HEX_DIGITS_PER_BYTE, '0') }

/** Parses a 32-hex-digit session id; null when malformed. */
fun parseSessionIdHex(hex: String?): ByteArray? {
    if (hex == null || hex.length != FIELD_SESSION_ID_LENGTH_BYTES * HEX_DIGITS_PER_BYTE) return null
    return runCatching {
        ByteArray(FIELD_SESSION_ID_LENGTH_BYTES) { index ->
            hex.substring(index * HEX_DIGITS_PER_BYTE, index * HEX_DIGITS_PER_BYTE + HEX_DIGITS_PER_BYTE).toInt(HEX_RADIX).toByte()
        }
    }.getOrNull()
}
