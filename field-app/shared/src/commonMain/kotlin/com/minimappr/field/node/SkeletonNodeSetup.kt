package com.minimappr.field.node

import com.minimappr.field.FieldNodeConfiguration
import com.minimappr.field.ffi.FieldNodeSetup
import com.minimappr.field.ffi.FieldStage1Mode

/** Session ids are 16 bytes on the wire (`SessionId([u8; 16])`). */
private const val SESSION_ID_LENGTH_BYTES = 16

/**
 * Placeholder identity until QR join lands: this phone is node 2 and expects
 * a host at node 1. With no transport the host never answers, so the state
 * machine retries joins and then enters solo, which is the real behaviour
 * of a phone that cannot reach its host.
 */
private const val SKELETON_OWN_NODE_ID = 2u
private const val SKELETON_EXPECTED_HOST_NODE_ID = 1u

fun skeletonSoloNodeSetup(
    nodeConfiguration: FieldNodeConfiguration,
    platformName: String,
    appVersion: String,
): FieldNodeSetup = FieldNodeSetup(
    nodeId = SKELETON_OWN_NODE_ID,
    sessionId = ByteArray(SESSION_ID_LENGTH_BYTES),
    initialHostNodeId = SKELETON_EXPECTED_HOST_NODE_ID,
    platformName = platformName,
    appVersion = appVersion,
    canClassify = false,
    canHost = false,
    canEmitChirps = false,
    stage1Mode = FieldStage1Mode.BIRDS,
    rollingBufferSeconds = nodeConfiguration.rollingBufferSeconds.toUInt(),
    nowMs = 0uL,
)
