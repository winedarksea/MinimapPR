package com.minimappr.field.session

/**
 * A host found by NSD (`_minimappr-field._tcp`). `sessionIdHex` (TXT `sid`)
 * lets the joiner insist on that session instead of adopting any answer.
 */
data class DiscoveredFieldHost(
    val serviceName: String,
    val hostAddress: String,
    val port: Int,
    val sessionIdHex: String?,
    val hostNodeId: UInt?,
    val protocolVersion: String?,
)
