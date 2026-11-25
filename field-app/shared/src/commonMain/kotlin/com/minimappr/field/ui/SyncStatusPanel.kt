package com.minimappr.field.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.minimappr.field.ffi.FieldPeerSyncStatus
import com.minimappr.field.session.FieldSessionStatus

private val BADGE_PADDING = 6.dp
private const val RANGE_DECIMALS = 3
private const val OFFSET_DECIMALS = 1
private const val MILLISECOND_DECIMALS = 2

/**
 * Live sync readout: the unsealed-transport badge (always visible: there is
 * no key exchange yet), session clock, round counters, this phone's last
 * round and, on the host, the per-peer range / clock offset / rate table.
 * Values are raw (no speaker/mic offset correction yet).
 */
@Composable
fun SyncStatusPanel(session: FieldSessionStatus) {
    Column(verticalArrangement = Arrangement.spacedBy(ROW_SPACING)) {
        Text(
            "UNSEALED DEV TRANSPORT",
            color = MaterialTheme.colorScheme.onError,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.background(MaterialTheme.colorScheme.error).padding(BADGE_PADDING),
        )
        Text("Endpoint: ${session.transportEndpoint ?: "-"} (peers ${session.transport.connectedPeerCount})")
        val sync = session.sync ?: run {
            Text("Starting…")
            return
        }
        Text("Role ${sync.role.name.lowercase()}, node ${sync.nodeId}, host ${sync.hostNodeId ?: "-"}")
        Text("Session ${sync.sessionIdHex.ifEmpty { "-" }}")
        Text("Joined: ${sync.joined}; clock aligned: ${sync.sessionClockAligned}")
        Text(
            "Session clock offset ${sync.sessionClockOffsetMs?.formatDecimals(MILLISECOND_DECIMALS) ?: "-"} ms, " +
                "best RTT ${sync.sessionClockBestRoundTripMs?.formatDecimals(MILLISECOND_DECIMALS) ?: "-"} ms " +
                "(${sync.clockProbeExchangesAccepted} probes)",
        )
        Text("Rounds started ${sync.roundsStarted}, completed ${sync.roundsCompleted}, heard ${sync.roundsHeard}")
        Text("Emissions requested ${sync.emissionsRequested}, played ${sync.emissionsPlayed}, skipped ${sync.emissionsSkipped}")
        if (sync.mutedNodeIds.isNotEmpty()) Text("Muted: ${sync.mutedNodeIds}", color = MaterialTheme.colorScheme.error)
        sync.lastLocalRound?.let { round ->
            val slots = round.slots.joinToString { slot ->
                "${if (slot.isSelf) "self" else slot.emitterNodeId.toString()}:" +
                    (slot.snrDb?.formatDecimals(OFFSET_DECIMALS)?.let { "$it dB" } ?: "miss")
            }
            Text("Last round ${round.roundId}: $slots")
        }
        sync.peers.forEach { PeerRow(it) }
        session.noticeHistory.lastOrNull()?.let { Text("Last notice: $it") }
        session.transport.lastError?.let { Text("Transport: $it", color = MaterialTheme.colorScheme.error) }
    }
}

@Composable
private fun PeerRow(peer: FieldPeerSyncStatus) {
    Text(
        "Peer ${peer.nodeId}${if (peer.isReference) " (ref)" else ""}: " +
            "range ${peer.rangeM?.formatDecimals(RANGE_DECIMALS) ?: "-"} m, " +
            "θ ${peer.roundClockOffsetUs?.formatDecimals(OFFSET_DECIMALS) ?: "-"} µs " +
            "(filt ${peer.filteredClockOffsetUs?.formatDecimals(OFFSET_DECIMALS) ?: "-"}), " +
            "rate ${peer.rateOffsetPpm?.formatDecimals(OFFSET_DECIMALS) ?: "-"} ppm, ${peer.lastFilterOutcome}",
    )
}
