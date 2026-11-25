package com.minimappr.field.session

import com.minimappr.field.ffi.FieldLocalRoundStatus
import com.minimappr.field.ffi.FieldPeerSyncStatus
import com.minimappr.field.ffi.FieldSlotArrivalStatus
import com.minimappr.field.ffi.FieldSyncStatus
import com.minimappr.field.json.JsonObjectBuilder
import com.minimappr.field.json.jsonObject

/**
 * `SYNC_STATUS {json}`: every FieldSyncStatus field in snake_case, the
 * contract the laptop e2e script polls through dumpsys. Extra shell-side
 * keys go under `shell` so they never collide with Rust field names.
 */
fun FieldSyncStatus.toSnakeCaseJson(shellExtras: (JsonObjectBuilder.() -> Unit)? = null): String = jsonObject {
    put("role", role.name.lowercase())
    put("node_id", nodeId.toLong())
    put("host_node_id", hostNodeId?.toLong())
    put("session_id_hex", sessionIdHex)
    put("joined", joined)
    put("session_clock_aligned", sessionClockAligned)
    put("session_clock_offset_ms", sessionClockOffsetMs)
    put("session_clock_best_round_trip_ms", sessionClockBestRoundTripMs)
    put("clock_probe_exchanges_accepted", clockProbeExchangesAccepted.toLong())
    put("own_clock_in_sync", ownClockInSync)
    put("own_clock_offset_us", ownClockOffsetUs)
    put("own_clock_rate_ppm", ownClockRatePpm)
    put("capture_anchor_count", captureAnchorCount.toLong())
    put("capture_rate_offset_ppm", captureRateOffsetPpm)
    put("capture_anchor_residual_us", captureAnchorResidualUs)
    put("capture_gap_count", captureGapCount.toLong())
    put("rounds_started", roundsStarted.toLong())
    put("rounds_completed", roundsCompleted.toLong())
    put("rounds_heard", roundsHeard.toLong())
    put("emissions_requested", emissionsRequested.toLong())
    put("emissions_played", emissionsPlayed.toLong())
    put("emissions_skipped", emissionsSkipped.toLong())
    putLongs("muted_node_ids", mutedNodeIds.map { it.toLong() })
    val round = lastLocalRound
    if (round == null) put("last_local_round", null as String?) else putObject("last_local_round") { localRound(round) }
    putObjects("peers", peers.map { peer -> { peerStatus(peer) } })
    put("transport_unsealed", transportUnsealed)
    shellExtras?.let { putObject("shell", it) }
}

private fun JsonObjectBuilder.localRound(round: FieldLocalRoundStatus) {
    put("round_id", round.roundId.toLong())
    put("own_emission_played", round.ownEmissionPlayed)
    put("own_chirp_inaudible", round.ownChirpInaudible)
    put("unsearchable_slot_count", round.unsearchableSlotCount.toLong())
    putObjects("slots", round.slots.map { slot -> { slotArrival(slot) } })
}

private fun JsonObjectBuilder.slotArrival(slot: FieldSlotArrivalStatus) {
    put("emitter_node_id", slot.emitterNodeId.toLong())
    put("is_self", slot.isSelf)
    put("detected", slot.detected)
    put("snr_db", slot.snrDb)
    put("doppler_skew_samples", slot.dopplerSkewSamples)
    put("peak_fraction", slot.peakFraction)
}

private fun JsonObjectBuilder.peerStatus(peer: FieldPeerSyncStatus) {
    put("node_id", peer.nodeId.toLong())
    put("range_m", peer.rangeM)
    put("round_clock_offset_us", peer.roundClockOffsetUs)
    put("round_clock_offset_sigma_us", peer.roundClockOffsetSigmaUs)
    put("filtered_clock_offset_us", peer.filteredClockOffsetUs)
    put("filtered_clock_offset_sigma_us", peer.filteredClockOffsetSigmaUs)
    put("rate_offset_ppm", peer.rateOffsetPpm)
    put("last_innovation_sigma", peer.lastInnovationSigma)
    put("last_filter_outcome", peer.lastFilterOutcome)
    put("last_round_id", peer.lastRoundId.toLong())
    put("is_reference", peer.isReference)
    put("reference_node_id", peer.referenceNodeId.toLong())
    put("round_reference_time_s", peer.roundReferenceTimeS)
}
