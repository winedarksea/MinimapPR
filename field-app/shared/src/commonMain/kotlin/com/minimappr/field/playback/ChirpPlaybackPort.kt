package com.minimappr.field.playback

/**
 * Outcome of one scheduled chirp. `played = false` covers late targets, a
 * missing timestamp and a stopped engine. `firstFrameLocalNs` is the
 * predicted presentation of the chirp's first frame (the target when not
 * played, which the device ignores).
 */
fun interface ChirpEmissionOutcomeListener {
    fun onChirpEmissionOutcome(played: Boolean, firstFrameLocalNs: Long)
}

/**
 * What the session controller needs from a speaker. Contract: every call
 * produces exactly one outcome callback, possibly synchronously, possibly
 * from the playback thread.
 */
interface ChirpPlaybackPort {
    /** `emissionId` is only a label for the run log (events.jsonl `emission_index`). */
    fun scheduleChirpAtLocalNs(
        emissionId: Long,
        targetPresentationNs: Long,
        pcm16StereoLittleEndian: ByteArray,
        outcomeListener: ChirpEmissionOutcomeListener,
    )
}
