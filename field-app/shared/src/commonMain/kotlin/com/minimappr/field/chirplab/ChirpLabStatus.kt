package com.minimappr.field.chirplab

import com.minimappr.field.json.jsonObject

/** Live chirp lab readout for the debug screen and `CHIRP_LAB_STATUS`. */
data class ChirpLabStatus(
    val runId: String = "",
    val role: String = "",
    val running: Boolean = false,
    val emissions: Long = 0,
    val emissionsSkipped: Long = 0,
    val selfChirpsDetected: Long = 0,
    val selfChirpsMissed: Long = 0,
    val lastSelfSnrDb: Double? = null,
    val lastSelfPeakFraction: Float? = null,
    /** Detected minus predicted capture frame of the last self chirp (anchor quality). */
    val lastSelfArrivalErrorFrames: Double? = null,
    val captureFrames: Long = 0,
    val writerOverflows: Long = 0,
    val captureGaps: Long = 0,
    val playbackUnderruns: Int = 0,
    val captureSourceName: String? = null,
    val playbackPerformanceMode: String? = null,
    val outputDirectory: String? = null,
    val failure: String? = null,
) {
    fun toJson(): String = jsonObject {
        put("run_id", runId)
        put("role", role)
        put("running", running)
        put("emissions", emissions)
        put("emissions_skipped", emissionsSkipped)
        put("self_chirps_detected", selfChirpsDetected)
        put("self_chirps_missed", selfChirpsMissed)
        put("last_self_snr_db", lastSelfSnrDb)
        put("last_self_peak_fraction", lastSelfPeakFraction)
        put("last_self_arrival_error_frames", lastSelfArrivalErrorFrames)
        put("capture_frames", captureFrames)
        put("writer_overflows", writerOverflows)
        put("capture_gaps", captureGaps)
        put("playback_underruns", playbackUnderruns)
        put("capture_source", captureSourceName)
        put("playback_performance_mode", playbackPerformanceMode)
        put("output_directory", outputDirectory)
        put("failure", failure)
    }
}
