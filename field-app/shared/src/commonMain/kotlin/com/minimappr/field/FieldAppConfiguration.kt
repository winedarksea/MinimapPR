package com.minimappr.field

/**
 * Operational constants for the shell, in one place (AGENTS.md: no magic
 * numbers). Mesh/DSP tunables live in minimappr-core (reached through
 * `FieldMeshTuning`), not here.
 */
data class FieldCaptureConfiguration(
    /** Device clock rate every TDOA computation in minimappr-core assumes. */
    val captureSampleRateHz: Int = 48_000,
    /** 20 ms blocks: one Stage-1 hop per read. */
    val captureBlockSampleCount: Int = 960,
    /**
     * OS capture buffer, in blocks. Large enough to ride out a GC pause on the
     * capture thread without an overrun (which would be a TDOA gap).
     */
    val audioRecordBufferBlockCount: Int = 16,
    /**
     * How long stopping capture waits for the read thread to exit before
     * releasing the recorder anyway. AudioRecord.stop() unblocks a blocking
     * read, so this is normally one 20 ms block; the bound keeps a wedged
     * audio HAL from hanging the main thread into an ANR.
     */
    val captureStopJoinTimeoutMilliseconds: Long = 500,
    /** AudioRecord.getTimestamp cadence: anchors for the device and the run log. */
    val captureTimestampIntervalMilliseconds: Long = 1_000,
    /** Duplex start waits this long for the first valid capture timestamp. */
    val firstCaptureTimestampTimeoutMilliseconds: Long = 3_000,
    /**
     * Capture timestamps advancing this much further than their frames is a
     * silent drop (`SilentCaptureDropDetector`). Above the Pixel 10 Pro XL's
     * 5.33 ms (one HAL burst) timestamp wander; matches the analyser's
     * `capture_timestamp_step_threshold_s` and the core anchor's jump gate.
     */
    val silentCaptureDropThresholdMilliseconds: Long = 15,
)

data class FieldPlaybackConfiguration(
    val playbackSampleRateHz: Int = 48_000,
    /** 5 ms of stereo frames per AudioTrack write: splice granularity only. */
    val writeBlockFrameCount: Int = 240,
    /** AudioTrack buffer as a multiple of getMinBufferSize (underrun headroom). */
    val audioTrackBufferMinimumSizeMultiplier: Int = 2,
    /**
     * Floor on the AudioTrack buffer. Latency is irrelevant here (chirps are
     * scheduled through timestamps with a lead), underruns are not: a 5 ms
     * writer at a few-ms buffer would underrun on any scheduler hiccup.
     */
    val audioTrackMinimumBufferMilliseconds: Int = 40,
    /**
     * Frames beyond the last written frame a splice must start at. The writer
     * may be mid-block when a chirp is scheduled; one block plus slack.
     */
    val schedulingSafetyMarginMilliseconds: Long = 20,
    /** AudioTrack.getTimestamp refresh: keeps the scheduler anchor fresh. */
    val playbackTimestampPollIntervalMilliseconds: Long = 250,
    /** One playback anchor in this many polls goes to events.jsonl (~1 Hz). */
    val playbackTimestampLogEveryNthPoll: Int = 4,
    /** Duplex start waits this long for the first valid AudioTrack timestamp. */
    val firstPlaybackTimestampTimeoutMilliseconds: Long = 3_000,
    val writerThreadJoinTimeoutMilliseconds: Long = 1_000,
)

data class FieldNodeConfiguration(
    /** Plan: about 30 s of PCM kept for clip requests. */
    val rollingBufferSeconds: Int = 30,
)

/** Defaults for a chirp lab run when the launch intent omits an extra. */
data class FieldChirpLabConfiguration(
    val defaultPeriodMilliseconds: Int = 4_000,
    val defaultLeadMilliseconds: Int = 300,
    val defaultBandLowHz: Double = 13_000.0,
    val defaultBandHighHz: Double = 16_000.0,
    val defaultChirpDurationSeconds: Double = 0.05,
    val defaultLevelDbfs: Double = -6.0,
    /** Self-chirp search window either side of the predicted capture frame. */
    val selfChirpSearchMarginMilliseconds: Int = 150,
    /** Capture history kept for self-chirp windows; must exceed lead + margin. */
    val selfChirpCaptureRingSeconds: Int = 3,
    /** WAV writer queue in capture blocks (~10 s at 20 ms) before overflow. */
    val wavWriterQueueCapacityBlocks: Int = 512,
    /** Patch the WAV header this often so a killed run still has a near-true size. */
    val wavHeaderRefreshIntervalBlocks: Int = 250,
    val eventsWriterQueueCapacity: Int = 4_096,
    val attitudeSampleIntervalMilliseconds: Long = 1_000,
    val thermalPollIntervalMilliseconds: Long = 10_000,
    /** Instrumented loopback acceptance: weakest-part self SNR. */
    val minimumAcceptableSelfSnrDb: Double = 15.0,
    /** A self chirp at or above this peak fraction counts as clipped. */
    val clippingPeakFraction: Float = 0.98f,
)

/** LAN session shell constants (transport, discovery, cadence). */
data class FieldSessionConfiguration(
    /** Device tick cadence: emission scheduling and round reporting latency. */
    val deviceTickIntervalMilliseconds: Long = 20,
    /** UI / dumpsys refresh of FieldSyncStatus. */
    val statusRefreshIntervalMilliseconds: Long = 250,
    val nsdServiceType: String = "_minimappr-field._tcp",
    val nsdTxtProtocolVersion: String = "1",
    /** Fixed so a typed IP:port join is easy; 0 would pick an ephemeral port. */
    val hostListenPort: Int = 47_480,
    val webSocketPath: String = "/minimappr-field",
    val webSocketPingIntervalMilliseconds: Long = 5_000,
    /** Generous: Wi-Fi power save stalls links for hundreds of ms. */
    val webSocketTimeoutMilliseconds: Long = 15_000,
    val joinerReconnectDelayMilliseconds: Long = 1_000,
    /** Per-connection outbound queue; full = drop and count (never block the tick). */
    val outboundQueueCapacityEnvelopes: Int = 256,
    val defaultAirTemperatureC: Double = 20.0,
    /** A join with no typed address waits this long for an NSD host. */
    val nsdAutoJoinTimeoutMilliseconds: Long = 30_000,
)

data class FieldAppConfiguration(
    val capture: FieldCaptureConfiguration = FieldCaptureConfiguration(),
    val playback: FieldPlaybackConfiguration = FieldPlaybackConfiguration(),
    val node: FieldNodeConfiguration = FieldNodeConfiguration(),
    val chirpLab: FieldChirpLabConfiguration = FieldChirpLabConfiguration(),
    val session: FieldSessionConfiguration = FieldSessionConfiguration(),
)
