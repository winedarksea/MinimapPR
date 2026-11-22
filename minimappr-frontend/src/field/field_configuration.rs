//! Operational constants of the Field PWA (AGENTS.md: no magic numbers).

/// Device clock the core's Stage-1 and clip codec are designed for.
pub const REQUESTED_CAPTURE_SAMPLE_RATE_HZ: u32 = 48_000;
/// Clip codec decimates exactly 2:1 from this rate; other rates skip clips.
pub const CLIP_CODEC_REQUIRED_SAMPLE_RATE_HZ: u32 = 48_000;
/// Worklet chunk: 20 ms at 48 kHz, one Stage-1 hop per message.
pub const WORKLET_CHUNK_SAMPLE_COUNT: u32 = 960;

/// Rolling capture buffer feeding classification windows and clips.
pub const ROLLING_BUFFER_SECONDS: f64 = 12.0;
/// Audio kept before the Stage-1 trigger so the call's start is inside.
pub const CLASSIFICATION_PRE_ROLL_SECONDS: f64 = 0.5;
/// BirdNET V3 window. 3 s scores best for short calls (birdnet_v3.md
/// window-dilution table: 0.92 at 3 s vs 0.83 at 6 s).
pub const CLASSIFICATION_WINDOW_SECONDS: f64 = 3.0;
/// While a trigger stays open, a new window starts this long after the last.
pub const CLASSIFICATION_WINDOW_HOP_SECONDS: f64 = 3.0;
/// Windows waiting for the (single) classifier; the oldest is dropped beyond.
pub const MAXIMUM_QUEUED_CLASSIFICATION_WINDOWS: usize = 2;
/// Clip cut around each classified window (plan: shared 1.0-1.5 s window).
pub const CLIP_WINDOW_SECONDS: f64 = 1.5;

/// BirdNET V3 input rate (Hz).
pub const BIRDNET_INPUT_SAMPLE_RATE_HZ: u32 = 32_000;
/// Classes reported per window.
pub const BIRDNET_TOP_CLASS_COUNT: u32 = 5;
/// Detections at or above this top score are pushed to the feed and relay.
/// Untuned for V3 (W3-birdnet-cutover re-tunes thresholds).
pub const BIRDNET_REPORT_MINIMUM_SCORE: f32 = 0.25;
/// Wire id of the classifier in `DetectionEvent::classifier_model_id`.
pub const BIRDNET_V3_CLASSIFIER_MODEL_ID: u16 = 3;

/// Windowed-sinc resampler: taps on each side of the interpolation point.
pub const RESAMPLER_HALF_WIDTH_TAPS: usize = 16;
/// Precomputed fractional phases (timing error <= 1 / (2 * phases) sample).
pub const RESAMPLER_PHASE_COUNT: usize = 256;
/// Cutoff as a fraction of the output Nyquist (transition band headroom).
pub const RESAMPLER_CUTOFF_FRACTION_OF_NYQUIST: f64 = 0.92;
/// Kaiser window shape for the resampler kernel (~80 dB sidelobes).
pub const RESAMPLER_KAISER_BETA: f64 = 8.0;

/// Asset URLs (same origin; the dev server mounts the gitignored model dir).
pub const WORKLET_MODULE_URL: &str = "/field-mic-worklet.js";
pub const ORT_WASM_BASE_URL: &str = "/";
pub const BIRDNET_MODEL_URL: &str = "/models/birdnet_v3/BirdNET+_V3.0_Global_12K_FP16_pruned.onnx";
pub const BIRDNET_LABELS_URL: &str = "/models/birdnet_v3/BirdNET+_V3.0_Global_12K_Labels.csv";
pub const SERVICE_WORKER_URL: &str = "/field-sw.js";
/// Relay origin; empty = the page's own origin (production: one Worker
/// serves PWA + relay; dev: field-dev-server proxies /v1/).
pub const RELAY_HTTP_BASE_URL: &str = "";

/// Relay driver timer (reconnect checks and gated flushes).
pub const RELAY_TICK_INTERVAL_MILLISECONDS: u32 = 100;
/// Peer id the host binds (joiners bind provisional ids >= 0x8000_0000).
pub const HOST_RELAY_PEER_ID: u32 = 1;

/// Entries kept in the on-screen relay message log.
pub const RELAY_MESSAGE_LOG_CAPACITY: usize = 30;
/// Local detections kept in the shared feed.
pub const LOCAL_DETECTION_FEED_CAPACITY: usize = 50;
