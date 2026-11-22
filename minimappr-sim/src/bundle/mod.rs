//! MinimapPR calibration-bundle zip writer (layout documented in
//! `minimappr/calibration/bundle.py`):
//!
//! ```text
//! manifest.json  ground_truth.json  detections.json  [expectations.json]
//! audio/{node_id}.wav               reference_audio/README.txt
//! ```

pub mod geo;
pub mod ground_truth;
pub mod manifest;
pub mod wav;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde_json::{json, Value};
use zip::write::SimpleFileOptions;
use zip::CompressionMethod;

use crate::render::RenderedDevice;
use crate::scenario::Scenario;
use crate::world::WorldPlan;
use geo::{FlatLocalFrame, GeoPoint};

pub const BUNDLE_KIND: &str = "minimappr_calibration_bundle";
pub const BUNDLE_SCHEMA_VERSION: u32 = 1;
pub const SIMULATOR_NAME: &str = "minimappr-sim";

const REFERENCE_AUDIO_README: &str =
    "Simulated bundle: no reference-node audio. Ground truth comes from the\n\
simulator (ground_truth.json → simulator.*) rather than a source-placed node.\n";

/// Fastest deflate level: noisy PCM barely compresses better at higher
/// levels, while level 6 is ~10x slower to write.
const DEFLATE_LEVEL: i64 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BundleCompression {
    /// Deflate (smaller; slower to write).
    #[default]
    Deflated,
    /// Stored (fast to write and read; large).
    Stored,
}

pub fn site_frame(scenario: &Scenario) -> FlatLocalFrame {
    FlatLocalFrame::new(GeoPoint {
        lat: scenario.site.origin_lat,
        lon: scenario.site.origin_lon,
        alt_m: scenario.site.origin_alt_m,
    })
}

fn expectations_json(scenario: &Scenario) -> Result<Option<Value>, String> {
    scenario
        .expectations
        .as_ref()
        .map(|toml_value| {
            serde_json::to_value(toml_value).map_err(|e| format!("expectations → JSON: {e}"))
        })
        .transpose()
}

/// Write the bundle zip for an already-rendered world.
pub fn write_bundle(
    out_path: &Path,
    scenario: &Scenario,
    plan: &WorldPlan,
    rendered: &[RenderedDevice],
    compression: BundleCompression,
) -> Result<(), String> {
    if let Some(parent) = out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let frame = site_frame(scenario);
    let file = File::create(out_path).map_err(|e| format!("create {}: {e}", out_path.display()))?;
    let mut zip = zip::ZipWriter::new(BufWriter::new(file));
    let method = match compression {
        BundleCompression::Deflated => CompressionMethod::Deflated,
        BundleCompression::Stored => CompressionMethod::Stored,
    };
    let level = (compression == BundleCompression::Deflated).then_some(DEFLATE_LEVEL);
    let options = SimpleFileOptions::default()
        .compression_method(method)
        .compression_level(level)
        .large_file(true);
    let mut put = |name: &str, bytes: &[u8]| -> Result<(), String> {
        zip.start_file(name, options)
            .map_err(|e| format!("zip {name}: {e}"))?;
        zip.write_all(bytes).map_err(|e| format!("zip {name}: {e}"))
    };
    let pretty = |value: &Value| serde_json::to_vec_pretty(value).map_err(|e| e.to_string());

    put(
        "manifest.json",
        &pretty(&manifest::build_manifest(plan, scenario, &frame))?,
    )?;
    put(
        "ground_truth.json",
        &pretty(&ground_truth::build_ground_truth(plan, scenario, &frame))?,
    )?;
    put(
        "detections.json",
        &pretty(&json!({"schema_version": BUNDLE_SCHEMA_VERSION, "detections": []}))?,
    )?;
    if let Some(expectations) = expectations_json(scenario)? {
        put("expectations.json", &pretty(&expectations)?)?;
    }
    for device in rendered {
        let wav_bytes = wav::encode_pcm16_wav(&device.channels, scenario.sample_rate_hz);
        put(&manifest::audio_file_name(&device.device_id), &wav_bytes)?;
    }
    put(
        "reference_audio/README.txt",
        REFERENCE_AUDIO_README.as_bytes(),
    )?;
    zip.finish().map_err(|e| format!("finish zip: {e}"))?;
    Ok(())
}
