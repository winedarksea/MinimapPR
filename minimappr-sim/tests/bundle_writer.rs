//! Calibration-bundle zip layout and JSON contents (the Python loader side
//! is covered by tests/test_sim_bundle_roundtrip.py at the repo root).

use std::io::Read;
use std::path::Path;

use minimappr_sim::bundle::geo::{FlatLocalFrame, GeoPoint};
use minimappr_sim::bundle::manifest::iso8601_utc_from_epoch_ns;
use minimappr_sim::bundle::BundleCompression;
use minimappr_sim::{render_bundle, Scenario};
use serde_json::Value;

fn read_entry(archive: &mut zip::ZipArchive<std::fs::File>, name: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    archive
        .by_name(name)
        .unwrap_or_else(|_| panic!("missing {name}"))
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}

#[test]
fn roundtrip_scenario_bundle_has_v1_layout_and_extended_ground_truth() {
    let scenario = Scenario::from_path(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("scenarios/roundtrip_small.toml"),
    )
    .unwrap();
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("roundtrip_small_bundle.zip");
    render_bundle(&scenario, &out, BundleCompression::Deflated).unwrap();
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&out).unwrap()).unwrap();

    let manifest: Value =
        serde_json::from_slice(&read_entry(&mut archive, "manifest.json")).unwrap();
    assert_eq!(manifest["kind"], "minimappr_calibration_bundle");
    assert_eq!(manifest["schema_version"], 1);
    let nodes = manifest["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 3);
    for node in nodes {
        let channel_count = node["channel_sensor_ids"].as_array().unwrap().len();
        assert_eq!(
            node["sensor_offsets_m"].as_array().unwrap().len(),
            channel_count
        );
        let wav = read_entry(&mut archive, node["audio_file"].as_str().unwrap());
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(
            u16::from_le_bytes([wav[22], wav[23]]) as usize,
            channel_count
        );
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            scenario.sample_rate_hz
        );
        let expected_bytes =
            (scenario.duration_s * scenario.sample_rate()) as usize * channel_count * 2;
        assert_eq!(wav.len() - 44, expected_bytes);
    }
    assert_eq!(
        nodes[0]["sensor_offsets_m"].as_array().unwrap().len(),
        4,
        "tetra node"
    );
    assert_eq!(
        nodes[2]["sensor_offsets_m"].as_array().unwrap().len(),
        1,
        "phone node"
    );

    let ground_truth: Value =
        serde_json::from_slice(&read_entry(&mut archive, "ground_truth.json")).unwrap();
    let events = ground_truth["events"].as_array().unwrap();
    assert_eq!(events.len(), 2);
    for event in events {
        for key in [
            "event_id",
            "label",
            "label_category",
            "source",
            "start_ns",
            "end_ns",
            "individual_id",
            "species",
            "is_echo_source",
        ] {
            assert!(!event[key].is_null(), "event missing {key}");
        }
        assert_eq!(event["geometry"]["type"], "static");
        // position_geo must invert back to position_m in the flat frame.
        let frame = FlatLocalFrame::new(GeoPoint {
            lat: scenario.site.origin_lat,
            lon: scenario.site.origin_lon,
            alt_m: scenario.site.origin_alt_m,
        });
        let geo = &event["geometry"]["position_geo"];
        let local = frame.geo_to_local(GeoPoint {
            lat: geo["lat"].as_f64().unwrap(),
            lon: geo["lon"].as_f64().unwrap(),
            alt_m: geo["alt_m"].as_f64().unwrap(),
        });
        let position: Vec<f64> = event["position_m"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        for axis in 0..3 {
            assert!((local[axis] - position[axis]).abs() < 1e-6);
        }
    }
    assert!(!ground_truth["reflectors"].as_array().unwrap().is_empty());
    assert_eq!(
        ground_truth["simulator"]["devices"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    let expectations: Value =
        serde_json::from_slice(&read_entry(&mut archive, "expectations.json")).unwrap();
    assert_eq!(expectations["runtime"]["classifier_backend"], "stub");
    read_entry(&mut archive, "detections.json");
}

#[test]
fn iso_timestamp_matches_known_epoch() {
    assert_eq!(iso8601_utc_from_epoch_ns(0), "1970-01-01T00:00:00Z");
    assert_eq!(
        iso8601_utc_from_epoch_ns(1_763_700_000_000_000_000),
        "2025-11-21T04:40:00Z"
    );
}
