use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JournalPayloadHandle {
    pub journal_epoch: u64,
    pub segment_id: String,
    pub stream_key: String,
    pub payload_offset_bytes: u64,
    pub payload_length_bytes: u64,
    pub toa_ns: Option<u64>,
    pub tor_ns: Option<u64>,
    pub sample_index_start: Option<u64>,
    pub sample_count: Option<u64>,
    pub integrity_hash: String,
    pub segment_path: PathBuf,
}

pub fn stable_segment_path(journal_root: &Path, stream_key: &str, segment_id: &str) -> PathBuf {
    journal_root
        .join("streams")
        .join(stream_key)
        .join("segments")
        .join(format!("{segment_id}.bin"))
}
