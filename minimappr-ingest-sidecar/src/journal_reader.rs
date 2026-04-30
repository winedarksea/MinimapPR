use std::{
    fs::File,
    path::{Path, PathBuf},
};

use memmap2::MmapOptions;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

type BoxedError = Box<dyn std::error::Error + Send + Sync>;
type BoxedResult<T> = Result<T, BoxedError>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JournalPayloadHandle {
    pub journal_epoch: u64,
    pub segment_id: String,
    pub stream_key: String,
    pub payload_offset_bytes: u64,
    pub payload_length_bytes: u64,
    pub sample_index_start: Option<u64>,
    pub sample_count: Option<u64>,
    pub integrity_hash: String,
    pub segment_path: PathBuf,
}

#[derive(Debug)]
pub struct JournalHandleReadError {
    detail: String,
}

impl JournalHandleReadError {
    fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for JournalHandleReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for JournalHandleReadError {}

pub fn read_payload_with_mmap(handle: &JournalPayloadHandle) -> BoxedResult<Vec<u8>> {
    let file = File::open(&handle.segment_path).map_err(|error| {
        JournalHandleReadError::new(format!(
            "journal segment {} could not be reopened: {error}",
            handle.segment_path.display()
        ))
    })?;
    let metadata = file.metadata()?;
    let end = handle
        .payload_offset_bytes
        .checked_add(handle.payload_length_bytes)
        .ok_or_else(|| JournalHandleReadError::new("journal handle byte range overflowed"))?;
    if end > metadata.len() {
        return Err(Box::new(JournalHandleReadError::new(format!(
            "journal handle range {}..{} exceeds segment length {} for {}",
            handle.payload_offset_bytes,
            end,
            metadata.len(),
            handle.segment_path.display(),
        ))));
    }
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let start = usize::try_from(handle.payload_offset_bytes)?;
    let end = usize::try_from(end)?;
    let payload = mmap[start..end].to_vec();
    verify_integrity_hash(&payload, &handle.integrity_hash)?;
    Ok(payload)
}

pub fn verify_integrity_hash(payload: &[u8], integrity_hash: &str) -> BoxedResult<()> {
    if integrity_hash.trim().is_empty() {
        return Ok(());
    }
    let expected = integrity_hash
        .strip_prefix("sha256:")
        .unwrap_or(integrity_hash)
        .to_ascii_lowercase();
    let actual = hex_digest(payload);
    if actual != expected {
        return Err(Box::new(JournalHandleReadError::new(format!(
            "journal payload integrity hash mismatch: expected {expected}, got {actual}"
        ))));
    }
    Ok(())
}

pub fn stable_segment_path(journal_root: &Path, stream_key: &str, segment_id: &str) -> PathBuf {
    journal_root
        .join("streams")
        .join(stream_key)
        .join("segments")
        .join(format!("{segment_id}.bin"))
}

pub fn hex_digest(payload: &[u8]) -> String {
    let digest = Sha256::digest(payload);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
