/// StreamRangeLease: pins all journal segments whose toa_ns falls within
/// [start_ns, end_ns] for a given stream_key.
///
/// Key invariant: `end_ns` is set at creation time as
/// `min(requested_end_ns, start_ns + MAX_DURATION_NS)` and is IMMUTABLE.
/// Heartbeat calls are liveness checks only; they update `heartbeat_deadline_ns`
/// but can never extend `end_ns`.  Missing a heartbeat causes immediate GC of
/// the lease on the next sweep, freeing tmpfs memory.
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use tokio::fs;
use uuid::Uuid;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;
type BoxedResult<T> = Result<T, BoxedError>;

/// Maximum recording duration the sidecar will honour, regardless of what
/// Python requests.  This is the sole OOM defence for the tmpfs ring.
pub const MAX_DURATION_NS: u64 = 5 * 60 * 1_000_000_000; // 5 minutes

/// How long a lease survives without a heartbeat before it is GC'd.
/// 120s floor: worst-case pipeline lag with a full 128-manifest backlog and
/// 1.5s BirdNET calls is ~192s; 120s matches the BirdNET subprocess timeout.
pub const HEARTBEAT_TTL_NS: u128 = 120 * 1_000_000_000; // 120 seconds

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamRangeLeaseRequest {
    pub owner: String,
    pub stream_key: String,
    pub start_ns: u64,
    /// Python's desired end; sidecar caps it at `start_ns + MAX_DURATION_NS`.
    pub requested_end_ns: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamRangeLeaseRecord {
    pub lease_id: String,
    pub owner: String,
    pub stream_key: String,
    pub start_ns: u64,
    /// Immutable.  Set to `min(requested_end_ns, start_ns + MAX_DURATION_NS)`.
    pub end_ns: u64,
    /// If `now_ns > heartbeat_deadline_ns` the lease is considered dead and
    /// will be GC'd on the next active_index() sweep.
    pub heartbeat_deadline_ns: u128,
}

#[derive(Clone, Debug, Serialize)]
pub struct StreamRangeLeaseResponse {
    pub lease_id: String,
    pub start_ns: u64,
    /// The effective (capped) end timestamp.
    pub end_ns: u64,
    /// When the first heartbeat must arrive by.
    pub heartbeat_deadline_ns: u128,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StreamRangeLeaseIndex {
    pub leases: HashMap<String, StreamRangeLeaseRecord>,
}

/// Persistent store for StreamRangeLease records, backed by a single JSON
/// file with atomic rename-on-write.
#[derive(Clone, Debug)]
pub struct StreamRangeLeaseStore {
    path: PathBuf,
}

impl StreamRangeLeaseStore {
    pub fn new(journal_root: &Path) -> Self {
        Self {
            path: journal_root.join("stream_range_leases.json"),
        }
    }

    pub async fn create(
        &self,
        request: StreamRangeLeaseRequest,
        now_ns: u128,
    ) -> BoxedResult<(StreamRangeLeaseResponse, StreamRangeLeaseIndex)> {
        let mut index = self.load().await?;
        self.purge_dead_leases(&mut index, now_ns);

        let capped_end = request
            .start_ns
            .saturating_add(MAX_DURATION_NS)
            .min(request.requested_end_ns);

        let lease_id = format!("srl-{}", Uuid::new_v4());
        let heartbeat_deadline_ns = now_ns.saturating_add(HEARTBEAT_TTL_NS);

        let record = StreamRangeLeaseRecord {
            lease_id: lease_id.clone(),
            owner: request.owner,
            stream_key: request.stream_key,
            start_ns: request.start_ns,
            end_ns: capped_end,
            heartbeat_deadline_ns,
        };
        index.leases.insert(lease_id.clone(), record);
        self.save(&index).await?;

        Ok((
            StreamRangeLeaseResponse {
                lease_id,
                start_ns: request.start_ns,
                end_ns: capped_end,
                heartbeat_deadline_ns,
            },
            index,
        ))
    }

    /// Record a heartbeat for an existing lease.  Refreshes
    /// `heartbeat_deadline_ns` but NEVER changes `end_ns`.
    pub async fn heartbeat(
        &self,
        lease_id: &str,
        now_ns: u128,
    ) -> BoxedResult<StreamRangeLeaseResponse> {
        let mut index = self.load().await?;
        self.purge_dead_leases(&mut index, now_ns);

        let record = index
            .leases
            .get_mut(lease_id)
            .ok_or_else(|| format!("stream range lease {lease_id} not found or expired"))?;

        record.heartbeat_deadline_ns = now_ns.saturating_add(HEARTBEAT_TTL_NS);
        let response = StreamRangeLeaseResponse {
            lease_id: record.lease_id.clone(),
            start_ns: record.start_ns,
            end_ns: record.end_ns,
            heartbeat_deadline_ns: record.heartbeat_deadline_ns,
        };
        self.save(&index).await?;
        Ok(response)
    }

    pub async fn release(&self, lease_id: &str, now_ns: u128) -> BoxedResult<()> {
        let mut index = self.load().await?;
        self.purge_dead_leases(&mut index, now_ns);
        index.leases.remove(lease_id);
        self.save(&index).await?;
        Ok(())
    }

    pub async fn active_index(&self, now_ns: u128) -> BoxedResult<StreamRangeLeaseIndex> {
        let mut index = self.load().await?;
        if self.purge_dead_leases(&mut index, now_ns) {
            self.save(&index).await?;
        }
        Ok(index)
    }

    /// Returns true if the segment with `[seg_start_toa, seg_end_toa]` for
    /// `stream_key` is covered by at least one active range lease.
    pub fn segment_is_pinned(
        index: &StreamRangeLeaseIndex,
        stream_key: &str,
        seg_start_toa: Option<u64>,
        seg_end_toa: Option<u64>,
        now_ns: u128,
    ) -> bool {
        let seg_start = match seg_start_toa {
            Some(v) => v,
            None => return false,
        };
        let seg_end = seg_end_toa.unwrap_or(seg_start);

        index
            .leases
            .values()
            .filter(|lease| lease.stream_key == stream_key && lease.heartbeat_deadline_ns > now_ns)
            .any(|lease| {
                // Ranges overlap when neither is entirely before the other.
                seg_start <= lease.end_ns && seg_end >= lease.start_ns
            })
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn purge_dead_leases(&self, index: &mut StreamRangeLeaseIndex, now_ns: u128) -> bool {
        let before = index.leases.len();
        index
            .leases
            .retain(|_, lease| lease.heartbeat_deadline_ns > now_ns);
        before != index.leases.len()
    }

    async fn load(&self) -> BoxedResult<StreamRangeLeaseIndex> {
        match fs::read_to_string(&self.path).await {
            Ok(contents) => Ok(serde_json::from_str(&contents)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(StreamRangeLeaseIndex::default())
            }
            Err(error) => Err(Box::new(error)),
        }
    }

    async fn save(&self, index: &StreamRangeLeaseIndex) -> BoxedResult<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).await?;
        }
        let tmp = self.path.with_file_name(".stream_range_leases.json.tmp");
        fs::write(&tmp, serde_json::to_vec(index)?).await?;
        fs::rename(&tmp, &self.path).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn create_caps_end_ns() {
        let tmp = tempfile::tempdir().unwrap();
        let store = StreamRangeLeaseStore::new(tmp.path());
        let now_ns: u128 = 1_000_000_000_000;
        let start_ns: u64 = 1_000_000_000;
        let excessive_end: u64 = start_ns + MAX_DURATION_NS * 3;

        let (resp, _) = store
            .create(
                StreamRangeLeaseRequest {
                    owner: "test".into(),
                    stream_key: "s1".into(),
                    start_ns,
                    requested_end_ns: excessive_end,
                },
                now_ns,
            )
            .await
            .unwrap();

        assert_eq!(resp.end_ns, start_ns + MAX_DURATION_NS);
    }

    #[tokio::test]
    async fn heartbeat_does_not_extend_end_ns() {
        let tmp = tempfile::tempdir().unwrap();
        let store = StreamRangeLeaseStore::new(tmp.path());
        let now_ns: u128 = 1_000_000_000_000;
        let start_ns: u64 = 1_000_000_000;
        let end_ns: u64 = start_ns + 60_000_000_000; // 1 minute

        let (initial, _) = store
            .create(
                StreamRangeLeaseRequest {
                    owner: "test".into(),
                    stream_key: "s1".into(),
                    start_ns,
                    requested_end_ns: end_ns,
                },
                now_ns,
            )
            .await
            .unwrap();

        let after_hb = store
            .heartbeat(&initial.lease_id, now_ns + 10_000_000_000)
            .await
            .unwrap();

        assert_eq!(after_hb.end_ns, initial.end_ns, "end_ns must not change");
        assert!(
            after_hb.heartbeat_deadline_ns > initial.heartbeat_deadline_ns,
            "heartbeat_deadline_ns should advance"
        );
    }

    #[tokio::test]
    async fn expired_lease_not_pinned() {
        let tmp = tempfile::tempdir().unwrap();
        let store = StreamRangeLeaseStore::new(tmp.path());
        let now_ns: u128 = 1_000_000_000_000;

        let (resp, _) = store
            .create(
                StreamRangeLeaseRequest {
                    owner: "test".into(),
                    stream_key: "s1".into(),
                    start_ns: 0,
                    requested_end_ns: MAX_DURATION_NS,
                },
                now_ns,
            )
            .await
            .unwrap();

        // Advance well past the heartbeat TTL.
        let future_ns = now_ns + HEARTBEAT_TTL_NS * 2;
        let index = store.active_index(future_ns).await.unwrap();

        assert!(
            !StreamRangeLeaseStore::segment_is_pinned(&index, "s1", Some(0), Some(100), future_ns),
            "lease {}: expired leases must not pin segments",
            resp.lease_id
        );
    }

    #[tokio::test]
    async fn segment_overlap_detection() {
        let tmp = tempfile::tempdir().unwrap();
        let store = StreamRangeLeaseStore::new(tmp.path());
        let now_ns: u128 = 1_000_000_000_000;

        store
            .create(
                StreamRangeLeaseRequest {
                    owner: "test".into(),
                    stream_key: "audio".into(),
                    start_ns: 1_000,
                    requested_end_ns: 9_000,
                },
                now_ns,
            )
            .await
            .unwrap();

        let index = store.active_index(now_ns).await.unwrap();

        assert!(StreamRangeLeaseStore::segment_is_pinned(
            &index,
            "audio",
            Some(500),
            Some(2_000),
            now_ns
        ));
        assert!(StreamRangeLeaseStore::segment_is_pinned(
            &index,
            "audio",
            Some(4_000),
            Some(6_000),
            now_ns
        ));
        assert!(StreamRangeLeaseStore::segment_is_pinned(
            &index,
            "audio",
            Some(8_500),
            Some(10_000),
            now_ns
        ));
        assert!(!StreamRangeLeaseStore::segment_is_pinned(
            &index,
            "audio",
            Some(10_000),
            Some(20_000),
            now_ns
        ));
        assert!(!StreamRangeLeaseStore::segment_is_pinned(
            &index,
            "other_stream",
            Some(4_000),
            Some(6_000),
            now_ns
        ));
    }
}
