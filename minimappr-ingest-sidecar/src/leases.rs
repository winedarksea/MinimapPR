use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use tokio::fs;
use uuid::Uuid;

use crate::journal_reader::JournalPayloadHandle;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;
type BoxedResult<T> = Result<T, BoxedError>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PinKind {
    Soft,
    Hard,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PinTargetKind {
    RawJournal,
    DerivedCache,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PinLeaseRequest {
    pub owner: String,
    pub pin_kind: PinKind,
    pub target_kind: PinTargetKind,
    pub lease_id: Option<String>,
    pub ttl_seconds: u64,
    pub handle: JournalPayloadHandle,
    pub promotion_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PinLeaseRecord {
    pub lease_id: String,
    pub owner: String,
    pub pin_kind: PinKind,
    pub target_kind: PinTargetKind,
    pub expires_ns: u128,
    pub handle: JournalPayloadHandle,
    pub promotion_id: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PinLeaseResponse {
    pub lease_id: String,
    pub expires_ns: u128,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PinLeaseIndex {
    pub leases: HashMap<String, PinLeaseRecord>,
}

#[derive(Clone, Debug)]
pub struct LeaseStore {
    path: PathBuf,
}

impl LeaseStore {
    pub fn new(journal_root: &Path) -> Self {
        Self {
            path: journal_root.join("pin_leases.json"),
        }
    }

    pub async fn upsert(
        &self,
        request: PinLeaseRequest,
        now_ns: u128,
    ) -> BoxedResult<(PinLeaseResponse, PinLeaseIndex)> {
        let mut index = self.load().await?;
        purge_expired_leases(&mut index, now_ns);
        let lease_id = request
            .lease_id
            .unwrap_or_else(|| format!("lease-{}", Uuid::new_v4()));
        let expires_ns = now_ns.saturating_add(u128::from(request.ttl_seconds) * 1_000_000_000);
        index.leases.insert(
            lease_id.clone(),
            PinLeaseRecord {
                lease_id: lease_id.clone(),
                owner: request.owner,
                pin_kind: request.pin_kind,
                target_kind: request.target_kind,
                expires_ns,
                handle: request.handle,
                promotion_id: request.promotion_id,
            },
        );
        self.save(&index).await?;
        Ok((
            PinLeaseResponse {
                lease_id,
                expires_ns,
            },
            index,
        ))
    }

    pub async fn release(&self, lease_id: &str, now_ns: u128) -> BoxedResult<PinLeaseIndex> {
        let mut index = self.load().await?;
        purge_expired_leases(&mut index, now_ns);
        index.leases.remove(lease_id);
        self.save(&index).await?;
        Ok(index)
    }

    pub async fn active_index(&self, now_ns: u128) -> BoxedResult<PinLeaseIndex> {
        let mut index = self.load().await?;
        if purge_expired_leases(&mut index, now_ns) {
            self.save(&index).await?;
        }
        Ok(index)
    }

    async fn load(&self) -> BoxedResult<PinLeaseIndex> {
        match fs::read_to_string(&self.path).await {
            Ok(contents) => Ok(serde_json::from_str(&contents)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(PinLeaseIndex::default())
            }
            Err(error) => Err(Box::new(error)),
        }
    }

    async fn save(&self, index: &PinLeaseIndex) -> BoxedResult<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).await?;
        }
        let tmp_path = self.path.with_file_name(".pin_leases.json.tmp");
        fs::write(&tmp_path, serde_json::to_vec(index)?).await?;
        fs::rename(&tmp_path, &self.path).await?;
        Ok(())
    }
}

fn purge_expired_leases(index: &mut PinLeaseIndex, now_ns: u128) -> bool {
    let before = index.leases.len();
    index.leases.retain(|_, lease| lease.expires_ns > now_ns);
    before != index.leases.len()
}
