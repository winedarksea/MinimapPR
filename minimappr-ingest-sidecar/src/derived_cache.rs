use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tokio::fs;
#[cfg(test)]
use uuid::Uuid;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;
type BoxedResult<T> = Result<T, BoxedError>;

#[derive(Clone, Debug)]
pub struct DerivedCacheConfig {
    pub budget_bytes: u64,
    pub admission_reserve_bytes: u64,
}

impl Default for DerivedCacheConfig {
    fn default() -> Self {
        Self {
            budget_bytes: 67_108_864,
            admission_reserve_bytes: 8_388_608,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DerivedCacheEntry {
    pub derived_id: String,
    pub artifact_type: String,
    pub path: PathBuf,
    pub byte_length: u64,
    pub created_ns: u128,
    pub last_access_ns: u128,
    pub source_journal_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DerivedCacheIndex {
    pub entries: Vec<DerivedCacheEntry>,
}

#[derive(Clone, Debug)]
pub struct DerivedCache {
    root: PathBuf,
    index_path: PathBuf,
    config: DerivedCacheConfig,
}

impl DerivedCache {
    pub fn new(journal_root: &Path, config: DerivedCacheConfig) -> Self {
        let root = journal_root.join("derived-cache");
        Self {
            index_path: root.join("index.json"),
            root,
            config,
        }
    }

    pub async fn ensure_initialized(&self) -> BoxedResult<()> {
        fs::create_dir_all(&self.root).await?;
        if !self.index_path.exists() {
            self.write_index(&DerivedCacheIndex::default()).await?;
        }
        Ok(())
    }

    pub async fn reserve_for_write(&self, append_length_bytes: u64) -> BoxedResult<()> {
        let (index, changed) = self.evict_index_for_append(append_length_bytes).await?;
        if changed {
            self.write_index(&index).await?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub async fn record_entry(
        &self,
        artifact_type: String,
        payload: &[u8],
        source_journal_ids: Vec<String>,
        now_ns: u128,
    ) -> BoxedResult<DerivedCacheEntry> {
        let append_length_bytes = u64::try_from(payload.len())?;
        let (mut index, _) = self.evict_index_for_append(append_length_bytes).await?;
        fs::create_dir_all(&self.root).await?;
        let derived_id = format!("derived-{}", Uuid::new_v4());
        let path = self.root.join(format!("{derived_id}.bin"));
        fs::write(&path, payload).await?;
        let entry = DerivedCacheEntry {
            derived_id,
            artifact_type,
            path,
            byte_length: u64::try_from(payload.len())?,
            created_ns: now_ns,
            last_access_ns: now_ns,
            source_journal_ids,
        };
        index.entries.push(entry.clone());
        self.write_index(&index).await?;
        Ok(entry)
    }

    async fn evict_index_for_append(
        &self,
        append_length_bytes: u64,
    ) -> BoxedResult<(DerivedCacheIndex, bool)> {
        if self.config.budget_bytes == 0 {
            return Ok((self.read_index().await?, false));
        }

        let mut index = self.read_index().await?;
        let mut changed = false;
        while total_bytes(&index)
            .saturating_add(append_length_bytes)
            .saturating_add(self.config.admission_reserve_bytes)
            > self.config.budget_bytes
        {
            let Some(oldest) = index
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_access_ns)
                .map(|(index, _)| index)
            else {
                return Err("derived cache budget exhausted and no entries are evictable".into());
            };
            let evicted = index.entries.remove(oldest);
            let _ = fs::remove_file(evicted.path).await;
            changed = true;
        }

        Ok((index, changed))
    }

    /// Update `last_access_ns` for an existing entry so LRU eviction stays accurate.
    #[allow(dead_code)]
    pub async fn touch(&self, derived_id: &str, now_ns: u128) -> BoxedResult<bool> {
        let mut index = self.read_index().await?;
        let Some(entry) = index
            .entries
            .iter_mut()
            .find(|e| e.derived_id == derived_id)
        else {
            return Ok(false);
        };
        entry.last_access_ns = now_ns;
        self.write_index(&index).await?;
        Ok(true)
    }

    async fn read_index(&self) -> BoxedResult<DerivedCacheIndex> {
        match fs::read_to_string(&self.index_path).await {
            Ok(contents) => Ok(serde_json::from_str(&contents)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(DerivedCacheIndex::default())
            }
            Err(error) => Err(Box::new(error)),
        }
    }

    async fn write_index(&self, index: &DerivedCacheIndex) -> BoxedResult<()> {
        fs::create_dir_all(&self.root).await?;
        let tmp_path = self.index_path.with_file_name(".index.json.tmp");
        fs::write(&tmp_path, serde_json::to_vec(index)?).await?;
        fs::rename(&tmp_path, &self.index_path).await?;
        Ok(())
    }
}

fn total_bytes(index: &DerivedCacheIndex) -> u64 {
    index.entries.iter().fold(0_u64, |total, entry| {
        total.saturating_add(entry.byte_length)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn derived_cache_evicts_oldest_entry_to_preserve_budget() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = DerivedCache::new(
            tmp.path(),
            DerivedCacheConfig {
                budget_bytes: 6,
                admission_reserve_bytes: 0,
            },
        );
        cache.ensure_initialized().await.unwrap();
        let first = cache
            .record_entry(
                "classifier_render".to_string(),
                b"123",
                vec!["jrnl-1".to_string()],
                1,
            )
            .await
            .unwrap();
        let second = cache
            .record_entry(
                "classifier_render".to_string(),
                b"456",
                vec!["jrnl-2".to_string()],
                2,
            )
            .await
            .unwrap();
        let third = cache
            .record_entry(
                "classifier_render".to_string(),
                b"789",
                vec!["jrnl-3".to_string()],
                3,
            )
            .await
            .unwrap();

        assert!(!first.path.exists());
        assert!(second.path.exists());
        assert!(third.path.exists());
    }
}
