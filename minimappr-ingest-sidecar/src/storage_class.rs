use std::path::{Path, PathBuf};

#[derive(Clone, Debug, serde::Serialize)]
pub struct StorageClassReport {
    pub path: PathBuf,
    pub storage_class: String,
    pub tmpfs_backed: bool,
    pub enforcement_enabled: bool,
}

#[derive(Debug)]
pub struct TmpfsRequiredError {
    path: PathBuf,
    storage_class: String,
}

impl std::fmt::Display for TmpfsRequiredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "journal root {} is {}, but journal mode requires tmpfs/ramfs",
            self.path.display(),
            self.storage_class,
        )
    }
}

impl std::error::Error for TmpfsRequiredError {}

pub fn validate_journal_storage_class(
    journal_root: &Path,
    enforce_tmpfs: bool,
) -> Result<StorageClassReport, TmpfsRequiredError> {
    let report = detect_storage_class(journal_root, enforce_tmpfs);
    if enforce_tmpfs && !report.tmpfs_backed {
        return Err(TmpfsRequiredError {
            path: report.path,
            storage_class: report.storage_class,
        });
    }
    Ok(report)
}

fn detect_storage_class(path: &Path, enforce_tmpfs: bool) -> StorageClassReport {
    let storage_class = platform_storage_class(path).unwrap_or_else(|| "unknown".to_string());
    let tmpfs_backed = matches!(storage_class.as_str(), "tmpfs" | "ramfs");
    StorageClassReport {
        path: path.to_path_buf(),
        storage_class,
        tmpfs_backed,
        enforcement_enabled: enforce_tmpfs,
    }
}

#[cfg(target_os = "linux")]
fn platform_storage_class(path: &Path) -> Option<String> {
    let canonical_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let mounts = std::fs::read_to_string("/proc/mounts").ok()?;
    let mut best_mount: Option<(usize, String)> = None;
    for line in mounts.lines() {
        let mut fields = line.split_whitespace();
        let _device = fields.next()?;
        let mount_point = unescape_mount_path(fields.next()?);
        let fs_type = fields.next()?.to_string();
        let mount_path = PathBuf::from(mount_point);
        if canonical_path.starts_with(&mount_path) {
            let length = mount_path.as_os_str().len();
            if best_mount
                .as_ref()
                .is_none_or(|(best_length, _)| length > *best_length)
            {
                best_mount = Some((length, fs_type));
            }
        }
    }
    best_mount.map(|(_, fs_type)| fs_type)
}

#[cfg(not(target_os = "linux"))]
fn platform_storage_class(_path: &Path) -> Option<String> {
    Some("development-unknown".to_string())
}

#[cfg(target_os = "linux")]
fn unescape_mount_path(raw: &str) -> String {
    raw.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}
