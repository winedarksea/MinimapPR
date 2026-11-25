//! Per-category tracking modes, mirroring the server's
//! `minimappr/core/tracking_mode_config.py` (Settings key
//! `tracking_mode_by_category`): which dynamics a category's tracks run, or
//! none at all.
//!
//! - `ContinuousMotion`: constant-velocity dynamics (drones, vehicles, people).
//! - `StationaryHold`: the perched random-walk filter with the relocation
//!   gate ([`super::perched_tracker`]); the default for wildlife.
//! - `None`: the category is not tracked.
//!
//! A track without a label (category `unknown`) always runs
//! `ContinuousMotion`, so `unknown` may not be configured. Parity:
//! `tests/wildlife_tracking_parity_cases.py` (`tracking_mode` cases).

use std::collections::BTreeMap;
use std::fmt;

/// The mapping entry that applies to every category not listed explicitly.
pub const DEFAULT_CATEGORY_KEY: &str = "default";
/// The category of a track that has no label yet.
pub const UNLABELED_TRACK_CATEGORY: &str = "unknown";
pub const UNLABELED_TRACK_MODE: TrackingMode = TrackingMode::ContinuousMotion;
pub const WILDLIFE_CATEGORY: &str = "wildlife";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackingMode {
    ContinuousMotion,
    StationaryHold,
    None,
}

impl TrackingMode {
    pub const ALL: [TrackingMode; 3] = [
        TrackingMode::ContinuousMotion,
        TrackingMode::StationaryHold,
        TrackingMode::None,
    ];

    /// The Settings spelling (`continuous_motion` | `stationary_hold` | `none`).
    pub fn name(self) -> &'static str {
        match self {
            TrackingMode::ContinuousMotion => "continuous_motion",
            TrackingMode::StationaryHold => "stationary_hold",
            TrackingMode::None => "none",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.name() == name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackingModeConfigError {
    MissingDefaultEntry,
    EmptyCategory,
    UnlabeledCategoryConfigured,
    UnknownMode { category: String, mode: String },
}

impl fmt::Display for TrackingModeConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingDefaultEntry => write!(formatter, "tracking_mode_by_category must include a {DEFAULT_CATEGORY_KEY:?} entry"),
            Self::EmptyCategory => write!(formatter, "tracking_mode_by_category has an empty category"),
            Self::UnlabeledCategoryConfigured => write!(
                formatter,
                "tracking_mode_by_category may not configure {UNLABELED_TRACK_CATEGORY:?}: unlabeled tracks always use {}",
                UNLABELED_TRACK_MODE.name()
            ),
            Self::UnknownMode { category, mode } => write!(formatter, "tracking_mode_by_category[{category:?}] has unknown mode {mode:?}"),
        }
    }
}

impl std::error::Error for TrackingModeConfigError {}

fn normalised_category(category: &str) -> String {
    category.trim().to_lowercase()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackingModeConfig {
    modes_by_category: BTreeMap<String, TrackingMode>,
    default_mode: TrackingMode,
}

impl Default for TrackingModeConfig {
    /// Shipped defaults: wildlife holds station, everything else keeps moving.
    fn default() -> Self {
        Self {
            modes_by_category: BTreeMap::from([(
                WILDLIFE_CATEGORY.to_string(),
                TrackingMode::StationaryHold,
            )]),
            default_mode: TrackingMode::ContinuousMotion,
        }
    }
}

impl TrackingModeConfig {
    /// Build from `(category, mode name)` entries, validated as the server does.
    pub fn from_category_modes<'a>(
        entries: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, TrackingModeConfigError> {
        let mut modes_by_category = BTreeMap::new();
        let mut default_mode = None;
        for (category, mode_name) in entries {
            let mode = TrackingMode::from_name(mode_name).ok_or_else(|| {
                TrackingModeConfigError::UnknownMode {
                    category: category.to_string(),
                    mode: mode_name.to_string(),
                }
            })?;
            if category == DEFAULT_CATEGORY_KEY {
                default_mode = Some(mode);
                continue;
            }
            let normalised = normalised_category(category);
            if normalised.is_empty() {
                return Err(TrackingModeConfigError::EmptyCategory);
            }
            if normalised == UNLABELED_TRACK_CATEGORY {
                return Err(TrackingModeConfigError::UnlabeledCategoryConfigured);
            }
            modes_by_category.insert(normalised, mode);
        }
        let default_mode = default_mode.ok_or(TrackingModeConfigError::MissingDefaultEntry)?;
        Ok(Self {
            modes_by_category,
            default_mode,
        })
    }

    pub fn default_mode(&self) -> TrackingMode {
        self.default_mode
    }

    pub fn mode_for_category(&self, category: &str) -> TrackingMode {
        let normalised = normalised_category(category);
        if normalised.is_empty() || normalised == UNLABELED_TRACK_CATEGORY {
            return UNLABELED_TRACK_MODE;
        }
        self.modes_by_category
            .get(&normalised)
            .copied()
            .unwrap_or(self.default_mode)
    }

    pub fn tracks_category(&self, category: &str) -> bool {
        self.mode_for_category(category) != TrackingMode::None
    }

    pub fn holds_station(&self, category: &str) -> bool {
        self.mode_for_category(category) == TrackingMode::StationaryHold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_defaults_hold_wildlife_and_move_everything_else() {
        let config = TrackingModeConfig::default();
        assert_eq!(
            config.mode_for_category("wildlife"),
            TrackingMode::StationaryHold
        );
        assert_eq!(
            config.mode_for_category(" Wildlife "),
            TrackingMode::StationaryHold
        );
        assert_eq!(
            config.mode_for_category("vehicle"),
            TrackingMode::ContinuousMotion
        );
        assert_eq!(
            config.mode_for_category("unknown"),
            TrackingMode::ContinuousMotion
        );
        assert_eq!(config.mode_for_category(""), TrackingMode::ContinuousMotion);
    }

    #[test]
    fn parsed_defaults_equal_the_built_in_defaults() {
        let parsed = TrackingModeConfig::from_category_modes([
            ("wildlife", "stationary_hold"),
            ("default", "continuous_motion"),
        ])
        .unwrap();
        assert_eq!(parsed, TrackingModeConfig::default());
    }

    #[test]
    fn unlabeled_tracks_move_even_when_the_default_is_none() {
        let config = TrackingModeConfig::from_category_modes([("default", "none")]).unwrap();
        assert!(!config.tracks_category("human"));
        assert!(config.tracks_category("unknown"));
    }

    #[test]
    fn invalid_mappings_are_rejected() {
        assert_eq!(
            TrackingModeConfig::from_category_modes([("wildlife", "stationary_hold")]),
            Err(TrackingModeConfigError::MissingDefaultEntry)
        );
        assert_eq!(
            TrackingModeConfig::from_category_modes([("default", "none"), ("unknown", "none")]),
            Err(TrackingModeConfigError::UnlabeledCategoryConfigured)
        );
        assert!(matches!(
            TrackingModeConfig::from_category_modes([("default", "hover")]),
            Err(TrackingModeConfigError::UnknownMode { .. })
        ));
        assert_eq!(
            TrackingModeConfig::from_category_modes([("default", "none"), ("  ", "none")]),
            Err(TrackingModeConfigError::EmptyCategory)
        );
    }
}
