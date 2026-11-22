//! Entity registry configuration (the server's `EntityReportingConfig`
//! minus the asyncio-only notification and maintenance timers, which belong
//! to the host shell). Defaults are the server's.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BirdEntityConfig {
    /// Sliding per-entity window: a detection absorbs into an entity last heard this recently.
    pub absorption_window_seconds: f64,
    /// Fixed absorption radius for a fix without a covariance (W3 rule).
    pub absorption_max_fix_distance_m: f64,
    /// A reporting node within this of the bird may absorb an unpositioned bout.
    pub absorption_max_node_distance_m: f64,
    pub absorption_max_bearing_difference_deg: f64,
    /// Pinpointed while the 1σ horizontal major axis stays within this.
    pub pinpointed_max_horizontal_uncertainty_m: f64,
    pub spatial_state_hold_seconds: f64,
    pub expiry_seconds: f64,
    pub bearing_default_half_width_deg: f64,
    pub bearing_min_half_width_deg: f64,
    pub bearing_max_half_width_deg: f64,
    /// k in the k·σ fan half width and range band.
    pub uncertainty_sigma_multiplier: f64,
    pub max_tracked_entities: usize,
    /// A label absorbs only as the entity's top species or with at least this belief.
    pub absorption_min_species_belief: f64,
    /// Horizontal Mahalanobis (2 dof) gate for a fix against a pinpointed entity.
    pub absorption_covariance_gate_enabled: bool,
    pub absorption_chi2_gate: f64,
    /// Jitter floor: a fix this close always absorbs.
    pub absorption_min_fix_distance_m: f64,
    /// Fixes this close in time and inconsistent beyond their covariances are two birds.
    pub concurrent_fix_window_seconds: f64,
    /// A fix too uncertain to pinpoint (or bearing-projected) joins a better-located entity without moving it.
    pub bearing_evidence_absorption_enabled: bool,
    /// Observations carry the bound track's fused position (the canonical row's), not the raw fix.
    pub canonical_row_fused_position_enabled: bool,
}

impl Default for BirdEntityConfig {
    fn default() -> Self {
        Self {
            absorption_window_seconds: 60.0,
            absorption_max_fix_distance_m: 12.0,
            absorption_max_node_distance_m: 50.0,
            absorption_max_bearing_difference_deg: 20.0,
            pinpointed_max_horizontal_uncertainty_m: 15.0,
            spatial_state_hold_seconds: 120.0,
            expiry_seconds: 300.0,
            bearing_default_half_width_deg: 15.0,
            bearing_min_half_width_deg: 5.0,
            bearing_max_half_width_deg: 45.0,
            uncertainty_sigma_multiplier: 2.0,
            max_tracked_entities: 512,
            absorption_min_species_belief: 0.3,
            absorption_covariance_gate_enabled: true,
            absorption_chi2_gate: 9.21,
            absorption_min_fix_distance_m: 3.0,
            concurrent_fix_window_seconds: 2.0,
            bearing_evidence_absorption_enabled: true,
            canonical_row_fused_position_enabled: true,
        }
    }
}
