//! MinimapPR Field Mode Layer 1 acoustic world simulator.
//!
//! Renders sample-accurate per-device audio in virtual time — exact
//! fractional delay with per-receiver clock offset/drift, moving paths
//! (Doppler), 1/r + ISO 9613-1 absorption, ground image + point-reflector
//! multipath, mic responses, phone self-recording (d_ii, limiter,
//! body-conduction precursor) and chaos knobs — and writes MinimapPR
//! calibration bundles consumable by `minimappr/calibration/bundle.py`.
//!
//! Pipeline: `Scenario` (TOML) → `world::plan_world` (all random draws) →
//! `render::render_world` → `bundle::write_bundle`.

pub mod bundle;
pub mod physics;
pub mod render;
pub mod scenario;
pub mod signals;
pub mod world;

use std::path::Path;
use std::time::{Duration, Instant};

pub use render::RenderedDevice;
pub use scenario::Scenario;
pub use world::WorldPlan;

/// Wall-clock cost of the planning and rendering stages.
#[derive(Clone, Copy, Debug)]
pub struct RenderTiming {
    pub planning: Duration,
    pub rendering: Duration,
}

/// Plan and render a scenario.
pub fn render_scenario(
    scenario: &Scenario,
) -> Result<(WorldPlan, Vec<RenderedDevice>, RenderTiming), String> {
    let planning_start = Instant::now();
    let plan = world::plan_world(scenario)?;
    let planning = planning_start.elapsed();
    let rendering_start = Instant::now();
    let rendered = render::render_world(&plan);
    let rendering = rendering_start.elapsed();
    Ok((
        plan,
        rendered,
        RenderTiming {
            planning,
            rendering,
        },
    ))
}

/// Plan, render and write a calibration bundle zip.
pub fn render_bundle(
    scenario: &Scenario,
    out_path: &Path,
    compression: bundle::BundleCompression,
) -> Result<RenderTiming, String> {
    let (plan, rendered, timing) = render_scenario(scenario)?;
    bundle::write_bundle(out_path, scenario, &plan, &rendered, compression)?;
    Ok(timing)
}
