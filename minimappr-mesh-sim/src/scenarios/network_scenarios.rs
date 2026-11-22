use super::{report, ScenarioReport};

pub fn network_faults(seed: u64) -> Result<ScenarioReport, String> {
    report("network_faults", seed, Ok("todo".into()))
}

pub fn offline_host_loss(seed: u64) -> Result<ScenarioReport, String> {
    report("offline_host_loss", seed, Ok("todo".into()))
}

pub fn transport_fallback(seed: u64) -> Result<ScenarioReport, String> {
    report("transport_fallback", seed, Ok("todo".into()))
}

pub fn constrained_link(seed: u64) -> Result<ScenarioReport, String> {
    report("constrained_link", seed, Ok("todo".into()))
}
