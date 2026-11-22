//! Developer aid for a failing seed: run one scenario spec and dump host
//! tenures, call records, solve traces (with truth scoring), islands and
//! node link state.
//!
//! cargo run --release -p minimappr-mesh-sim --example inspect_scenario -- <spec> <seed>
//!
//! Specs: host_death, thermal, network_faults, offline_host_loss,
//! transport_relay, transport_cell, mirrored_layout, island_split_merge.

use minimappr_core::mesh::host_transport_policy::SessionTransportPreference;
use minimappr_mesh_sim::scenario::ScenarioSpec;
use minimappr_mesh_sim::scenarios::{host_scenarios, network_scenarios, sync_scenarios};
use minimappr_mesh_sim::ScenarioRun;

fn spec_named(name: &str, seed: u64) -> Option<ScenarioSpec> {
    Some(match name {
        "host_death" => host_scenarios::host_death_spec(seed),
        "thermal" => host_scenarios::thermal_spec(seed, SessionTransportPreference::Auto),
        "network_faults" => network_scenarios::network_faults_spec(seed),
        "offline_host_loss" => network_scenarios::offline_host_loss_spec(seed),
        "transport_relay" => network_scenarios::relay_drop_and_cap_spec(seed),
        "transport_cell" => network_scenarios::host_cell_loss_spec(seed),
        "mirrored_layout" => sync_scenarios::mirrored_layout_spec(seed),
        "island_split_merge" => sync_scenarios::island_split_merge_spec(seed),
        _ => return None,
    })
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let usage = "usage: inspect_scenario <spec> <seed>";
    let name = arguments.next().expect(usage);
    let seed: u64 = arguments.next().expect(usage).parse().expect("seed must be an integer");
    let spec = spec_named(&name, seed).unwrap_or_else(|| panic!("unknown spec {name}"));
    let outcome = ScenarioRun::new(spec, seed).run();
    println!("tenures {:?}", outcome.metrics.host_tenures);
    println!("stalls {:?}", outcome.metrics.stalls_applied);
    for (device, host) in outcome.all_hosts() {
        println!("host {device} term {} counters {:?}", host.host_term, host.calls.counters);
        println!("  rounds completed at {:?}", host.sync.round_completion_ms);
    }
    for record in outcome.call_records() {
        println!(
            "call {} received {} ms clips {:?} walked {:?} chirality {:?} locked {} failure {:?}",
            record.call_id, record.received_ms, record.clip_nodes, record.walked_nodes, record.chirality_confidence, record.chirality_locked, record.failure
        );
    }
    for fix in outcome.scored_fixes() {
        println!(
            "fix call {} nodes {:?} error {:.2} m σ {:.2} m mirrored {:?}",
            fix.call_index, fix.node_ids, fix.horizontal_error_m, fix.reported_major_sigma_m, fix.layout_mirrored
        );
    }
    if let Some((_, host)) = outcome.active_host() {
        for (id, island) in &host.sync.engine.islands {
            let filters: Vec<_> = island.filters.iter().map(|(node, filter)| (node.0, filter.is_valid())).collect();
            println!("island {id} members {:?} reference {:?} filters {filters:?}", island.members, island.reference_node);
        }
    }
    println!("notices {:?}", outcome.metrics.notices);
    for device in &outcome.devices {
        println!("device {} link {:?} counters {:?} outbox {}", device.index, device.node.link, device.node.counters, device.node.outbox.len());
    }
}
