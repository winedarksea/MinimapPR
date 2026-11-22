//! Developer aid: dump call records and solve traces for one scenario run.
use minimappr_mesh_sim::scenarios::host_scenarios::{host_death_spec, thermal_spec};
use minimappr_mesh_sim::ScenarioRun;
fn main() {
    let name = std::env::args().nth(1).unwrap();
    let seed: u64 = std::env::args().nth(2).unwrap().parse().unwrap();
    let spec = match name.as_str() {
        "host_death" => host_death_spec(seed),
        "thermal" => thermal_spec(seed, minimappr_core::mesh::host_transport_policy::SessionTransportPreference::Auto),
        "offline" => minimappr_mesh_sim::scenarios::network_scenarios::offline_host_loss_spec(seed),
        "faults" => minimappr_mesh_sim::scenarios::network_scenarios::network_faults_spec(seed),
        "cell" => minimappr_mesh_sim::scenarios::network_scenarios::host_cell_loss_spec(seed),
        "mirror" => minimappr_mesh_sim::scenarios::sync_scenarios::mirrored_layout_spec(seed),
        "split" => minimappr_mesh_sim::scenarios::sync_scenarios::island_split_merge_spec(seed),
        "relay" => minimappr_mesh_sim::scenarios::network_scenarios::relay_drop_and_cap_spec(seed),
        _ => panic!(),
    };
    let outcome = ScenarioRun::new(spec, seed).run();
    for (d, h) in outcome.all_hosts() { println!("host {d} term {} counters {:?} rounds {:?}", h.host_term, h.calls.counters, h.sync.round_completion_ms); }
    println!("tenures {:?} stalls {:?}", outcome.metrics.host_tenures, outcome.metrics.stalls_applied);
    for r in outcome.call_records() { println!("call {} recv {} clips {:?} domains {:?} fail {:?}", r.call_id, r.received_ms, r.clip_nodes, r.clip_domains, r.failure); }
    for (t, f) in outcome.traces.iter().zip(outcome.scored_fixes()) { println!("trace call {} true {} nodes {:?} est {:?} fix {:?} err {:.2} sigma {:.2} mirrored {:?}", t.host_call_id, t.true_call_index, t.node_ids, t.estimated_node_positions_m, t.fix.as_ref().map(|f| (f.position_m, f.used_nodes.clone())), f.horizontal_error_m, f.reported_major_sigma_m, f.layout_mirrored); }
    for r in outcome.call_records() { println!("rec {} locked {} conf {:?}", r.call_id, r.chirality_locked, r.chirality_confidence); }
    if let Some((_, h)) = outcome.active_host() {
        println!("final rounds {:?}", h.sync.rounds_started.iter().rev().take(4).collect::<Vec<_>>());
        for (id, island) in &h.sync.engine.islands { println!("island {id} members {:?} ref {:?} filters {:?}", island.members, island.reference_node, island.filters.iter().map(|(n, f)| (n.0, f.is_valid())).collect::<Vec<_>>()); }
        println!("present {:?}", h.membership.present_nodes(outcome.end_s as u64 * 1000, 15000));
    }
    println!("notices {:?}", outcome.metrics.notices);
    for d in &outcome.devices { println!("dev {} link {:?} solo {} joined? counters {:?} outbox {}", d.index, d.node.link, d.node.link.solo, d.node.counters, d.node.outbox.len()); }
    for i in 0..outcome.world.devices.len() { println!("dev {i} true {:?}", outcome.world.position(i, 0.0)); }
}
