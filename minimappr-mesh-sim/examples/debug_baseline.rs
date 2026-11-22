//! Developer aid: dump call records and solve traces for one scenario run.
use minimappr_mesh_sim::scenarios::host_scenarios::{host_death_spec, thermal_spec};
use minimappr_mesh_sim::ScenarioRun;
fn main() {
    let name = std::env::args().nth(1).unwrap();
    let seed: u64 = std::env::args().nth(2).unwrap().parse().unwrap();
    let spec = match name.as_str() {
        "host_death" => host_death_spec(seed),
        "thermal" => thermal_spec(seed, minimappr_core::mesh::host_transport_policy::SessionTransportPreference::Auto),
        _ => panic!(),
    };
    let outcome = ScenarioRun::new(spec, seed).run();
    for (d, h) in outcome.all_hosts() { println!("host {d} term {} counters {:?} rounds {:?}", h.host_term, h.calls.counters, h.sync.round_completion_ms); }
    println!("tenures {:?} stalls {:?}", outcome.metrics.host_tenures, outcome.metrics.stalls_applied);
    for r in outcome.call_records() { println!("call {} recv {} clips {:?} domains {:?} fail {:?}", r.call_id, r.received_ms, r.clip_nodes, r.clip_domains, r.failure); }
    for (t, f) in outcome.traces.iter().zip(outcome.scored_fixes()) { println!("trace call {} true {} nodes {:?} est {:?} fix {:?} err {:.2} sigma {:.2}", t.host_call_id, t.true_call_index, t.node_ids, t.estimated_node_positions_m, t.fix.as_ref().map(|f| (f.position_m, f.used_nodes.clone())), f.horizontal_error_m, f.reported_major_sigma_m); }
    for i in 0..outcome.world.devices.len() { println!("dev {i} true {:?}", outcome.world.position(i, 0.0)); }
}
