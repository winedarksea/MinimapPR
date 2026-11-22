//! Resource bounds (time and peak heap per solve and per session). Its own
//! test binary so the counting allocator is installed and its thread-local
//! counters see only this gate. Reproduce with
//! `cargo test --release -p minimappr-sim-gates --test statistical_resources -- --nocapture`.

use minimappr_sim_gates::gates::assert_pull_request_gate;
use minimappr_sim_gates::heap_accounting::CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn resource_bounds() {
    assert_pull_request_gate("resource_bounds");
}

#[test]
fn every_catalogued_gate_has_a_test_target() {
    for gate in minimappr_sim_gates::gates::GATES {
        let path = format!("{}/tests/{}.rs", env!("CARGO_MANIFEST_DIR"), gate.test_target);
        let source = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing {path}"));
        assert!(source.contains(&format!("\"{}\"", gate.name)), "{} has no test in {path}", gate.name);
    }
}
