//! Developer aid: for each arrival-time solve in one run, re-solve with
//! truth substituted piece by piece (clock/timing, phone heights,
//! horizontal layout) to see which input carries a fix's error.
//!
//! cargo run --release -p minimappr-mesh-sim --example diagnose_fix_errors -- <spec> <seed>
//!
//! Specs: host_death, thermal, network_faults, offline_host_loss, island_split_merge.
//! Timing residuals are de-meaned (only differences enter a TDOA).

use minimappr_core::layout::gps_anchor_fit::weighted_kabsch_2d;
use minimappr_core::mesh::arrival_time_localizer::{ArrivalTimeLocalizer, NodeArrival};
use minimappr_core::mesh::host_transport_policy::SessionTransportPreference;
use minimappr_mesh_sim::scenario::ScenarioSpec;
use minimappr_mesh_sim::scenarios::{host_scenarios, network_scenarios, sync_scenarios};
use minimappr_mesh_sim::ScenarioRun;

fn spec_named(name: &str, seed: u64) -> ScenarioSpec {
    match name {
        "host_death" => host_scenarios::host_death_spec(seed),
        "thermal" => host_scenarios::thermal_spec(seed, SessionTransportPreference::Auto),
        "network_faults" => network_scenarios::network_faults_spec(seed),
        "offline_host_loss" => network_scenarios::offline_host_loss_spec(seed),
        "island_split_merge" => sync_scenarios::island_split_merge_spec(seed),
        other => panic!("unknown spec {other}"),
    }
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let usage = "usage: diagnose_fix_errors <spec> <seed>";
    let name = arguments.next().expect(usage);
    let seed: u64 = arguments.next().expect(usage).parse().expect("seed must be an integer");
    let outcome = ScenarioRun::new(spec_named(&name, seed), seed).run();
    let localizer = ArrivalTimeLocalizer::default();
    let world = &outcome.world;
    for (device, host) in outcome.all_hosts() {
        println!("host {device} rounds completed at {:?} ms", host.sync.round_completion_ms);
    }
    for (index, call) in world.calls.iter().enumerate() {
        println!("scripted call {index} at {:.1} s", call.time_s);
    }
    for (index, device) in world.devices.iter().enumerate() {
        let stalls: Vec<String> = device.stalls.iter().map(|s| format!("{:.2} s for {:.3} s (reported {})", s.time_s, s.duration_s, s.reported)).collect();
        println!("device {index} clock offset {:.6} s, drift {:.1} ppm, stalls {stalls:?}", device.clock_offset_s, device.clock_drift * 1e6);
    }
    for trace in &outcome.traces {
        if trace.localizer_inputs.len() < 3 {
            continue;
        }
        let call = &world.calls[trace.true_call_index as usize];
        let devices: Vec<usize> = trace.node_ids.iter().map(|n| n.0 as usize).collect();
        let truth_3d: Vec<[f64; 3]> = devices.iter().map(|d| world.position(*d, call.time_s)).collect();
        let true_arrivals: Vec<f64> = devices.iter().map(|d| world.arrival_time(call.time_s, call.position_m, *d)).collect();
        let inputs = &trace.localizer_inputs;
        // Frame: estimated (anchored) frame → truth via a proper rigid fit.
        let estimated_2d: Vec<[f64; 2]> = inputs.iter().map(|a| [a.position_m[0], a.position_m[1]]).collect();
        let truth_2d: Vec<[f64; 2]> = truth_3d.iter().map(|p| [p[0], p[1]]).collect();
        let to_truth = weighted_kabsch_2d(&estimated_2d, &truth_2d, &vec![1.0; inputs.len()], false);
        let from_truth = weighted_kabsch_2d(&truth_2d, &estimated_2d, &vec![1.0; inputs.len()], false);
        let timing_residual: Vec<f64> = inputs.iter().zip(&true_arrivals).map(|(a, t)| a.reference_arrival_s - t).collect();
        let mean = timing_residual.iter().sum::<f64>() / timing_residual.len() as f64;
        let error_of = |arrivals: &[NodeArrival]| -> String {
            match localizer.localize(arrivals) {
                Ok(fix) => {
                    let p = to_truth.apply([fix.fix.position_m[0], fix.fix.position_m[1]]);
                    format!(
                        "err {:.2} σ {:.2} z {:.1} used {}",
                        (p[0] - call.position_m[0]).hypot(p[1] - call.position_m[1]),
                        fix.fix.horizontal_major_sigma_m(),
                        fix.fix.position_m[2],
                        fix.used_arrival_indices.len()
                    )
                }
                Err(error) => format!("{error:?}"),
            }
        };
        let with = |true_timing: bool, true_height: bool, true_horizontal: bool| -> Vec<NodeArrival> {
            inputs
                .iter()
                .enumerate()
                .map(|(k, a)| {
                    let mut a = a.clone();
                    if true_timing {
                        a.reference_arrival_s = true_arrivals[k] + mean;
                    }
                    if true_horizontal {
                        let h = from_truth.apply(truth_2d[k]);
                        a.position_m[0] = h[0];
                        a.position_m[1] = h[1];
                    }
                    if true_height {
                        a.position_m[2] = truth_3d[k][2];
                    }
                    a
                })
                .collect()
        };
        println!("call {} host call {} nodes {:?} src {:?}", trace.true_call_index, trace.host_call_id, devices, call.position_m);
        for (k, a) in inputs.iter().enumerate() {
            let h = to_truth.apply([a.position_m[0], a.position_m[1]]);
            println!(
                "  n{} timing {:+.1} µs (σ {:.1}) horiz {:.2} m z_est {:.2} z_true {:.2} σpos {:.2}",
                devices[k],
                (timing_residual[k] - mean) * 1e6,
                a.timing_std_s * 1e6,
                (h[0] - truth_2d[k][0]).hypot(h[1] - truth_2d[k][1]),
                a.position_m[2],
                truth_3d[k][2],
                a.position_std_m
            );
        }
        println!("  as solved:            {}", error_of(inputs));
        println!("  true timing:          {}", error_of(&with(true, false, false)));
        println!("  true heights:         {}", error_of(&with(false, true, false)));
        println!("  true horizontal:      {}", error_of(&with(false, false, true)));
        println!("  true heights+horiz:   {}", error_of(&with(false, true, true)));
        println!("  all true:             {}", error_of(&with(true, true, true)));
    }
}
