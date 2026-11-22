//! Per-solve runtime of layout SRP on simulated clips (release):
//! `cargo bench -p minimappr-core --bench layout_srp`.
//! Measures searcher construction (grid + interval tables, once per layout)
//! separately from `localize` (alignment, curves, coarse scan, refinement,
//! contour), which is what runs per call / onset frame.

#[path = "../tests/srp_layout_support/mod.rs"]
mod srp_layout_support;

use criterion::{criterion_group, criterion_main, Criterion};
use minimappr_core::srp_layout::{LayoutSrpSearcher, NodeClip};
use srp_layout_support::*;

fn layout_srp_benchmarks(criterion: &mut Criterion) {
    let cases = [
        ("5_phones_inside", near_phone_layout(0), [7.0, 9.0, 6.0]),
        ("5_phones_100m", near_phone_layout(0), [80.0, 70.0, 8.0]),
        ("3_phones_2_tetra_inside", {
            let mut devices = near_phone_layout(0);
            devices.truncate(3);
            devices.push(tetra("tetra-0", [6.0, 21.0, 1.5]));
            devices.push(tetra("tetra-1", [-5.0, 11.0, 1.5]));
            devices
        }, [7.0, 9.0, 6.0]),
    ];
    for (name, devices, source) in cases {
        let rendered = render(&scene(devices, source, 0, 1));
        let nodes = layout_nodes(&rendered.plan);
        let config = config_for(&rendered.plan);
        let window = call_window(&rendered.plan);
        let clips: Vec<NodeClip<'_>> = rendered
            .channels_per_device
            .iter()
            .map(|channels| NodeClip { channels: channels.iter().map(Vec::as_slice).collect(), first_sample_index: 0 })
            .collect();
        criterion.bench_function(&format!("searcher_new/{name}"), |bencher| {
            bencher.iter(|| LayoutSrpSearcher::new(&nodes, &config).unwrap())
        });
        let searcher = LayoutSrpSearcher::new(&nodes, &config).unwrap();
        eprintln!(
            "{name}: {} coarse cells x {} pairs, window {} samples",
            searcher.coarse_cell_count(),
            searcher.pair_count(),
            window.sample_count
        );
        criterion.bench_function(&format!("localize/{name}"), |bencher| {
            bencher.iter(|| searcher.localize(&clips, window).unwrap())
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = layout_srp_benchmarks
}
criterion_main!(benches);
