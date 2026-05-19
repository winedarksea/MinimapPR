// Include only the self-contained modules needed for these benchmarks.
// Both dsp.rs and gcc_phat.rs have no intra-crate dependencies.
#[path = "../src/dsp.rs"]
#[allow(dead_code, unused_imports)]
mod dsp;
#[path = "../src/gcc_phat.rs"]
#[allow(dead_code, unused_imports)]
mod gcc_phat;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use dsp::SensorStreamBuffer;
use gcc_phat::tetrahedral_gcc_phat;

/// Hot path inside the DSP worker: append a 512-sample frame then query coverage.
fn bench_buffer_append_and_coverage(c: &mut Criterion) {
    let sr = 16_000_u32;
    let frame_samples = 512_usize;
    let ns_per_frame = frame_samples as i128 * 1_000_000_000_i128 / i128::from(sr);
    let t0 = 1_000_000_000_000_000_000_i128;
    let payload: Vec<f32> = (0..frame_samples).map(|i| (i as f32).sin()).collect();

    let mut group = c.benchmark_group("capture_plane/buffer");
    group.throughput(Throughput::Elements(frame_samples as u64));

    group.bench_function("append_32x512_samples", |b| {
        b.iter_batched(
            || SensorStreamBuffer::new(sr, 10.0),
            |mut buf| {
                for i in 0..32_i128 {
                    buf.append(t0 + i * ns_per_frame, &payload, None, None)
                        .unwrap();
                }
                buf
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("coverage_query_after_32_frames", |b| {
        let mut buf = SensorStreamBuffer::new(sr, 10.0);
        for i in 0..32_i128 {
            buf.append(t0 + i * ns_per_frame, &payload, None, None)
                .unwrap();
        }
        let end_ns = t0 + 32 * ns_per_frame;
        b.iter(|| buf.coverage_ending_at(end_ns, 32.0 * frame_samples as f64 / f64::from(sr)));
    });

    group.finish();
}

/// Single GCC-PHAT call for one mic pair (512-sample window at 16 kHz).
fn bench_gcc_phat_single_pair(c: &mut Criterion) {
    let sr = 16_000_u32;
    let n = 512_usize;
    let ch1: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
    let ch2: Vec<f32> = (2..n + 2).map(|i| (i as f32 * 0.1).sin()).collect();

    let mut group = c.benchmark_group("capture_plane/gcc_phat");
    group.throughput(Throughput::Elements(n as u64));

    group.bench_with_input(
        BenchmarkId::new("single_pair", n),
        &(&ch1, &ch2),
        |b, (c1, c2)| {
            b.iter(|| gcc_phat::gcc_phat(c1, c2, sr, 4));
        },
    );
    group.finish();
}

/// All 6 tetrahedral pairs in one call (512-sample window).
fn bench_tetrahedral_gcc_phat(c: &mut Criterion) {
    let sr = 16_000_u32;
    let n = 512_usize;
    let channels: [Vec<f32>; 4] = core::array::from_fn(|ch| {
        (0..n)
            .map(|i| ((i + ch * 100) as f32 * 0.1).sin())
            .collect()
    });

    let mut group = c.benchmark_group("capture_plane/tetrahedral");
    group.throughput(Throughput::Elements(6 * n as u64));

    group.bench_function("6_pairs_512_samples", |b| {
        b.iter(|| tetrahedral_gcc_phat(&channels, sr, None));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_buffer_append_and_coverage,
    bench_gcc_phat_single_pair,
    bench_tetrahedral_gcc_phat
);
criterion_main!(benches);
