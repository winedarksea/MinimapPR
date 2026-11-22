//! Per-thread heap accounting for the resource-bounds gate.
//!
//! A binary opts in with
//! `#[global_allocator] static ALLOCATOR: CountingAllocator = CountingAllocator;`
//! (the resource test and the nightly runner do). Counters are thread-local,
//! so measurements are not polluted by other tests running in parallel;
//! without the allocator installed every measurement reads 0 and the gate
//! reports "not measured" instead of passing silently.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

thread_local! {
    static LIVE_BYTES: Cell<isize> = const { Cell::new(0) };
    static PEAK_BYTES: Cell<isize> = const { Cell::new(0) };
}

static INSTALLED: AtomicBool = AtomicBool::new(false);

pub struct CountingAllocator;

fn record(delta: isize) {
    // try_with: TLS may already be torn down during thread exit.
    let _ = LIVE_BYTES.try_with(|live| {
        let now = live.get() + delta;
        live.set(now);
        let _ = PEAK_BYTES.try_with(|peak| {
            if now > peak.get() {
                peak.set(now);
            }
        });
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        INSTALLED.store(true, Ordering::Relaxed);
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size() as isize);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        record(-(layout.size() as isize));
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let grown = unsafe { System.realloc(pointer, layout, new_size) };
        if !grown.is_null() {
            record(new_size as isize - layout.size() as isize);
        }
        grown
    }
}

pub fn is_installed() -> bool {
    INSTALLED.load(Ordering::Relaxed)
}

/// Peak extra heap (bytes) this thread held while running `work`.
pub fn measure_peak_heap<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let baseline = LIVE_BYTES.with(Cell::get);
    PEAK_BYTES.with(|peak| peak.set(baseline));
    let value = work();
    let peak = PEAK_BYTES.with(Cell::get);
    (value, (peak - baseline).max(0) as usize)
}
