//! Which dispatch-gate state (trigger cooldown, localization cadence,
//! heartbeat, classifier-render schedule, the remembered event solve) a frame
//! uses.
//!
//! Why not the buffer key: every node of a cluster shares one buffer. Keyed
//! by it, whichever node posted first in a slot consumed the cluster's slots,
//! and since sensors at another sample rate are excluded from a solve (no
//! resampling), a node at a different rate never got a solve, render or
//! classification of its own. A solve's usable sensors are set by its solve
//! rate (the reporter's), so gates are kept per buffer and solve rate: a
//! same-rate cluster still solves each event once, and each rate group of a
//! mixed-rate cluster gets its own.

/// Appended to the buffer key; a key without it is a plain buffer key.
const SOLVE_SAMPLE_RATE_SEPARATOR: &str = "@solve_sample_rate_hz=";

pub(crate) fn dispatch_gate_key(buffer_key: &str, solve_sample_rate_hz: u32) -> String {
    format!("{buffer_key}{SOLVE_SAMPLE_RATE_SEPARATOR}{solve_sample_rate_hz}")
}

pub(crate) fn buffer_key_of_dispatch_gate_key(gate_key: &str) -> &str {
    gate_key
        .split_once(SOLVE_SAMPLE_RATE_SEPARATOR)
        .map_or(gate_key, |(buffer_key, _)| buffer_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_keys_split_by_rate_and_map_back_to_their_buffer() {
        let buffer_key = "cluster::auto-gps::audio_main";
        let at_48k = dispatch_gate_key(buffer_key, 48_000);
        assert_ne!(at_48k, dispatch_gate_key(buffer_key, 44_100));
        assert_eq!(at_48k, dispatch_gate_key(buffer_key, 48_000));
        assert_eq!(buffer_key_of_dispatch_gate_key(&at_48k), buffer_key);
        assert_eq!(buffer_key_of_dispatch_gate_key(buffer_key), buffer_key);
    }
}
