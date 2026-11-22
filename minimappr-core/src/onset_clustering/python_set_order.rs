//! Iteration order of a CPython (3.12) `set` of small non-negative ints.
//!
//! Why: the reference picks a cluster's LM reference sensor with
//! `Counter(... for index in cluster.frame_indices).most_common(1)`, which
//! breaks count ties by the set's iteration order. That order is the hash
//! table's slot order (hash(i) == i, linear probing, resizes), not ascending,
//! e.g. `{3, 9}` iterates as `[9, 3]`. Decision-level parity on ties needs it.

/// `PySet_MINSIZE`.
const MINIMUM_TABLE_SIZE: usize = 8;
/// `LINEAR_PROBES` in `setobject.c`.
const LINEAR_PROBES: usize = 9;
/// `PERTURB_SHIFT` in `setobject.c`.
const PERTURB_SHIFT: u32 = 5;
/// Resize when `fill * 5 >= mask * 3`.
const FILL_NUMERATOR: usize = 5;
const FILL_DENOMINATOR: usize = 3;
/// Growth factor below CPython's 50000-entry threshold.
const GROWTH_FACTOR: usize = 4;

fn probe_free_slot(table: &[Option<usize>], value: usize) -> usize {
    let mask = table.len() - 1;
    let mut slot = value & mask;
    let mut perturb = value;
    loop {
        let probes = if slot + LINEAR_PROBES <= mask {
            LINEAR_PROBES
        } else {
            0
        };
        for offset in 0..=probes {
            if table[slot + offset].is_none() {
                return slot + offset;
            }
        }
        perturb >>= PERTURB_SHIFT;
        slot = (slot.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
    }
}

/// The order Python iterates `{v for v in insertion}`.
pub fn small_int_set_iteration_order(insertion: &[usize]) -> Vec<usize> {
    let mut table: Vec<Option<usize>> = vec![None; MINIMUM_TABLE_SIZE];
    let mut used = 0;
    for &value in insertion {
        if table.contains(&Some(value)) {
            continue;
        }
        let slot = probe_free_slot(&table, value);
        table[slot] = Some(value);
        used += 1;
        let mask = table.len() - 1;
        if used * FILL_NUMERATOR >= mask * FILL_DENOMINATOR {
            let minimum_used = used * GROWTH_FACTOR;
            let mut new_size = MINIMUM_TABLE_SIZE;
            while new_size <= minimum_used {
                new_size <<= 1;
            }
            let mut resized: Vec<Option<usize>> = vec![None; new_size];
            for entry in table.iter().flatten() {
                let slot = probe_free_slot(&resized, *entry);
                resized[slot] = Some(*entry);
            }
            table = resized;
        }
    }
    table.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::small_int_set_iteration_order;

    #[test]
    fn matches_known_cpython_orders() {
        // python3.12 -c "print(list({3, 9}))" -> [9, 3]
        assert_eq!(small_int_set_iteration_order(&[3, 9]), vec![9, 3]);
        assert_eq!(small_int_set_iteration_order(&[2, 10]), vec![2, 10]);
        assert_eq!(small_int_set_iteration_order(&[5, 1, 5]), vec![1, 5]);
        // Random insertion sequences checked against CPython 3.12.
        assert_eq!(
            small_int_set_iteration_order(&[36, 54, 51, 48, 4]),
            vec![36, 4, 48, 51, 54]
        );
        assert_eq!(
            small_int_set_iteration_order(&[7, 31, 48, 28, 30, 41, 24, 50, 13]),
            vec![7, 41, 13, 48, 50, 24, 28, 30, 31]
        );
        assert_eq!(
            small_int_set_iteration_order(&[31, 1, 57, 53]),
            vec![1, 57, 53, 31]
        );
        assert_eq!(
            small_int_set_iteration_order(&[27, 38, 48, 49, 0, 44, 28, 17, 46, 51, 14, 37, 60]),
            vec![0, 37, 38, 44, 46, 14, 48, 49, 17, 51, 60, 27, 28]
        );
        assert_eq!(
            small_int_set_iteration_order(&[57, 20, 1, 1]),
            vec![57, 20, 1]
        );
    }
}
