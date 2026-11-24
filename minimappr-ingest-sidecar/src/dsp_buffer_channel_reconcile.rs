//! Moving buffered audio with its sensor when the registry's sensor order
//! changes (a join that sorts mid-list, or an eviction).

/// How the worker's buffers for one key must change to match a new order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum BufferChannelReconcile {
    Unchanged,
    /// New sensors sort after every existing one: existing channels keep
    /// their index, so a parked solve stays valid as is.
    AppendChannels,
    /// Existing channels move or vanish. Entry `i` is the current channel
    /// that becomes channel `i` (`None` = new, empty buffer).
    Reorder {
        source_channel_by_new_channel: Vec<Option<usize>>,
    },
}

pub(crate) fn plan_buffer_channel_reconcile(
    current_sensor_ids: &[String],
    target_sensor_ids: &[String],
) -> BufferChannelReconcile {
    if current_sensor_ids == target_sensor_ids {
        return BufferChannelReconcile::Unchanged;
    }
    if target_sensor_ids.starts_with(current_sensor_ids) {
        return BufferChannelReconcile::AppendChannels;
    }
    BufferChannelReconcile::Reorder {
        source_channel_by_new_channel: target_sensor_ids
            .iter()
            .map(|sensor_id| {
                current_sensor_ids
                    .iter()
                    .position(|current| current == sensor_id)
            })
            .collect(),
    }
}
