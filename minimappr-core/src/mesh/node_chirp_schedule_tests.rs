//! A chirp schedule becomes per-slot listen windows on the session clock,
//! own slots included, so the phone's listener can place a search window
//! for every emission (the closing chirp repeats the opener).

use crate::protocol::session_messages::{ChirpSchedule, ChirpSlotAssignment};
use crate::protocol::wire_codes::SessionId;
use crate::protocol::{FieldMessage, NodeId};

use super::config::MeshConfig;
use super::node_split_brain_tests::{phone, SilentAudio};
use super::node_state_machine::{FieldNodeStateMachine, NodeAction, NodeInput};

const HOST: NodeId = NodeId(1);
const SELF: NodeId = NodeId(2);
const SAMPLES_PER_MS: u64 = 48;

#[test]
fn schedule_yields_listen_slots_and_own_emissions() {
    let config = MeshConfig::default();
    let slot_ms = config.scheduler.slot_duration_ms;
    let mut node = FieldNodeStateMachine::new(SELF, phone(), SessionId([0; 16]), HOST, config.clone(), 0);
    let emitters = [HOST, SELF, HOST];
    let schedule = ChirpSchedule {
        island_id: 0,
        round_id: 7,
        round_start_host_sample_index: 10_000 * SAMPLES_PER_MS,
        slots: emitters
            .iter()
            .enumerate()
            .map(|(index, emitter)| ChirpSlotAssignment {
                emitter_node_id: *emitter,
                slot_offset_samples: (index as u64 * slot_ms * SAMPLES_PER_MS) as u32,
                chirp_low_hz: 13_000.0,
                chirp_high_hz: 16_000.0,
                chirp_code_id: emitter.0 as u16,
            })
            .collect(),
    };
    let actions = node.handle(9_000, 0, NodeInput::MessageReceived { from: HOST, message: FieldMessage::ChirpSchedule(schedule) }, &SilentAudio);
    let emissions: Vec<u64> = actions
        .iter()
        .filter_map(|action| match action {
            NodeAction::EmitChirp { emit_at_ms, .. } => Some(*emit_at_ms),
            _ => None,
        })
        .collect();
    assert_eq!(emissions, vec![10_000 + slot_ms]);
    let listen = actions.iter().find_map(|action| match action {
        NodeAction::ListenForChirpRound { round_start_ms, listen_until_ms, slots, .. } => Some((*round_start_ms, *listen_until_ms, slots.clone())),
        _ => None,
    });
    let (round_start_ms, listen_until_ms, slots) = listen.expect("listen action");
    assert_eq!(round_start_ms, 10_000);
    assert_eq!(slots.len(), 3);
    assert_eq!(slots.iter().map(|slot| slot.emitter_node_id).collect::<Vec<_>>(), emitters.to_vec());
    assert_eq!(slots[2].slot_start_ms, 10_000 + 2 * slot_ms);
    assert_eq!(listen_until_ms, 10_000 + 3 * slot_ms + config.scheduler.round_listen_tail_ms);
}
