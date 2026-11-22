//! Message enum round trips and forward-compatibility handling.

use minimappr_core::protocol::detection_messages::*;
use minimappr_core::protocol::mesh_messages::*;
use minimappr_core::protocol::session_messages::*;
use minimappr_core::protocol::wire_codes::*;
use minimappr_core::protocol::{
    decode_field_message_frame, DecodedFrame, FieldMessage, FieldMessageDecodeError, WireHeader,
    FIELD_PROTOCOL_MAJOR_VERSION, KNOWN_MESSAGE_KIND_IDS,
};

const SESSION: SessionId = SessionId([7; 16]);

fn species() -> Vec<SpeciesScore> {
    vec![
        SpeciesScore {
            label_id: 4_211,
            score: 0.82,
        },
        SpeciesScore {
            label_id: 17,
            score: 0.11,
        },
    ]
}

fn band() -> FrequencyBandHz {
    FrequencyBandHz {
        low_hz: 2_500.0,
        high_hz: 7_800.0,
    }
}

fn detection(local_sequence: u64) -> DetectionEvent {
    DetectionEvent {
        node_id: NodeId(3),
        local_sequence,
        window_first_device_sample_index: 9_876_543_210,
        window_sample_count: 144_000,
        classifier_model_id: 30,
        species_scores: species(),
        stage1_contrast_db: 11.5,
        stage1_active_band: Some(band()),
    }
}

/// One populated sample of every message kind.
fn one_of_every_message() -> Vec<FieldMessage> {
    let descriptor = NodeDescriptor {
        node_kind: NodeKind::Phone,
        platform_name: "android-16".into(),
        app_version: "0.1.0".into(),
        capture_sample_rate_hz: 48_000,
        microphone_count: 2,
        microphone_positions_body_frame_m: vec![[0.0, 0.07, 0.0], [0.0, -0.07, 0.0]],
        can_classify: true,
        can_host: true,
        can_emit_chirps: true,
        has_pps_clock: false,
    };
    vec![
        FieldMessage::JoinRequest(JoinRequest {
            session_id: SESSION,
            join_token: [9; 16],
            requested_role: SessionRole::Joiner,
            node_descriptor: descriptor.clone(),
        }),
        FieldMessage::JoinAccepted(JoinAccepted {
            session_id: SESSION,
            assigned_node_id: NodeId(3),
            host_node_id: NodeId(1),
            chirp_slot_index: 2,
            event_log_head_sequence: 41,
        }),
        FieldMessage::JoinCodeKeyExchange(JoinCodeKeyExchange {
            session_id: SESSION,
            joiner_attempt_id: 77,
            flight: KeyExchangeFlight::JoinerSpakeMessage,
            payload: vec![0x41; 33],
        }),
        FieldMessage::ChirpSchedule(ChirpSchedule {
            island_id: 1,
            round_id: 5,
            round_start_host_sample_index: 1_000_000,
            slots: vec![ChirpSlotAssignment {
                emitter_node_id: NodeId(3),
                slot_offset_samples: 4_800,
                chirp_low_hz: 13_000.0,
                chirp_high_hz: 18_000.0,
                chirp_code_id: 2,
            }],
        }),
        FieldMessage::ChirpArrivalReport(ChirpArrivalReport {
            round_id: 5,
            reporting_node_id: NodeId(4),
            own_emission_device_sample_index: Some(55),
            arrivals: vec![ChirpArrival {
                emitter_node_id: NodeId(3),
                arrival_device_sample_index: 1_234_567,
                arrival_fractional_sample: -0.25,
                peak_to_noise_db: 31.0,
            }],
            motion_state: MotionState::Still,
            own_chirp_inaudible: false,
        }),
        FieldMessage::IslandMembership(IslandMembership {
            island_id: 1,
            membership_epoch: 3,
            member_node_ids: vec![NodeId(1), NodeId(3), NodeId(4)],
        }),
        FieldMessage::DetectionEvent(detection(12)),
        FieldMessage::ReadinessQuery(ReadinessQuery {
            query_id: 8,
            target_node_id: NodeId(4),
            centre_device_sample_index: 2_000_000,
            half_window_samples: 36_000,
            band: band(),
        }),
        FieldMessage::ReadinessAnswer(ReadinessAnswer {
            query_id: 8,
            node_id: NodeId(4),
            clock_in_sync: true,
            audio_still_buffered: true,
            speech_free: true,
            handling_noise_free: false,
            in_band_level_over_floor_db: 6.5,
        }),
        FieldMessage::DoaBearing(DoaBearing {
            node_id: NodeId(9),
            centre_device_sample_index: 3,
            azimuth_deg: 271.5,
            elevation_deg: 12.0,
            azimuth_standard_deviation_deg: 4.0,
            band: band(),
        }),
        FieldMessage::ClipRequest(ClipRequest {
            request_id: 100,
            target_node_id: NodeId(4),
            window_first_device_sample_index: 1_999_000,
            window_sample_count: 72_000,
            decimation_filter_id: 1,
            abandon_after_milliseconds: 4_000,
        }),
        FieldMessage::ClipPayload(ClipPayload {
            request_id: 100,
            node_id: NodeId(4),
            clip_container: b"MMCL\x01rest".to_vec(),
            onsets: vec![ClipOnset {
                device_sample_index: 1_999_500,
                positive_flux_db: 9.0,
            }],
            sensor_frames: vec![SensorFrameSample {
                device_sample_index: 1_999_000,
                orientation_quaternion: [1.0, 0.0, 0.0, 0.0],
                linear_acceleration_m_per_s2: [0.0, 0.1, 0.0],
                pressure_hectopascal: Some(1_013.2),
            }],
        }),
        FieldMessage::ClipPayloadChunk(ClipPayloadChunk {
            request_id: 100,
            node_id: NodeId(4),
            chunk_index: 1,
            chunk_count: 3,
            chunk_bytes: vec![1, 2, 3],
        }),
        FieldMessage::LocoOcclusionRequest(LocoOcclusionRequest {
            request_id: 7,
            target_node_id: NodeId(3),
            classification_window_first_device_sample_index: 9_876_543_210,
            cluster_masks: vec![ClusterOcclusionMask {
                cluster_id: 0,
                frame_ranges: vec![FrameIndexRange {
                    first_frame: 3,
                    end_frame: 9,
                }],
                band_ranges: vec![BandIndexRange {
                    first_band: 20,
                    end_band: 31,
                }],
            }],
            maximum_reruns: 2,
        }),
        FieldMessage::LocoRescoredSpecies(LocoRescoredSpecies {
            request_id: 7,
            node_id: NodeId(3),
            rescores: vec![ClusterRescore {
                occluded_cluster_id: 0,
                species_scores: species(),
            }],
            reruns_used: 1,
        }),
        FieldMessage::TapToHearRequest(TapToHearRequest {
            request_id: 5,
            target_node_id: NodeId(3),
            bird_entity_id: 991,
            window_first_device_sample_index: 9_876_543_210,
        }),
        FieldMessage::TapToHearPayload(TapToHearPayload {
            request_id: 5,
            node_id: NodeId(3),
            clip_container: vec![0xFF; 10],
        }),
        FieldMessage::LayoutModel(LayoutModel {
            island_id: 1,
            layout_epoch: 2,
            origin_latitude_deg: 47.6,
            origin_longitude_deg: -122.3,
            origin_altitude_m: 30.0,
            node_positions: vec![NodePositionEstimate {
                node_id: NodeId(3),
                east_north_up_m: [10.0, -4.0, 1.2],
                position_standard_deviation_m: [0.3, 0.3, 0.8],
            }],
        }),
        FieldMessage::ClockModel(ClockModel {
            island_id: 1,
            clock_epoch: 2,
            reference_node_id: NodeId(1),
            node_clocks: vec![NodeClockEstimate {
                node_id: NodeId(3),
                anchor_node_sample_index: 1_000,
                offset_samples: -12_345.25,
                rate_offset_ppm: 21.5,
                offset_standard_deviation_samples: 0.4,
            }],
        }),
        FieldMessage::BirdEntityUpdate(BirdEntityUpdate {
            bird_entity_id: 991,
            event_log_sequence: 42,
            state: BirdEntityState::Pinpointed,
            island_id: 1,
            position_east_north_up_m: Some([20.0, 31.0, 6.0]),
            bearing: None,
            likelihood_contour_east_north_m: vec![[19.0, 30.0], [21.0, 30.0], [20.0, 33.0]],
            species_belief: species(),
            just_sang: true,
            last_heard_reference_sample_index: 2_000_100,
        }),
        FieldMessage::QueuedDetectionUpload(QueuedDetectionUpload {
            node_id: NodeId(3),
            detections: vec![detection(13), detection(14)],
        }),
        FieldMessage::Heartbeat(Heartbeat {
            node_id: NodeId(3),
            local_sequence: 900,
            device_sample_index_now: 2_000_000,
            thermal_headroom_fraction: 0.6,
            battery_fraction: 0.8,
            placement: PhonePlacement::Pocket,
            motion_state: MotionState::Walking,
            mic_only_mode: false,
            capture_overrun_count: 0,
            gnss_fix: Some(GnssFix {
                latitude_deg: 47.6,
                longitude_deg: -122.3,
                altitude_m: 31.0,
                horizontal_accuracy_m: 4.0,
                fix_device_sample_index: 1_999_000,
            }),
        }),
        FieldMessage::PeopleNearbySummary(PeopleNearbySummary {
            node_id: NodeId(3),
            summary_window_seconds: 60,
            unknown_advertiser_count: 5,
            signal_strength_minimum_dbm: -95,
            signal_strength_median_dbm: -80,
            signal_strength_maximum_dbm: -61,
        }),
        FieldMessage::GroupStatus(GroupStatus {
            node_id: NodeId(3),
            status: GroupStatusCode::HeadingHome,
            note: Some("back at trailhead".into()),
        }),
        FieldMessage::EventLogAppend(EventLogAppend {
            host_node_id: NodeId(1),
            host_term: 2,
            host_head_sequence: 41,
            entries: vec![EventLogEntry {
                sequence: 40,
                host_term: 2,
                field_message_frame: vec![1, 0, 7, 9, 9],
            }],
        }),
        FieldMessage::EventLogCatchUpRequest(EventLogCatchUpRequest {
            node_id: NodeId(3),
            since_sequence: 39,
        }),
        FieldMessage::HostAnnouncement(HostAnnouncement {
            host_node_id: NodeId(1),
            host_term: 2,
            event_log_head_sequence: 41,
            transport: SessionTransportCode::CloudRelay,
            reason: SessionChangeReasonCode::HostSilent,
        }),
        FieldMessage::HostHandover(HostHandover {
            from_host_node_id: NodeId(1),
            to_node_id: NodeId(2),
            host_term: 3,
            reason: SessionChangeReasonCode::HostThermalHeadroomLow,
            event_log_head_sequence: 41,
        }),
        FieldMessage::SessionMemberRecord(SessionMemberRecord {
            node_id: NodeId(3),
            join_order: 2,
            node_descriptor: descriptor.clone(),
            self_path_length_m: 0.02,
        }),
        FieldMessage::NodeStatusChange(NodeStatusChange {
            node_id: NodeId(3),
            local_sequence: 7,
            change: NodeStatusChangeCode::InputRouteChanged,
            device_sample_index: 123_456,
        }),
        FieldMessage::ChirpDirectionReport(ChirpDirectionReport {
            round_id: 9,
            reporting_node_id: NodeId(4),
            emitter_node_id: NodeId(3),
            world_azimuth_deg: 37.5,
            azimuth_standard_deviation_deg: 4.0,
        }),
        FieldMessage::ClassificationRota(ClassificationRota {
            island_id: 1,
            rota_epoch: 3,
            slot_duration_milliseconds: 1_500,
            clumps: vec![ClassificationClump {
                member_node_ids: vec![NodeId(2), NodeId(3), NodeId(5)],
            }],
        }),
        FieldMessage::TransportNotice(TransportNotice {
            host_node_id: NodeId(1),
            transport: SessionTransportCode::LocalHotspot,
            reason: SessionChangeReasonCode::RelayMessageCapReached,
            clip_fetch_enabled: true,
        }),
    ]
}

#[test]
fn every_message_kind_round_trips() {
    let messages = one_of_every_message();
    let mut kinds: Vec<u16> = messages.iter().map(FieldMessage::message_kind_id).collect();
    kinds.sort_unstable();
    assert_eq!(
        kinds, KNOWN_MESSAGE_KIND_IDS,
        "test must cover every kind exactly once"
    );
    for message in messages {
        let frame = message.encode_frame().expect("encodes");
        match decode_field_message_frame(&frame).expect("decodes") {
            DecodedFrame::Known {
                message: decoded,
                minor_version,
            } => {
                assert_eq!(minor_version, 0);
                assert_eq!(decoded, message);
            }
            other => panic!("expected known, got {other:?}"),
        }
    }
}

#[test]
fn group_status_is_reserved_at_a_stable_kind_id() {
    let status = FieldMessage::GroupStatus(GroupStatus {
        node_id: NodeId(1),
        status: GroupStatusCode::RequestAssistance,
        note: None,
    });
    assert_eq!(status.message_kind_id(), 24);
}

#[test]
fn unknown_kind_from_newer_peer_is_kept_not_rejected() {
    let header = WireHeader {
        major_version: FIELD_PROTOCOL_MAJOR_VERSION,
        minor_version: 3,
        message_kind_id: 999,
    };
    let mut frame = postcard::to_allocvec(&header).unwrap();
    frame.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    assert_eq!(
        decode_field_message_frame(&frame),
        Ok(DecodedFrame::UnknownKind {
            minor_version: 3,
            message_kind_id: 999,
            body: vec![0xDE, 0xAD, 0xBE, 0xEF]
        })
    );
}

#[test]
fn newer_minor_with_appended_fields_still_decodes() {
    let message = FieldMessage::DetectionEvent(detection(5));
    let mut frame = message.encode_frame().unwrap();
    frame.extend_from_slice(&[0x05, 0x01, 0x02, 0x03, 0x04, 0x05]); // a future appended Vec<u8>
    match decode_field_message_frame(&frame).unwrap() {
        DecodedFrame::Known {
            message: decoded, ..
        } => assert_eq!(decoded, message),
        other => panic!("{other:?}"),
    }
}

#[test]
fn unknown_wire_code_inside_a_body_degrades_to_unrecognized() {
    let heartbeat = Heartbeat {
        node_id: NodeId(2),
        local_sequence: 1,
        device_sample_index_now: 2,
        thermal_headroom_fraction: 1.0,
        battery_fraction: 1.0,
        placement: PhonePlacement::Unrecognized(77),
        motion_state: MotionState::from(200),
        mic_only_mode: true,
        capture_overrun_count: 0,
        gnss_fix: None,
    };
    let frame = FieldMessage::Heartbeat(heartbeat.clone())
        .encode_frame()
        .unwrap();
    match decode_field_message_frame(&frame).unwrap() {
        DecodedFrame::Known {
            message: FieldMessage::Heartbeat(decoded),
            ..
        } => {
            assert_eq!(decoded.placement, PhonePlacement::Unrecognized(77));
            assert_eq!(decoded.motion_state, MotionState::Unrecognized(200));
            assert_eq!(decoded, heartbeat);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(u8::from(PhonePlacement::Handheld), 3);
}

#[test]
fn other_major_version_and_malformed_frames_are_errors() {
    let header = WireHeader {
        major_version: 2,
        minor_version: 0,
        message_kind_id: 7,
    };
    let frame = postcard::to_allocvec(&header).unwrap();
    assert_eq!(
        decode_field_message_frame(&frame),
        Err(FieldMessageDecodeError::UnsupportedMajorVersion(2))
    );
    assert_eq!(
        decode_field_message_frame(&[]),
        Err(FieldMessageDecodeError::MalformedHeader)
    );
    let full = FieldMessage::DetectionEvent(detection(1))
        .encode_frame()
        .unwrap();
    assert_eq!(
        decode_field_message_frame(&full[..full.len() / 2]),
        Err(FieldMessageDecodeError::MalformedBody { message_kind_id: 7 })
    );
}

#[test]
fn readiness_query_stays_tiny() {
    let frame = FieldMessage::ReadinessQuery(ReadinessQuery {
        query_id: 8,
        target_node_id: NodeId(4),
        centre_device_sample_index: 86_400 * 48_000,
        half_window_samples: 36_000,
        band: band(),
    })
    .encode_frame()
    .unwrap();
    // Plan budgets ~20 B for the readiness query.
    assert!(
        frame.len() <= 24,
        "readiness query is {} bytes",
        frame.len()
    );
}
