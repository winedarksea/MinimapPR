//! Joint TDOA + DOA solver tests over mixed node descriptors.

use nalgebra::{UnitQuaternion, Vector3};

use super::bearing_fusion::{fuse_bearings, BearingFusionConfig, BearingObservation};
use super::canopy_height_prior::CanopyHeightPrior;
use super::joint_objective::{solve_joint_position, JointSolverConfig};
use super::joint_residual_terms::{
    DoaDirection, DoaObservation, JointObservations, MicPairTdoaObservation, WorldBearingRay,
};
use super::node_descriptor::{
    ClockSource, NodeClock, NodeDescriptor, NodeKind, NodeOrientation, OrientationSource,
    PositionSource,
};
use super::test_support::{sample_covariance, DeterministicGaussian, SOUND_SPEED_MPS};
use super::TdoaSolverError;

const SIRITH_OFFSETS_M: [[f64; 3]; 4] = [
    [-0.016238, 0.025000, -0.010205],
    [0.027063, 0.000000, -0.010205],
    [-0.016238, -0.025000, -0.010205],
    [0.005413, 0.000000, 0.030615],
];
const PHONE_DOMAIN: u32 = 1;
const STEREO_HALF_SPACING_M: f64 = 0.075;

fn node(
    node_id: &str,
    kind: NodeKind,
    position_m: [f64; 3],
    mic_offsets_m: Vec<[f64; 3]>,
    quaternion: UnitQuaternion<f64>,
) -> NodeDescriptor {
    let q = quaternion.quaternion();
    NodeDescriptor {
        node_id: node_id.into(),
        kind,
        position_m,
        position_source: PositionSource::AcousticMesh,
        position_std_m: 0.0,
        mic_offsets_m,
        orientation: NodeOrientation {
            body_to_world_quaternion_wxyz: [q.w, q.i, q.j, q.k],
            source: OrientationSource::Imu,
            orientation_std_rad: 0.0,
        },
        clock: NodeClock {
            source: ClockSource::AcousticChirp,
            clock_domain_id: PHONE_DOMAIN,
            clock_offset_std_s: 0.0,
        },
        has_speaker: true,
    }
}

fn mono(node_id: &str, position_m: [f64; 3]) -> NodeDescriptor {
    node(
        node_id,
        NodeKind::PhoneMono,
        position_m,
        vec![[0.0; 3]],
        UnitQuaternion::identity(),
    )
}

fn stereo(node_id: &str, position_m: [f64; 3], quaternion: UnitQuaternion<f64>) -> NodeDescriptor {
    node(
        node_id,
        NodeKind::PhoneStereo,
        position_m,
        vec![
            [0.0, 0.0, STEREO_HALF_SPACING_M],
            [0.0, 0.0, -STEREO_HALF_SPACING_M],
        ],
        quaternion,
    )
}

fn tetra(node_id: &str, position_m: [f64; 3]) -> NodeDescriptor {
    node(
        node_id,
        NodeKind::TetrahedralArray,
        position_m,
        SIRITH_OFFSETS_M.to_vec(),
        UnitQuaternion::identity(),
    )
}

fn exact_tdoa(
    nodes: &[NodeDescriptor],
    source: &Vector3<f64>,
    a: (usize, usize),
    b: (usize, usize),
    timing_std_s: f64,
) -> MicPairTdoaObservation {
    let mic_a = nodes[a.0].mic_world_position(a.1).unwrap();
    let mic_b = nodes[b.0].mic_world_position(b.1).unwrap();
    MicPairTdoaObservation {
        node_a: nodes[a.0].node_id.clone(),
        mic_a: a.1,
        node_b: nodes[b.0].node_id.clone(),
        mic_b: b.1,
        tdoa_seconds: ((source - mic_a).norm() - (source - mic_b).norm()) / SOUND_SPEED_MPS,
        timing_std_s,
    }
}

fn all_inter_node_tdoas(
    nodes: &[NodeDescriptor],
    source: &Vector3<f64>,
    timing_std_s: f64,
) -> Vec<MicPairTdoaObservation> {
    let mut observations = Vec::new();
    for first in 0..nodes.len() {
        for second in first + 1..nodes.len() {
            observations.push(exact_tdoa(
                nodes,
                source,
                (first, 0),
                (second, 0),
                timing_std_s,
            ));
        }
    }
    observations
}

#[test]
fn tetra_doa_plus_two_mono_phones_fixes_a_canopy_bird() {
    let source = Vector3::new(14.0, 9.0, 11.0);
    let nodes = vec![
        tetra("array", [0.0, 0.0, 1.5]),
        mono("p1", [10.0, -3.0, 1.4]),
        mono("p2", [4.0, 12.0, 1.2]),
    ];
    let world_direction = (source
        - nodes[0]
            .mic_world_positions()
            .unwrap()
            .iter()
            .sum::<Vector3<f64>>()
            / 4.0)
        .normalize();
    let observations = JointObservations {
        mic_pair_tdoas: all_inter_node_tdoas(&nodes, &source, 2.0e-5),
        doas: vec![DoaObservation {
            node_id: "array".into(),
            direction: DoaDirection::BodyFrameAzimuthElevation {
                azimuth_rad: world_direction.y.atan2(world_direction.x),
                elevation_rad: world_direction.z.asin(),
            },
            angular_std_rad: 0.05,
        }],
        world_bearing_rays: vec![],
    };
    let config = JointSolverConfig {
        canopy_height_prior: Some(CanopyHeightPrior::default()),
        ..JointSolverConfig::default()
    };
    let result = solve_joint_position(&nodes, &observations, &config).unwrap();
    assert!(
        (result.position_m - source).norm() < 1e-4,
        "{:?}",
        result.position_m
    );
    assert!(result.chi_square < 1e-8);
    assert!(result.covariance_m2.trace().is_finite() && result.covariance_m2.trace() > 0.0);
    // 3 inter-node TDOAs + 3 DOA rows + 1 canopy row.
    assert_eq!(result.residual_count, 3 + 3 + 1);
    // 3 TDOAs + 2 angular dof for the DOA, minus 3 position unknowns.
    assert_eq!(result.degrees_of_freedom, 3 + 2 - 3);
}

#[test]
fn stereo_cones_with_rotated_phones_are_zero_residual_at_truth_and_recovered() {
    let source = Vector3::new(-6.0, 15.0, 8.0);
    // Phones held at different attitudes: mic axes point along world x and y.
    let tilt_to_x =
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), std::f64::consts::FRAC_PI_2);
    let tilt_to_y =
        UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -std::f64::consts::FRAC_PI_2);
    let nodes = vec![
        stereo("s1", [0.0, 0.0, 1.4], tilt_to_x),
        stereo("s2", [8.0, 2.0, 1.3], tilt_to_y),
        mono("m1", [2.0, 10.0, 1.5]),
        mono("m2", [-9.0, 4.0, 1.1]),
    ];
    let mut tdoas = all_inter_node_tdoas(&nodes, &source, 3.0e-5);
    tdoas.push(exact_tdoa(&nodes, &source, (0, 0), (0, 1), 2.0e-5));
    tdoas.push(exact_tdoa(&nodes, &source, (1, 0), (1, 1), 2.0e-5));
    let observations = JointObservations {
        mic_pair_tdoas: tdoas,
        ..JointObservations::default()
    };
    let config = JointSolverConfig {
        canopy_height_prior: Some(CanopyHeightPrior::default()),
        ..JointSolverConfig::default()
    };
    let result = solve_joint_position(&nodes, &observations, &config).unwrap();
    assert!(
        (result.position_m - source).norm() < 1e-3,
        "{:?}",
        result.position_m
    );
}

#[test]
fn single_stereo_phone_alone_is_an_explicit_unobservable_error() {
    let source = Vector3::new(5.0, 5.0, 5.0);
    let nodes = vec![stereo("s1", [0.0, 0.0, 1.4], UnitQuaternion::identity())];
    let observations = JointObservations {
        mic_pair_tdoas: vec![exact_tdoa(&nodes, &source, (0, 0), (0, 1), 2.0e-5)],
        ..JointObservations::default()
    };
    let outcome = solve_joint_position(&nodes, &observations, &JointSolverConfig::default());
    assert!(
        matches!(
            outcome,
            Err(TdoaSolverError::Singular { .. }) | Err(TdoaSolverError::IllConditioned { .. })
        ),
        "{outcome:?}"
    );
}

#[test]
fn coplanar_mono_phones_report_the_mirror_unless_the_canopy_prior_breaks_it() {
    let source = Vector3::new(6.0, 7.0, 10.0);
    let nodes = vec![
        mono("p0", [0.0, 0.0, 0.0]),
        mono("p1", [15.0, 0.0, 0.0]),
        mono("p2", [15.0, 14.0, 0.0]),
        mono("p3", [0.0, 15.0, 0.0]),
    ];
    let observations = JointObservations {
        mic_pair_tdoas: all_inter_node_tdoas(&nodes, &source, 2.0e-5),
        ..JointObservations::default()
    };
    let free = solve_joint_position(&nodes, &observations, &JointSolverConfig::default()).unwrap();
    let alternate = free
        .ambiguous_alternate_position_m
        .expect("mirror should be reported");
    assert!((free.position_m.z + alternate.z).abs() < 1e-3 && free.position_m.z.abs() > 5.0);

    let config = JointSolverConfig {
        canopy_height_prior: Some(CanopyHeightPrior::default()),
        ..JointSolverConfig::default()
    };
    let with_prior = solve_joint_position(&nodes, &observations, &config).unwrap();
    assert!(
        (with_prior.position_m - source).norm() < 1e-3,
        "{:?}",
        with_prior.position_m
    );
    assert!(with_prior.ambiguous_alternate_position_m.is_none());
}

#[test]
fn nodes_in_different_clock_domains_cannot_share_a_tdoa() {
    let source = Vector3::new(3.0, 4.0, 5.0);
    let mut nodes = vec![mono("p0", [0.0, 0.0, 0.0]), mono("p1", [10.0, 0.0, 0.0])];
    let observation = exact_tdoa(&nodes, &source, (0, 0), (1, 0), 1e-5);
    nodes[1].clock.clock_domain_id = PHONE_DOMAIN + 1;
    let observations = JointObservations {
        mic_pair_tdoas: vec![observation],
        ..JointObservations::default()
    };
    assert!(matches!(
        solve_joint_position(&nodes, &observations, &JointSolverConfig::default()),
        Err(TdoaSolverError::ClockDomainMismatch { .. })
    ));
}

#[test]
fn fusion_cones_as_doa_terms_agree_with_closed_form_triangulation() {
    let source = Vector3::new(20.0, 30.0, 6.0);
    let cone = |node_id: &str, origin: [f64; 3]| {
        let direction = (source - Vector3::new(origin[0], origin[1], origin[2])).normalize();
        BearingObservation {
            node_id: node_id.into(),
            origin_m: origin,
            direction: [direction.x, direction.y, direction.z],
            lateral_std_m: 0.5,
            confidence: 0.8,
            event_time_ns: 0,
        }
    };
    let cones = [cone("a", [0.0, 0.0, 1.5]), cone("b", [40.0, 0.0, 1.5])];
    let closed_form = fuse_bearings(&cones, &BearingFusionConfig::default()).unwrap();
    let range_prior_m = 30.0;
    let observations = JointObservations {
        world_bearing_rays: cones
            .iter()
            .map(|cone| WorldBearingRay::from_bearing_observation(cone, range_prior_m))
            .collect(),
        ..JointObservations::default()
    };
    let joint = solve_joint_position(
        &[mono("anchor", [20.0, 0.0, 1.5])],
        &observations,
        &JointSolverConfig::default(),
    )
    .unwrap();
    assert!((joint.position_m - closed_form.position_m).norm() < 1e-6);
}

#[test]
fn joint_covariance_matches_monte_carlo_scatter() {
    let source = Vector3::new(8.0, 6.0, 9.0);
    let nodes = vec![
        mono("p0", [0.0, 0.0, 1.0]),
        mono("p1", [16.0, 1.0, 1.4]),
        mono("p2", [15.0, 15.0, 1.2]),
        mono("p3", [-1.0, 14.0, 1.6]),
        mono("p4", [7.0, -6.0, 1.1]),
    ];
    let timing_std_s = 2.0e-5;
    let ideal = all_inter_node_tdoas(&nodes, &source, timing_std_s);
    // Independent per-row noise matches the whitened model's assumption.
    let mut noise = DeterministicGaussian::new(1234);
    let trial_count = 400;
    let mut solved = Vec::with_capacity(trial_count);
    let mut predicted = None;
    for _ in 0..trial_count {
        let noisy: Vec<MicPairTdoaObservation> = ideal
            .iter()
            .map(|row| MicPairTdoaObservation {
                tdoa_seconds: row.tdoa_seconds + noise.sample(timing_std_s),
                ..row.clone()
            })
            .collect();
        let observations = JointObservations {
            mic_pair_tdoas: noisy,
            ..JointObservations::default()
        };
        let config = JointSolverConfig {
            canopy_height_prior: Some(CanopyHeightPrior::default()),
            ..JointSolverConfig::default()
        };
        let result = solve_joint_position(&nodes, &observations, &config).unwrap();
        predicted.get_or_insert(result.covariance_m2);
        solved.push(result.position_m);
    }
    let empirical = sample_covariance(&solved);
    let predicted = predicted.unwrap();
    let trace_ratio = empirical.trace() / predicted.trace();
    assert!(
        (0.8..1.25).contains(&trace_ratio),
        "trace ratio {trace_ratio}"
    );
    for axis in 0..3 {
        let ratio = empirical[(axis, axis)] / predicted[(axis, axis)];
        assert!(
            (0.7..1.4).contains(&ratio),
            "axis {axis} variance ratio {ratio}"
        );
    }
}
