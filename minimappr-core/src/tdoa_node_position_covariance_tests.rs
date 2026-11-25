use super::*;

const SOUND_SPEED_MPS: f64 = 343.0;
/// Python's result for each case (`python -m tests.node_position_covariance_parity_cases`).
const PARITY_FIXTURE_JSON: &str =
    include_str!("../../tests/fixtures/node_position_covariance_parity_cases.json");
/// Both sides are f64 with the same operation order up to summation; the
/// slack covers the ridge-inverted condition number of far-field cases.
const PARITY_RELATIVE_TOLERANCE: f64 = 1.0e-7;
/// Sirith tetra offsets about the node origin.
const TETRA_OFFSETS_M: [[f64; 3]; 4] = [
    [-0.016238, 0.025000, -0.010205],
    [0.027063, 0.000000, -0.010205],
    [-0.016238, -0.025000, -0.010205],
    [0.005413, 0.000000, 0.030615],
];

fn vec3(value: &serde_json::Value) -> [f64; 3] {
    let axes = value.as_array().expect("3-vector");
    [0, 1, 2].map(|axis| axes[axis].as_f64().expect("number"))
}

fn matrix3(value: &serde_json::Value) -> [[f64; 3]; 3] {
    let rows = value.as_array().expect("3x3 matrix");
    [0, 1, 2].map(|row| vec3(&rows[row]))
}

#[test]
fn node_position_term_matches_the_python_parity_fixture() {
    let fixture: serde_json::Value =
        serde_json::from_str(PARITY_FIXTURE_JSON).expect("fixture JSON");
    let sound_speed_mps = fixture["sound_speed_mps"].as_f64().expect("sound speed");
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(cases.len() >= 5);
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let sensor_positions_m: Vec<[f64; 3]> = case["sensor_positions_m"]
            .as_array()
            .expect("sensors")
            .iter()
            .map(vec3)
            .collect();
        let rows: Vec<WeightedTdoaRow> = case["tdoa_rows"]
            .as_array()
            .expect("rows")
            .iter()
            .map(|row| WeightedTdoaRow {
                sensor_a: row["sensor_a"].as_u64().expect("sensor_a") as usize,
                sensor_b: row["sensor_b"].as_u64().expect("sensor_b") as usize,
                weight: row["weight"].as_f64().expect("weight"),
            })
            .collect();
        let node_index_by_sensor: Vec<usize> = case["node_index_by_sensor"]
            .as_array()
            .expect("nodes")
            .iter()
            .map(|node| node.as_u64().expect("node index") as usize)
            .collect();
        let node_position_std_m: Vec<f64> = case["node_position_std_m"]
            .as_array()
            .expect("stds")
            .iter()
            .map(|std_m| std_m.as_f64().expect("std"))
            .collect();
        let rust = node_position_error_covariance_m2(
            vec3(&case["position_m"]),
            &sensor_positions_m,
            &rows,
            matrix3(&case["additional_normal_matrix"]),
            sound_speed_mps,
            SensorNodePositionStd {
                node_index_by_sensor: &node_index_by_sensor,
                node_position_std_m: &node_position_std_m,
            },
        );
        let expected = &case["expected_covariance_m2"];
        if expected.is_null() {
            assert!(
                rust.is_none(),
                "{name}: Python states no term, Rust {rust:?}"
            );
            continue;
        }
        let expected = matrix3(expected);
        let rust = rust.unwrap_or_else(|| panic!("{name}: Python has a term, Rust none"));
        let scale = expected
            .iter()
            .flatten()
            .fold(0.0_f64, |max, value| max.max(value.abs()));
        for row in 0..3 {
            for column in 0..3 {
                let difference = (rust[row][column] - expected[row][column]).abs();
                assert!(
                    difference <= PARITY_RELATIVE_TOLERANCE * scale,
                    "{name}[{row}][{column}]: Rust {} vs Python {}",
                    rust[row][column],
                    expected[row][column]
                );
            }
        }
    }
}

#[test]
fn one_rigid_node_moves_the_fix_by_its_own_error() {
    // Every sensor on one node: shifting the whole array by d moves the
    // solved source by -d exactly (to first order), so the term is sigma^2 I.
    const POSITION_STD_M: f64 = 0.3;
    let sensors: Vec<[f64; 3]> = TETRA_OFFSETS_M.to_vec();
    let rows: Vec<WeightedTdoaRow> = (1..sensors.len())
        .map(|sensor_a| WeightedTdoaRow {
            sensor_a,
            sensor_b: 0,
            weight: 1.0,
        })
        .collect();
    let covariance = node_position_error_covariance_m2(
        [0.4, 0.7, 0.3],
        &sensors,
        &rows,
        [[0.0; 3]; 3],
        SOUND_SPEED_MPS,
        SensorNodePositionStd {
            node_index_by_sensor: &[0, 0, 0, 0],
            node_position_std_m: &[POSITION_STD_M],
        },
    )
    .expect("a stated node error has a term");
    for row in 0..3 {
        for column in 0..3 {
            let expected = if row == column {
                POSITION_STD_M * POSITION_STD_M
            } else {
                0.0
            };
            assert!(
                (covariance[row][column] - expected).abs() < 1.0e-6,
                "[{row}][{column}] = {}",
                covariance[row][column]
            );
        }
    }
}

#[test]
fn unstated_or_unrelated_nodes_add_nothing() {
    let sensors = [
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ];
    let rows: Vec<WeightedTdoaRow> = (1..4)
        .map(|sensor_a| WeightedTdoaRow {
            sensor_a,
            sensor_b: 0,
            weight: 1.0,
        })
        .collect();
    let covariance_with = |node_position_std_m: &[f64]| {
        node_position_error_covariance_m2(
            [3.0, 4.0, 5.0],
            &sensors,
            &rows,
            [[0.0; 3]; 3],
            SOUND_SPEED_MPS,
            SensorNodePositionStd {
                node_index_by_sensor: &[0, 1, 2, 3],
                node_position_std_m,
            },
        )
    };
    assert!(covariance_with(&[0.0, 0.0, 0.0, 0.0]).is_none());
    // A fifth node with a stated error but no sensor in any row.
    assert!(covariance_with(&[0.0, 0.0, 0.0, 0.0, 0.5]).is_none());
    assert!(covariance_with(&[0.0, 0.2, 0.0, 0.0]).is_some());
}
