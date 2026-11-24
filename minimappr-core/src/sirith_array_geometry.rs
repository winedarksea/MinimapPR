//! Fixed mic geometry of the Sirith tetrahedral node. Field Mode nodes carry
//! their own geometry in a node descriptor; this constant stays the reference
//! layout for the tetra render/SRP paths and their parity fixtures.

/// Centroid-relative Sirith tetrahedral mic positions [MK1, MK2, MK3, MK4].
pub const SIRITH_MIC_POSITIONS_M: [[f32; 3]; 4] = [
    [0.0, 0.050, 0.0],
    [0.0433, 0.025, 0.0],
    [0.0, 0.0, 0.0],
    [0.02165, 0.025, 0.04082],
];

/// Corner radius of the Sirith planar array (firmware `kPlanarArrayRadiusM`).
pub const SIRITH_PLANAR_CORNER_RADIUS_M: f32 = 0.025;

/// Sirith planar 5-mic body-frame offsets, mirroring firmware
/// `kPlanarSensorOffsetsM` (`firmware/nodes/sirith_planar/include/node_config.h`):
/// ch0..ch3 = corners NE, NW, SW, SE on a 25 mm circle, ch4 = centre; all z = 0.
pub const SIRITH_PLANAR_MIC_OFFSETS_M: [[f32; 3]; 5] = {
    let corner_xy_m = SIRITH_PLANAR_CORNER_RADIUS_M * std::f32::consts::FRAC_1_SQRT_2;
    [
        [corner_xy_m, corner_xy_m, 0.0],
        [-corner_xy_m, corner_xy_m, 0.0],
        [-corner_xy_m, -corner_xy_m, 0.0],
        [corner_xy_m, -corner_xy_m, 0.0],
        [0.0, 0.0, 0.0],
    ]
};
