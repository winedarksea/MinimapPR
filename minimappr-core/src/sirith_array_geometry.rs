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
