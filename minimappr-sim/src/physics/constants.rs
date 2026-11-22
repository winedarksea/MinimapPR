//! Named physical constants. Scenario-tunable values live in `scenario::*`
//! config structs; only laws-of-nature and standard reference values go here.

/// Standard atmosphere, ISO 9613-1 reference ambient pressure p_r.
pub const STANDARD_ATMOSPHERE_PRESSURE_KPA: f64 = 101.325;

/// ISO 9613-1 reference air temperature T_0 (20 °C).
pub const ISO9613_REFERENCE_TEMPERATURE_K: f64 = 293.15;

/// ISO 9613-1 triple-point isotherm temperature T_01 used in the
/// saturation vapour pressure formula.
pub const ISO9613_TRIPLE_POINT_ISOTHERM_K: f64 = 273.16;

pub const CELSIUS_TO_KELVIN_OFFSET: f64 = 273.15;

/// 0 dB SPL reference pressure (20 µPa).
pub const REFERENCE_SOUND_PRESSURE_PA: f64 = 20e-6;

/// Linear speed-of-sound model. Mirrors `speed_of_sound_mps` in
/// `minimappr/core/localization.py` so simulated propagation uses exactly the
/// speed the server localizer assumes (otherwise every TDOA carries a bias
/// that the simulator, not the system under test, introduced).
pub const SPEED_OF_SOUND_AT_ZERO_CELSIUS_MPS: f64 = 331.3;
pub const SPEED_OF_SOUND_SLOPE_PER_DEGREE_CELSIUS_MPS: f64 = 0.606;
pub const SPEED_OF_SOUND_SLOPE_PER_HUMIDITY_PERCENT_MPS: f64 = 0.0124;

/// Structural (body-conduction) sound speed through a phone chassis; the
/// self-chirp precursor arrives ahead of the airborne path at this speed.
pub const SOUND_SPEED_IN_PHONE_BODY_MPS: f64 = 5000.0;

pub const NANOSECONDS_PER_SECOND: f64 = 1e9;
pub const PARTS_PER_MILLION: f64 = 1e6;
pub const PERCENT_PER_FRACTION: f64 = 100.0;

/// Speed of sound at the given temperature/humidity (see constants above).
pub fn speed_of_sound_mps(temperature_c: f64, humidity_fraction: f64) -> f64 {
    let humidity_percent = humidity_fraction.clamp(0.0, 1.0) * PERCENT_PER_FRACTION;
    SPEED_OF_SOUND_AT_ZERO_CELSIUS_MPS
        + SPEED_OF_SOUND_SLOPE_PER_DEGREE_CELSIUS_MPS * temperature_c
        + SPEED_OF_SOUND_SLOPE_PER_HUMIDITY_PERCENT_MPS * humidity_percent
}

/// Root-mean-square pressure (Pa) for a sound pressure level in dB SPL.
pub fn pascals_from_db_spl(level_db_spl: f64) -> f64 {
    REFERENCE_SOUND_PRESSURE_PA * 10f64.powf(level_db_spl / 20.0)
}

/// Linear amplitude ratio for a level in decibels.
pub fn amplitude_ratio_from_db(level_db: f64) -> f64 {
    10f64.powf(level_db / 20.0)
}
