//! Speaker channel map (plan A1): which stereo channel drives the bottom
//! speaker, and whether rotating the phone swaps it. One run plays one
//! channel in one pose; the listener's remote-chirp SNR per (model,
//! channel, orientation) is the evidence, tabulated across runs by
//! `summarize`.

use crate::bundle_schema::BundleEvent;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum PhoneOrientation {
    Portrait,
    PortraitUpsideDown,
    LandscapeLeft,
    LandscapeRight,
    FlatFaceUp,
    FlatFaceDown,
    Unknown,
}

/// Gravity in the device frame from a rotation-vector quaternion
/// (device → world, world z up): g_device = Rᵀ · [0, 0, 1].
fn gravity_in_device_frame(quaternion: [f64; 4]) -> [f64; 3] {
    let [w, x, y, z] = quaternion;
    // Third row of R (world z axis expressed in device coordinates).
    [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]
}

pub fn orientation_from_quaternion(quaternion: [f64; 4]) -> PhoneOrientation {
    let norm = quaternion.iter().map(|c| c * c).sum::<f64>().sqrt();
    if norm < 1e-6 {
        return PhoneOrientation::Unknown;
    }
    let [gx, gy, gz] = gravity_in_device_frame(quaternion.map(|c| c / norm));
    let (ax, ay, az) = (gx.abs(), gy.abs(), gz.abs());
    if az >= ax && az >= ay {
        if gz > 0.0 { PhoneOrientation::FlatFaceUp } else { PhoneOrientation::FlatFaceDown }
    } else if ay >= ax {
        if gy > 0.0 { PhoneOrientation::Portrait } else { PhoneOrientation::PortraitUpsideDown }
    } else if gx > 0.0 {
        PhoneOrientation::LandscapeLeft
    } else {
        PhoneOrientation::LandscapeRight
    }
}

/// Most frequent orientation over the run's attitude samples.
pub fn dominant_orientation(events: &[BundleEvent]) -> PhoneOrientation {
    let mut counts: Vec<(PhoneOrientation, usize)> = Vec::new();
    for event in events {
        if let BundleEvent::Attitude { quaternion, .. } = event {
            let orientation = orientation_from_quaternion(*quaternion);
            match counts.iter_mut().find(|(o, _)| *o == orientation) {
                Some((_, count)) => *count += 1,
                None => counts.push((orientation, 1)),
            }
        }
    }
    counts.into_iter().max_by_key(|(_, count)| *count).map_or(PhoneOrientation::Unknown, |(o, _)| o)
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ChannelMapObservation {
    pub emitter_platform: String,
    pub emitter_channel: String,
    pub emitter_orientation: PhoneOrientation,
    pub listener_platform: String,
    pub median_remote_snr_db: Option<f64>,
    pub remote_detected: usize,
    pub remote_expected: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_quaternion_is_flat_face_up_and_pitched_up_is_portrait() {
        assert_eq!(orientation_from_quaternion([1.0, 0.0, 0.0, 0.0]), PhoneOrientation::FlatFaceUp);
        // Rotate +90° about device x: the top edge points up (held upright).
        let half = std::f64::consts::FRAC_PI_4;
        assert_eq!(orientation_from_quaternion([half.cos(), half.sin(), 0.0, 0.0]), PhoneOrientation::Portrait);
    }
}
