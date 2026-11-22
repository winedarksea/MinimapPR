//! Apparent source positions of reflections for ground-truth echo events.
//! A trunk is a point scatterer: it re-radiates from its own position (delayed
//! by r₁/c), so every receiver — compact array or distributed mesh — sees the
//! echo coming from the trunk itself. The ground is a mirror: its image source
//! (z' = 2h − z) is consistent for all receivers. A localizer that promotes a
//! reflection to its own source puts it here, so scoring can tell an echo
//! promotion from an unrelated false positive.

use crate::physics::geometry::{add, distance, norm, scale, sub, Vec3};
use crate::world::WorldPlan;

/// Below this the reflector sits on the centroid and has no direction.
const MINIMUM_DIRECTION_LENGTH_M: f64 = 1e-9;

/// One reflection of a source, as observed from the array centroid.
#[derive(Clone, Debug, PartialEq)]
pub struct EchoImage {
    /// `reflector_id` of the trunk, or `"ground"`.
    pub reflector_id: String,
    /// `"trunk"` or `"ground"` (bundle v2 recommended `reflector.kind`).
    pub kind: &'static str,
    /// Reflecting point (trunk position, or the specular point on the ground).
    pub reflecting_point_m: Vec3,
    /// Apparent source position (what a localizer would report): the trunk
    /// itself, or the ground image source.
    pub image_position_m: Vec3,
    /// Reflected minus direct path delay at the centroid.
    pub extra_delay_s: f64,
    /// Level below the direct arrival, positive dB.
    pub attenuation_db: f64,
}

fn attenuation_db(direct_length_m: f64, echo_length_m: f64, coefficient: f64) -> f64 {
    if coefficient <= 0.0 {
        return f64::INFINITY;
    }
    20.0 * (echo_length_m / (coefficient * direct_length_m.max(MINIMUM_DIRECTION_LENGTH_M))).log10()
}

/// Every reflection (trunks, then the ground plane) of a source at `source_m`
/// heard at `centroid_m`. Mirrors the renderer's path model in
/// `render::propagation::PathKind` (point scatterer: r₁ + r₂; ground: image in z = h).
pub fn echo_images(plan: &WorldPlan, source_m: Vec3, centroid_m: Vec3) -> Vec<EchoImage> {
    let direct_length_m = distance(source_m, centroid_m);
    let mut images: Vec<EchoImage> = plan
        .reflectors
        .iter()
        .filter_map(|reflector| {
            let toward_reflector = sub(reflector.position_m, centroid_m);
            let reflector_range_m = norm(toward_reflector);
            if reflector_range_m < MINIMUM_DIRECTION_LENGTH_M {
                return None;
            }
            let echo_length_m = distance(source_m, reflector.position_m) + reflector_range_m;
            Some(EchoImage {
                reflector_id: reflector.reflector_id.clone(),
                kind: "trunk",
                reflecting_point_m: reflector.position_m,
                image_position_m: reflector.position_m,
                extra_delay_s: (echo_length_m - direct_length_m) / plan.speed_of_sound_mps,
                attenuation_db: attenuation_db(
                    direct_length_m,
                    echo_length_m,
                    reflector.reflection_coefficient,
                ),
            })
        })
        .collect();
    if let Some((height_m, coefficient)) = plan.ground {
        let image = [source_m[0], source_m[1], 2.0 * height_m - source_m[2]];
        let echo_length_m = distance(image, centroid_m);
        // Specular point: where the image→centroid line crosses z = h.
        let span_z = centroid_m[2] - image[2];
        let fraction = if span_z.abs() > MINIMUM_DIRECTION_LENGTH_M {
            ((height_m - image[2]) / span_z).clamp(0.0, 1.0)
        } else {
            0.5
        };
        images.push(EchoImage {
            reflector_id: "ground".into(),
            kind: "ground",
            reflecting_point_m: add(image, scale(sub(centroid_m, image), fraction)),
            image_position_m: image,
            extra_delay_s: (echo_length_m - direct_length_m) / plan.speed_of_sound_mps,
            attenuation_db: attenuation_db(direct_length_m, echo_length_m, coefficient),
        });
    }
    images
}

/// Mean reported position of all devices (the scoring `array_centroid`:
/// simulated bundles have no reference nodes).
pub fn array_centroid_m(plan: &WorldPlan) -> Vec3 {
    let count = plan.devices.len().max(1) as f64;
    let sum = plan
        .devices
        .iter()
        .fold([0.0; 3], |acc, device| add(acc, device.reported_position_m));
    scale(sum, 1.0 / count)
}
