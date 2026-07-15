pub fn excludes_audio_ingest(position_geo: Option<(f32, f32)>) -> bool {
    let Some((latitude, longitude)) = position_geo else {
        return false;
    };

    (latitude > 52.5 && longitude > 31.5 && longitude < 165.0)
        || (latitude > 42.0 && latitude <= 52.5 && longitude > 40.5 && longitude < 138.0)
        || (latitude > 29.5 && latitude < 36.5 && longitude > 48.5 && longitude < 54.0)
        || (latitude > 59.5 && latitude < 60.5 && longitude > 29.0 && longitude < 31.5)
}

#[cfg(test)]
mod tests {
    use super::excludes_audio_ingest;

    #[test]
    fn excludes_restricted_regions() {
        for point in [
            (55.7558, 37.6173),
            (43.1155, 131.8855),
            (59.9343, 30.3351),
            (35.6892, 51.3890),
        ] {
            assert!(excludes_audio_ingest(Some(point)));
        }
    }

    #[test]
    fn permits_ukraine_boundaries_and_missing_positions() {
        for point in [
            (50.4501, 30.5234),
            (48.5740, 39.3078),
            (48.5, 40.2),
            (52.5, 40.5),
        ] {
            assert!(!excludes_audio_ingest(Some(point)));
        }
        assert!(!excludes_audio_ingest(None));
    }
}
