use super::schema::DeviceRecord;

pub fn load_devices() -> Vec<DeviceRecord> {
    crate::prefs::get_json(crate::prefs::KEY_DEVICES).unwrap_or_default()
}

pub fn save_devices(devices: &[DeviceRecord]) {
    crate::prefs::set_json(crate::prefs::KEY_DEVICES, &devices);
}
