"""Home Assistant MQTT bridge (outbound publish only).

Optional subsystem, fully dormant unless ``hass_enabled`` is set and an MQTT
host is configured. Mirrors ``core/effectors`` in shape: a transport Protocol,
a lazily-imported concrete driver, and a module-level factory as the test seam.
"""
