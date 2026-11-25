package com.minimappr.field.ui

import com.minimappr.field.chirplab.ChirpLabRole
import com.minimappr.field.session.DiscoveredFieldHost

/** Everything the shared UI can ask the platform shell to do. */
class FieldAppActions(
    val onStartCapture: () -> Unit = {},
    val onStopService: () -> Unit = {},
    val onStartChirpLab: (ChirpLabRole) -> Unit = {},
    val onHostSession: () -> Unit = {},
    val onJoinDiscoveredHost: (DiscoveredFieldHost) -> Unit = {},
    val onJoinTypedAddress: (String) -> Unit = {},
)
