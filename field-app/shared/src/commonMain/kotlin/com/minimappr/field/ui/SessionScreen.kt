package com.minimappr.field.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import com.minimappr.field.node.FieldCaptureServiceState
import com.minimappr.field.node.FieldServiceMode
import com.minimappr.field.node.FieldServiceStatus
import com.minimappr.field.session.DiscoveredFieldHost

private const val SESSION_ID_PREVIEW_HEX_DIGITS = 8

/**
 * Stage B: host a session, join a host found by NSD, or join a typed
 * IP:port (explicit choice for networks where mDNS is filtered).
 */
@Composable
fun SessionScreen(
    serviceStatus: FieldServiceStatus,
    discoveredHosts: List<DiscoveredFieldHost>,
    actions: FieldAppActions,
) {
    val running = serviceStatus.serviceState == FieldCaptureServiceState.RUNNING
    var typedAddress by rememberSaveable { mutableStateOf("") }
    Row(horizontalArrangement = Arrangement.spacedBy(ROW_SPACING)) {
        Button(onClick = actions.onHostSession, enabled = !running) { Text("Host") }
        OutlinedButton(onClick = actions.onStopService) { Text("Stop") }
    }
    if (!running) {
        Text("Discovered hosts (${discoveredHosts.size}):")
        discoveredHosts.forEach { host ->
            OutlinedButton(onClick = { actions.onJoinDiscoveredHost(host) }) {
                Text("Join ${host.serviceName} @ ${host.hostAddress}:${host.port} sid ${host.sessionIdHex?.take(SESSION_ID_PREVIEW_HEX_DIGITS) ?: "?"}")
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(ROW_SPACING)) {
            OutlinedTextField(value = typedAddress, onValueChange = { typedAddress = it }, label = { Text("Host IP:port") }, singleLine = true)
        }
        Button(onClick = { actions.onJoinTypedAddress(typedAddress) }, enabled = typedAddress.isNotBlank()) { Text("Join typed address") }
    }
    serviceStatus.session?.takeIf { serviceStatus.mode == FieldServiceMode.SESSION }?.let { SyncStatusPanel(it) }
}
