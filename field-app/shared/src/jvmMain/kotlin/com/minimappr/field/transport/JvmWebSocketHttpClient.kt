package com.minimappr.field.transport

import com.minimappr.field.FieldSessionConfiguration
import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.websocket.WebSockets

/** Host-JVM joiner client (tests): CIO engine. */
fun createJvmWebSocketHttpClient(configuration: FieldSessionConfiguration): HttpClient = HttpClient(CIO) {
    install(WebSockets) { pingIntervalMillis = configuration.webSocketPingIntervalMilliseconds }
}
