package com.minimappr.field.transport

import com.minimappr.field.FieldSessionConfiguration
import io.ktor.client.HttpClient
import io.ktor.client.engine.okhttp.OkHttp
import io.ktor.client.plugins.websocket.WebSockets

/**
 * Android joiner client: OkHttp engine, because OkHttp accepts a
 * `Network.socketFactory` (needed later to pin traffic to a
 * LocalOnlyHotspot network that has no internet route).
 */
fun createAndroidWebSocketHttpClient(configuration: FieldSessionConfiguration): HttpClient = HttpClient(OkHttp) {
    install(WebSockets) { pingIntervalMillis = configuration.webSocketPingIntervalMilliseconds }
}
