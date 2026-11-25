package com.minimappr.field.session

import com.minimappr.field.FieldSessionConfiguration
import com.minimappr.field.ffi.FieldSyncStatus
import com.minimappr.field.ffi.defaultFieldMeshTuning
import com.minimappr.field.transport.KtorHostWebSocketServer
import com.minimappr.field.transport.KtorJoinerWebSocketClient
import com.minimappr.field.transport.createJvmWebSocketHttpClient
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Two FieldSessionControllers (host + joiner) over a REAL Ktor WebSocket on
 * 127.0.0.1, with synthetic audio rendered from the chirps each device asks
 * to play. Runs in real time (~15–25 s): join, session-clock lock, and at
 * least two completed chirp rounds with a range to the joiner.
 */
class KtorLoopbackSessionTest {
    private val loopbackAddress = "127.0.0.1"
    private val distanceM = 3.0
    /** The FFI's default d_ii for an unknown platform (`IslandSyncConfig`). */
    private val selfPathM = 0.02
    private val captureStepMilliseconds = 10L
    private val sessionTimeoutMilliseconds = 60_000L
    private val requiredRoundsCompleted = 2u
    private val rangeToleranceM = 0.05
    private val secondNs = 1_000_000_000L

    @Test
    fun hostAndJoinerJoinAlignClocksAndCompleteRoundsOverAWebSocket() = runBlocking {
        val configuration = FieldSessionConfiguration(hostListenPort = 0)
        val tuning = defaultFieldMeshTuning().copy(periodicRoundIntervalMs = 3_000uL)
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val testStartNs = System.nanoTime()
        val trueNowNs = { System.nanoTime() - testStartNs }
        // Unrelated monotonic epochs: host booted ~3 h ago, joiner ~41 min.
        val phones = listOf(
            SyntheticPhone(monotonicOffsetNs = 10_800 * secondNs, captureStartTrueNs = 13_000_000),
            SyntheticPhone(monotonicOffsetNs = 2_460 * secondNs + 123_456_789, captureStartTrueNs = 370_000_000),
        )
        val world = SyntheticAcousticPairWorld(phones, distanceM, selfPathM, tuning.speedOfSound(), trueNowNs)
        val host = FieldSessionIdentity.newHost()
        val joiner = FieldSessionIdentity.newJoiner(expectedSessionId = null)
        val server = KtorHostWebSocketServer(host.nodeId, configuration, bindAddress = loopbackAddress, requestedPort = 0)
        val hostController = FieldSessionController(
            host.deviceSetup("synthetic", "test", rollingBufferSeconds = 20), tuning, configuration, server,
            world.playbackPort(0), { phones[0].localNs(trueNowNs()) },
        )
        phones[0].controller = hostController
        hostController.start(scope)
        val client = KtorJoinerWebSocketClient(loopbackAddress, server.boundPort, configuration, { createJvmWebSocketHttpClient(configuration) }, scope)
        val joinerController = FieldSessionController(
            joiner.deviceSetup("synthetic", "test", rollingBufferSeconds = 20), tuning, configuration, client,
            world.playbackPort(1), { phones[1].localNs(trueNowNs()) },
        )
        phones[1].controller = joinerController
        joinerController.start(scope)
        scope.launch {
            while (isActive) {
                world.captureUpToNow()
                delay(captureStepMilliseconds)
            }
        }

        var hostStatus: FieldSyncStatus = hostController.device.syncStatus()
        var joinerStatus: FieldSyncStatus = joinerController.device.syncStatus()
        val deadline = System.currentTimeMillis() + sessionTimeoutMilliseconds
        while (System.currentTimeMillis() < deadline) {
            hostStatus = hostController.device.syncStatus()
            joinerStatus = joinerController.device.syncStatus()
            val ranged = hostStatus.peers.any { it.nodeId == joiner.nodeId && it.rangeM != null }
            if (joinerStatus.joined && ranged && hostStatus.roundsCompleted >= requiredRoundsCompleted) break
            delay(captureStepMilliseconds * 10)
        }
        try {
            assertTrue(joinerStatus.sessionClockAligned, "joiner never locked the session clock: $joinerStatus")
            assertTrue(joinerStatus.joined, "joiner never joined: $joinerStatus")
            assertEquals(host.sessionIdHex, joinerStatus.sessionIdHex, "typed-IP joiner adopts the host's session id")
            val trueOffsetMs = (phones[0].monotonicOffsetNs - phones[1].monotonicOffsetNs) / 1e6
            val measuredOffsetMs = assertNotNull(joinerStatus.sessionClockOffsetMs)
            assertTrue(abs(measuredOffsetMs - trueOffsetMs) < 5.0, "session offset $measuredOffsetMs ms vs $trueOffsetMs ms")
            assertTrue(hostStatus.roundsCompleted >= requiredRoundsCompleted, "only ${hostStatus.roundsCompleted} rounds: $hostStatus")
            assertTrue(hostStatus.mutedNodeIds.isEmpty(), "$hostStatus")
            val range = assertNotNull(hostStatus.peers.first { it.nodeId == joiner.nodeId }.rangeM)
            assertTrue(abs(range - distanceM) < rangeToleranceM, "range $range m vs $distanceM m: $hostStatus")
            assertTrue(hostStatus.toSnakeCaseJson().contains("\"rounds_completed\":${hostStatus.roundsCompleted}"))
            println("loopback session: rounds=${hostStatus.roundsCompleted} range=$range m offset=$measuredOffsetMs ms " +
                "rtt=${joinerStatus.sessionClockBestRoundTripMs} ms")
        } finally {
            joinerController.close()
            hostController.close()
            scope.cancel()
        }
    }
}

private fun com.minimappr.field.ffi.FieldMeshTuning.speedOfSound(): Double =
    com.minimappr.field.ffi.speedOfSoundForTemperature(airTemperatureC)
