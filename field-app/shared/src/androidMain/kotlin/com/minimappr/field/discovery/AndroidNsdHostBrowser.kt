package com.minimappr.field.discovery

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.os.Build
import android.util.Log
import com.minimappr.field.FieldSessionConfiguration
import com.minimappr.field.session.DiscoveredFieldHost
import java.net.Inet4Address
import java.net.InetAddress
import java.util.concurrent.ConcurrentHashMap

private const val LOG_TAG = "FieldNsdBrowser"
private const val API_LEVEL_SERVICE_INFO_CALLBACK = 34

/**
 * Joiner: browses `_minimappr-field._tcp` and resolves each host.
 *
 * Resolution: API 34+ uses `registerServiceInfoCallback` (live updates, no
 * one-at-a-time limit); older releases (the A03s, API 33) use
 * `resolveService`, which fails with FAILURE_ALREADY_ACTIVE if two resolves
 * overlap, so those are queued and run one by one.
 */
class AndroidNsdHostBrowser(
    context: Context,
    private val configuration: FieldSessionConfiguration,
    private val onHostsChanged: (List<DiscoveredFieldHost>) -> Unit,
) {
    private val nsdManager = context.getSystemService(Context.NSD_SERVICE) as NsdManager
    private val mainExecutor = context.mainExecutor
    private val hostsByServiceName = ConcurrentHashMap<String, DiscoveredFieldHost>()
    private val serviceInfoCallbacks = ConcurrentHashMap<String, NsdManager.ServiceInfoCallback>()
    private val pendingLegacyResolves = ArrayDeque<NsdServiceInfo>()
    private var legacyResolveInFlight = false
    private var discoveryListener: NsdManager.DiscoveryListener? = null

    fun start() {
        if (discoveryListener != null) return
        val listener = object : NsdManager.DiscoveryListener {
            override fun onDiscoveryStarted(serviceType: String) = Log.i(LOG_TAG, "browsing $serviceType").let {}
            override fun onDiscoveryStopped(serviceType: String) = Unit
            override fun onStartDiscoveryFailed(serviceType: String, errorCode: Int) = Log.e(LOG_TAG, "discovery failed: $errorCode").let {}
            override fun onStopDiscoveryFailed(serviceType: String, errorCode: Int) = Unit
            override fun onServiceFound(serviceInfo: NsdServiceInfo) = resolve(serviceInfo)
            override fun onServiceLost(serviceInfo: NsdServiceInfo) {
                serviceInfoCallbacks.remove(serviceInfo.serviceName)?.let { runCatching { nsdManager.unregisterServiceInfoCallback(it) } }
                if (hostsByServiceName.remove(serviceInfo.serviceName) != null) publish()
            }
        }
        discoveryListener = listener
        nsdManager.discoverServices(configuration.nsdServiceType, NsdManager.PROTOCOL_DNS_SD, listener)
    }

    private fun resolve(serviceInfo: NsdServiceInfo) {
        if (Build.VERSION.SDK_INT >= API_LEVEL_SERVICE_INFO_CALLBACK) resolveWithServiceInfoCallback(serviceInfo) else enqueueLegacyResolve(serviceInfo)
    }

    @androidx.annotation.RequiresApi(API_LEVEL_SERVICE_INFO_CALLBACK)
    private fun resolveWithServiceInfoCallback(serviceInfo: NsdServiceInfo) {
        if (serviceInfoCallbacks.containsKey(serviceInfo.serviceName)) return
        val callback = object : NsdManager.ServiceInfoCallback {
            override fun onServiceInfoCallbackRegistrationFailed(errorCode: Int) = Log.w(LOG_TAG, "info callback failed: $errorCode").let {}
            override fun onServiceUpdated(updated: NsdServiceInfo) = recordResolved(updated, updated.hostAddresses)
            override fun onServiceLost() {
                if (hostsByServiceName.remove(serviceInfo.serviceName) != null) publish()
            }
            override fun onServiceInfoCallbackUnregistered() = Unit
        }
        serviceInfoCallbacks[serviceInfo.serviceName] = callback
        nsdManager.registerServiceInfoCallback(serviceInfo, mainExecutor, callback)
    }

    private fun enqueueLegacyResolve(serviceInfo: NsdServiceInfo) = synchronized(pendingLegacyResolves) {
        pendingLegacyResolves.addLast(serviceInfo)
        if (!legacyResolveInFlight) startNextLegacyResolve()
    }

    /** Caller holds the queue lock. */
    @Suppress("DEPRECATION") // resolveService is the only resolver before API 34.
    private fun startNextLegacyResolve() {
        val next = pendingLegacyResolves.removeFirstOrNull() ?: run {
            legacyResolveInFlight = false
            return
        }
        legacyResolveInFlight = true
        nsdManager.resolveService(next, object : NsdManager.ResolveListener {
            override fun onResolveFailed(serviceInfo: NsdServiceInfo, errorCode: Int) {
                Log.w(LOG_TAG, "resolve ${serviceInfo.serviceName} failed: $errorCode")
                synchronized(pendingLegacyResolves) { startNextLegacyResolve() }
            }

            override fun onServiceResolved(serviceInfo: NsdServiceInfo) {
                recordResolved(serviceInfo, listOfNotNull(serviceInfo.host))
                synchronized(pendingLegacyResolves) { startNextLegacyResolve() }
            }
        })
    }

    private fun recordResolved(serviceInfo: NsdServiceInfo, addresses: List<InetAddress>) {
        // The LAN session is IPv4 in practice; link-local IPv6 needs a scope id.
        val address = addresses.firstOrNull { it is Inet4Address } ?: addresses.firstOrNull() ?: return
        val attributes = serviceInfo.attributes
        fun text(key: String) = attributes[key]?.let { String(it, Charsets.UTF_8) }
        hostsByServiceName[serviceInfo.serviceName] = DiscoveredFieldHost(
            serviceName = serviceInfo.serviceName,
            hostAddress = address.hostAddress ?: return,
            port = serviceInfo.port,
            sessionIdHex = text(FieldNsdTxtKeys.SESSION_ID),
            hostNodeId = text(FieldNsdTxtKeys.HOST_NODE_ID)?.toUIntOrNull(),
            protocolVersion = text(FieldNsdTxtKeys.PROTOCOL_VERSION),
        )
        publish()
    }

    private fun publish() = onHostsChanged(hostsByServiceName.values.sortedBy { it.serviceName })

    fun stop() {
        discoveryListener?.let { runCatching { nsdManager.stopServiceDiscovery(it) } }
        discoveryListener = null
        serviceInfoCallbacks.values.forEach { runCatching { nsdManager.unregisterServiceInfoCallback(it) } }
        serviceInfoCallbacks.clear()
        hostsByServiceName.clear()
    }
}
