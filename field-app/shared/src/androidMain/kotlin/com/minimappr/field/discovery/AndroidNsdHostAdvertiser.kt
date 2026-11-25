package com.minimappr.field.discovery

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.util.Log
import com.minimappr.field.FieldSessionConfiguration

private const val LOG_TAG = "FieldNsdAdvertiser"

/** NSD TXT keys (joiner contract): protocol version, host node id, full session id. */
object FieldNsdTxtKeys {
    const val PROTOCOL_VERSION = "v"
    const val HOST_NODE_ID = "hid"
    /** 32 hex digits: the full 16-byte session id, so the joiner can insist on it. */
    const val SESSION_ID = "sid"
}

/**
 * Host: advertises `_minimappr-field._tcp` on the LAN with TXT
 * {v, hid, sid}. NSD may rename the service on a collision; the browser
 * keys on the TXT record, not the name.
 */
class AndroidNsdHostAdvertiser(context: Context, private val configuration: FieldSessionConfiguration) {
    private val nsdManager = context.getSystemService(Context.NSD_SERVICE) as NsdManager
    private var registrationListener: NsdManager.RegistrationListener? = null

    @Volatile var registeredServiceName: String? = null
        private set

    fun register(serviceName: String, port: Int, hostNodeId: UInt, sessionIdHex: String) {
        if (registrationListener != null) return
        val serviceInfo = NsdServiceInfo().apply {
            this.serviceName = serviceName
            serviceType = configuration.nsdServiceType
            this.port = port
            setAttribute(FieldNsdTxtKeys.PROTOCOL_VERSION, configuration.nsdTxtProtocolVersion)
            setAttribute(FieldNsdTxtKeys.HOST_NODE_ID, hostNodeId.toString())
            setAttribute(FieldNsdTxtKeys.SESSION_ID, sessionIdHex)
        }
        val listener = object : NsdManager.RegistrationListener {
            override fun onServiceRegistered(info: NsdServiceInfo) {
                registeredServiceName = info.serviceName
                Log.i(LOG_TAG, "registered ${info.serviceName} on port $port")
            }

            override fun onRegistrationFailed(info: NsdServiceInfo, errorCode: Int) =
                Log.e(LOG_TAG, "NSD registration failed: $errorCode").let {}

            override fun onServiceUnregistered(info: NsdServiceInfo) = Log.i(LOG_TAG, "unregistered ${info.serviceName}").let {}

            override fun onUnregistrationFailed(info: NsdServiceInfo, errorCode: Int) =
                Log.w(LOG_TAG, "NSD unregistration failed: $errorCode").let {}
        }
        registrationListener = listener
        nsdManager.registerService(serviceInfo, NsdManager.PROTOCOL_DNS_SD, listener)
    }

    fun unregister() {
        val listener = registrationListener ?: return
        registrationListener = null
        runCatching { nsdManager.unregisterService(listener) }
    }
}
