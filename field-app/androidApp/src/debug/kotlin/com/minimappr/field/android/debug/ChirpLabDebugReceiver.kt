package com.minimappr.field.android.debug

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.minimappr.field.android.FieldCaptureForegroundService

private const val LOG_TAG = "ChirpLabDebugReceiver"
private const val ACTION_CHIRP_LAB_STOP = "com.minimappr.field.CHIRP_LAB_STOP"
private const val ACTION_CHIRP_LAB_MARK = "com.minimappr.field.CHIRP_LAB_MARK"
private const val ACTION_SESSION_STOP = "com.minimappr.field.SESSION_STOP"
private const val EXTRA_LABEL = "label"

/**
 * Debug-only control from the laptop (`am broadcast -n .../.debug.ChirpLabDebugReceiver`):
 * stop a chirp lab or session run, or drop a `marker` event (foam / hand /
 * table / free text) into the running bundle's events.jsonl. Same process
 * as the service, so MARK goes straight to the live instance.
 */
class ChirpLabDebugReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            ACTION_CHIRP_LAB_STOP, ACTION_SESSION_STOP -> {
                Log.i(LOG_TAG, "stop requested (${intent.action})")
                FieldCaptureForegroundService.stop(context)
            }
            ACTION_CHIRP_LAB_MARK -> {
                val label = intent.getStringExtra(EXTRA_LABEL) ?: "unlabelled"
                val service = FieldCaptureForegroundService.activeInstance
                if (service == null) Log.w(LOG_TAG, "marker '$label' dropped: no run active") else service.markRun(label)
            }
        }
    }
}
