package com.minimappr.field.android

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import androidx.core.app.NotificationCompat

private const val CAPTURE_NOTIFICATION_CHANNEL_ID = "field-capture"
private const val OPEN_APP_REQUEST_CODE = 0
private const val STOP_CAPTURE_REQUEST_CODE = 1

/** Persistent notification the microphone foreground service must show. */
internal object FieldCaptureNotification {
    const val NOTIFICATION_ID = 4801

    fun ensureChannel(context: Context) {
        val channel = NotificationChannel(
            CAPTURE_NOTIFICATION_CHANNEL_ID,
            context.getString(R.string.field_capture_notification_channel_name),
            // Low: persistent and visible, but silent (we are recording audio).
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = context.getString(R.string.field_capture_notification_channel_description)
        }
        context.getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    fun build(context: Context): Notification {
        val openAppIntent = PendingIntent.getActivity(
            context,
            OPEN_APP_REQUEST_CODE,
            Intent(context, FieldMainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val stopCaptureIntent = PendingIntent.getService(
            context,
            STOP_CAPTURE_REQUEST_CODE,
            FieldCaptureForegroundService.stopIntent(context),
            PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(context, CAPTURE_NOTIFICATION_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle(context.getString(R.string.field_capture_notification_title))
            .setContentText(context.getString(R.string.field_capture_notification_text))
            .setContentIntent(openAppIntent)
            .addAction(0, context.getString(R.string.field_capture_notification_stop_action), stopCaptureIntent)
            .setOngoing(true)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()
    }
}
