package org.rosettajane.app;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

/**
 * Keeps the p2p service running while the app is in the background. Android requires a
 * visible notification for that, which also means the owner can always see it is on.
 */
public class PeerService extends Service {
    static final String CHANNEL = "p2p";
    static final String ACTION_STOP = "org.rosettajane.app.STOP";
    private PowerManager.WakeLock wakeLock;

    static PyObject bridge(Context context) {
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(context.getApplicationContext()));
        }
        return Python.getInstance().getModule("rj_app");
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null || ACTION_STOP.equals(intent.getAction())) {
            shutDown();
            return START_NOT_STICKY;
        }
        NotificationManager manager = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        if (Build.VERSION.SDK_INT >= 26) {
            manager.createNotificationChannel(
                    new NotificationChannel(CHANNEL, "P2P service", NotificationManager.IMPORTANCE_LOW));
        }
        PendingIntent open = PendingIntent.getActivity(this, 0, new Intent(this, MainActivity.class),
                PendingIntent.FLAG_IMMUTABLE);
        PendingIntent stop = PendingIntent.getService(this, 1,
                new Intent(this, PeerService.class).setAction(ACTION_STOP), PendingIntent.FLAG_IMMUTABLE);
        Notification.Builder builder = Build.VERSION.SDK_INT >= 26
                ? new Notification.Builder(this, CHANNEL) : new Notification.Builder(this);
        Notification notification = builder
                .setContentTitle("Rosetta-Jane p2p service is on")
                .setContentText("This device is computing for your group. Tap to open.")
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .setContentIntent(open)
                .addAction(new Notification.Action.Builder(null, "Stop", stop).build())
                .setOngoing(true)
                .build();
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(1, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        } else {
            startForeground(1, notification);
        }
        if (wakeLock == null) {
            wakeLock = ((PowerManager) getSystemService(POWER_SERVICE))
                    .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "rosettajane:p2p");
            wakeLock.acquire();
        }
        String[] peers = intent.getStringExtra("peers") == null
                ? new String[0] : intent.getStringExtra("peers").split("[,\\s]+");
        bridge(this).callAttr("start",
                intent.getStringExtra("model"), intent.getStringExtra("group"), peers,
                intent.getStringExtra("layers"), intent.getIntExtra("span", 2),
                intent.getBooleanExtra("training", false),
                getFilesDir().getAbsolutePath() + "/work", intent.getIntExtra("port", 7700));
        return START_NOT_STICKY;
    }

    private void shutDown() {
        try {
            bridge(this).callAttr("stop");
        } catch (Exception ignored) {
        }
        if (wakeLock != null && wakeLock.isHeld()) {
            wakeLock.release();
        }
        wakeLock = null;
        stopForeground(true);
        stopSelf();
    }

    @Override
    public void onDestroy() {
        shutDown();
        super.onDestroy();
    }
}
