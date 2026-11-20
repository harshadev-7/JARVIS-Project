package com.jarvis.ai

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import okhttp3.*
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * JARVIS Background Service
 * Maintains persistent WebSocket session with the JARVIS FastAPI backend.
 * Receives remote action dispatch requests (e.g. notifications, app launch).
 */
class JarvisClientService : Service() {

    companion object {
        private const val TAG = "JarvisService"
        private var webSocket: WebSocket? = null

        fun sendCommand(text: String) {
            val json = JSONObject().apply {
                put("type", "voice_command")
                put("command", text)
            }
            webSocket?.send(json.toString())
        }
    }

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        connectToJarvisHub("ws://192.168.1.100:8000/ws/devices")
        return START_STICKY
    }

    private fun connectToJarvisHub(url: String) {
        val request = Request.Builder().url(url).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i(TAG, "Connected to JARVIS Core Engine")
                // Transmit initial device handshake
                val handshake = JSONObject().apply {
                    put("type", "device_hello")
                    put("device_type", "android")
                    put("device_name", android.os.Build.MODEL)
                }
                webSocket.send(handshake.toString())
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.i(TAG, "Received message from JARVIS: $text")
                // Handle remote task executions and notifications
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "Connection error: ${t.message}. Reconnecting in 5s...")
            }
        })
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
