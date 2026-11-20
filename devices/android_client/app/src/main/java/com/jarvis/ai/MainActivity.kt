package com.jarvis.ai

import android.os.Bundle
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Environment
import android.os.StatFs
import android.speech.RecognizerIntent
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import java.util.Locale

/**
 * JARVIS AI — Android Mobile Controller
 * Handles native mobile voice commands, battery & storage telemetry,
 * and encrypted WebSocket communication with the JARVIS central engine.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var statusText: TextView
    private lateinit var telemetryText: TextView
    private lateinit var voiceButton: Button
    private val SPEECH_REQUEST_CODE = 101

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.tv_status)
        telemetryText = findViewById(R.id.tv_telemetry)
        voiceButton = findViewById(R.id.btn_voice)

        updateTelemetry()

        voiceButton.setOnClickListener {
            startSpeechRecognizer()
        }

        // Launch background WebSocket listener
        val serviceIntent = Intent(this, JarvisClientService::class.java)
        startService(serviceIntent)
    }

    private fun startSpeechRecognizer() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PROMPT, "JARVIS is listening...")
        }
        try {
            startActivityForResult(intent, SPEECH_REQUEST_CODE)
        } catch (e: Exception) {
            Toast.makeText(this, "Speech recognition unavailable: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == SPEECH_REQUEST_CODE && resultCode == RESULT_OK) {
            val spokenText = data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)?.firstOrNull()
            spokenText?.let {
                statusText.text = "JARVIS Processing: \"$it\""
                // Forward speech command to background WebSocket service
                JarvisClientService.sendCommand(it)
            }
        }
    }

    private fun updateTelemetry() {
        val batteryIntent = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        val batteryPct = if (level != -1 && scale != -1) (level * 100 / scale.toFloat()) else 0f

        val stat = StatFs(Environment.getDataDirectory().path)
        val freeBytes = stat.availableBlocksLong * stat.blockSizeLong
        val freeGb = freeBytes / (1024 * 1024 * 1024)

        telemetryText.text = "Battery: ${batteryPct.toInt()}% | Storage Free: ${freeGb} GB"
    }
}
