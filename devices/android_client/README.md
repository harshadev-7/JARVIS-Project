# 📱 JARVIS AI — Android Mobile Client Node

This mobile module connects your Android smartphone or tablet directly into the JARVIS multi-device intelligence mesh.

---

## ⚡ Features
- **Hands-Free Speech Capture**: Leverages Google Android SpeechRecognizer for local voice capture and dispatch to JARVIS.
- **Persistent WebSocket Link**: Auto-reconnecting socket connection communicating directly with `/ws/devices` or `/ws/stream`.
- **Remote Action Execution**: Allows JARVIS to remotely dispatch actions, triggers, and push notifications to your phone.
- **Hardware Telemetry Feedback**: Streams battery %, charge state, and storage metrics to the Holographic HUD in real-time.

---

## 🛠️ Building & Installing

### Prerequisites
- Android Studio Hedgehog / Iguana or later
- Android SDK 34 (Android 14+) with minimum SDK 24 (Android 7.0+)
- JDK 17

### 1. Build via Gradle
```bash
cd devices/android_client
./gradlew assembleDebug
```
The compiled APK will be generated at:
`devices/android_client/app/build/outputs/apk/debug/app-debug.apk`

### 2. Install on Device via ADB
```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### 3. Pairing Handshake
1. On your desktop HUD or CLI, run `POST /devices/pairing-code` to generate a 6-digit one-time code (e.g. `258310`).
2. Pair via API or UI:
```bash
curl -X POST http://<YOUR_PC_IP>:8000/devices/pair \
  -H "Content-Type: application/json" \
  -d '{"pairing_code": "258310", "name": "Harsh Pixel Phone", "device_type": "android"}'
```
3. Launch the JARVIS app on your device to begin telemetry streaming and remote control.
