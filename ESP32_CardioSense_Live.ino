
#include <WiFi.h>
#include <HTTPClient.h>
const char* WIFI_SSID     = "MyWiFi";
const char* WIFI_PASSWORD = "rasPi#401911";
const char* SERVER_IP     = "10.232.12.149";   // your laptop's IP (run ipconfig)
const int SERVER_PORT = 8000;

const int ECG_PIN  = 34;   // AD8232 OUTPUT
const int LO_PLUS  = 32;   // AD8232 LO+
const int LO_MINUS = 33;   // AD8232 LO-

const int   WINDOW       = 250;   // 250 samples = 1 second at 250Hz
const int   SAMPLE_DELAY = 4;     // 4ms between samples = 250Hz

float samples[WINDOW];
int   count = 0;

// PuTTY monitoring over WiFi (optional but useful for debugging)
WiFiServer telnetServer(23);
WiFiClient telnetClient;

void log(String msg) {
  Serial.println(msg);
  if (telnetClient && telnetClient.connected()) {
    telnetClient.println(msg);
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(LO_PLUS,  INPUT);
  pinMode(LO_MINUS, INPUT);

  Serial.println();
  Serial.println("================================================");
  Serial.println("  CardioSense AI — Starting Up");
  Serial.println("================================================");
  Serial.print("Target WiFi    : ");
  Serial.println(WIFI_SSID);
  Serial.print("Target Server  : http://");
  Serial.print(SERVER_IP);
  Serial.print(":");
  Serial.println(SERVER_PORT);
  Serial.println("------------------------------------------------");

  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);
  delay(500);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting");
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    tries++;
    if (tries > 40) {
      Serial.println();
      Serial.println("FAILED to connect. Restarting in 5 sec...");
      delay(5000);
      ESP.restart();
    }
  }

  Serial.println();
  Serial.println("================================================");
  Serial.println("  WiFi CONNECTED");
  Serial.println("================================================");
  Serial.print("ESP32 IP Address : ");
  Serial.println(WiFi.localIP());
  Serial.print("Signal Strength  : ");
  Serial.print(WiFi.RSSI());
  Serial.println(" dBm");
  Serial.println("================================================");

  // Test the FastAPI connection immediately, before the main loop
  testServerConnection();

  // Start Telnet server so you can also monitor via PuTTY
  telnetServer.begin();
  telnetServer.setNoDelay(true);
  Serial.println("PuTTY Telnet ready on port 23");
  Serial.println("------------------------------------------------");
  Serial.println("Waiting for electrode contact...");
  Serial.println();
}

// ════════════════════════════════════════════════════════════
void testServerConnection() {
  HTTPClient http;
  String url = "http://" + String(SERVER_IP) + ":" + String(SERVER_PORT) + "/ping";

  Serial.print("Testing connection to: ");
  Serial.println(url);

  http.begin(url);
  http.setTimeout(5000);
  int code = http.GET();

  if (code == 200) {
    Serial.println("SERVER REACHABLE!");
    Serial.println("Response: " + http.getString());
  } else if (code == -1) {
    Serial.println("SERVER NOT REACHABLE.");
    Serial.println("Check: FastAPI running? Correct IP? Firewall open?");
  } else {
    Serial.println("Server returned error code: " + String(code));
  }
  http.end();
  Serial.println("------------------------------------------------");
}

void loop() {

  // Accept PuTTY connections
  if (telnetServer.hasClient()) {
    if (!telnetClient || !telnetClient.connected()) {
      telnetClient = telnetServer.available();
      telnetClient.println("CardioSense AI — Live Monitor (via PuTTY)");
    }
  }

  // Reconnect WiFi if dropped
  if (WiFi.status() != WL_CONNECTED) {
    log("WiFi disconnected! Reconnecting...");
    WiFi.reconnect();
    delay(3000);
    return;
  }

  // Check electrode contact
  if (digitalRead(LO_PLUS) == HIGH || digitalRead(LO_MINUS) == HIGH) {
    log("Attach electrode pads to body...");
    count = 0;
    delay(1000);
    return;
  }

  // Read one ECG sample
  samples[count] = (float)analogRead(ECG_PIN);
  count++;

  if (count % 50 == 0) {
    log("Collecting... " + String(count) + "/" + String(WINDOW) +
        "  (last value: " + String(samples[count-1], 0) + ")");
  }

  // Buffer full (1 second of data) — send to FastAPI
  if (count >= WINDOW) {
    log("Sending " + String(WINDOW) + " samples to FastAPI...");
    sendToServer();
    count = 0;
  }

  delay(SAMPLE_DELAY);
}

// ════════════════════════════════════════════════════════════
void sendToServer() {
  if (WiFi.status() != WL_CONNECTED) {
    log("Cannot send — WiFi not connected");
    return;
  }

  // Build JSON matching FastAPI's ECGBatch model exactly:
  //   device_id   (string)
  //   patient_id  (string)
  //   sample_rate (int)
  //   samples     (array of floats)
  String json = "{\"device_id\":\"esp32-01\","
                 "\"patient_id\":\"patient-001\","
                 "\"sample_rate\":250,"
                 "\"samples\":[";

  for (int i = 0; i < WINDOW; i++) {
    json += String(samples[i], 1);
    if (i < WINDOW - 1) json += ",";
  }
  json += "]}";

  HTTPClient http;
  String url = "http://" + String(SERVER_IP) + ":" + String(SERVER_PORT) + "/stream_batch";

  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(10000);

  int code = http.POST(json);

  if (code == 200) {
    log("Sent OK. Response: " + http.getString());
  } else if (code == -1) {
    log("ERROR -1: Cannot reach server. Check IP and firewall.");
  } else {
    log("Server error code: " + String(code));
  }

  http.end();
}
