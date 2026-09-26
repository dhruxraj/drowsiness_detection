/*
  Driver Drowsiness Detection - Arduino alarm node
  ------------------------------------------------
  Receives one-byte commands from the PC over USB serial (9600 baud):
     'A' -> alarm ON  (beeping buzzer + relay closed + LED on)
     'S' -> alarm OFF
  Wiring (Arduino Uno / Nano):
     D8  -> active or passive buzzer (+), buzzer (-) -> GND
            (use a transistor, e.g. 2N2222 + 1k base resistor, for buzzers > 20 mA)
     D7  -> relay MODULE IN pin (module VCC -> 5V, GND -> GND)
            relay contacts can switch an external siren / vehicle alarm input
     D13 -> on-board LED (status)
*/

const int BUZZER_PIN = 8;
const int RELAY_PIN  = 7;
const int LED_PIN    = 13;
const bool RELAY_ACTIVE_LOW = true;   // most cheap relay modules are active-LOW

const unsigned long BEEP_ON_MS  = 250;
const unsigned long BEEP_OFF_MS = 150;
const unsigned int  TONE_HZ     = 2500;

bool alarmOn = false;
bool beepPhase = false;
unsigned long lastToggle = 0;

void setRelay(bool on) {
  digitalWrite(RELAY_PIN, (on ^ RELAY_ACTIVE_LOW) ? HIGH : LOW);
}

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  setRelay(false);
  Serial.begin(9600);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'A') { alarmOn = true;  setRelay(true);  digitalWrite(LED_PIN, HIGH); }
    if (c == 'S') { alarmOn = false; setRelay(false); digitalWrite(LED_PIN, LOW);
                    noTone(BUZZER_PIN); digitalWrite(BUZZER_PIN, LOW); }
  }

  if (alarmOn) {                       // non-blocking beep pattern
    unsigned long now = millis();
    unsigned long wait = beepPhase ? BEEP_ON_MS : BEEP_OFF_MS;
    if (now - lastToggle >= wait) {
      lastToggle = now;
      beepPhase = !beepPhase;
      if (beepPhase) tone(BUZZER_PIN, TONE_HZ);   // passive buzzer: tone; active buzzer: also works
      else           noTone(BUZZER_PIN);
    }
  }
}
