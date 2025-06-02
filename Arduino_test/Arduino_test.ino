#include <Wire.h>

// Constants for synthetic signal generation
const float SIGNAL_FREQUENCY = 60.0;  // 60 Hz signal
const float SIGNAL_AMPLITUDE = 1000.0; // Amplitude of the signal
const float SIGNAL_OFFSET = 1000.0;   // DC offset
const unsigned long SAMPLE_RATE = 1000; // 1000 Hz sampling rate (1ms interval)

unsigned long lastReading = 0;
const unsigned long readingInterval = 1; // Read every 1ms (1000Hz)

void setup() {
  Serial.begin(115200);
  
  // Wait for serial connection
  while (!Serial) {
    delay(10);
  }
  
  Serial.println("Synthetic 60Hz Signal Generator");
  Serial.println("timestamp,als_raw,white_raw,lux");
}

void loop() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastReading >= readingInterval) {
    // Calculate time in seconds for sine wave
    float t = currentTime / 1000.0;
    
    // Generate synthetic signal using sine wave
    // sin(2π * f * t) gives us a sine wave at frequency f
    float signal = SIGNAL_AMPLITUDE * sin(2 * PI * SIGNAL_FREQUENCY * t) + SIGNAL_OFFSET;
    
    // Ensure signal is positive and within reasonable range
    uint16_t alsValue = (uint16_t)constrain(signal, 0, 65535);
    uint16_t whiteValue = (uint16_t)constrain(signal * 0.8, 0, 65535); // White is slightly lower
    float luxValue = signal * 0.0036; // Convert to lux using same factor as original
    
    // Send data as CSV format: timestamp,als_raw,white_raw,lux
    Serial.print(currentTime);
    Serial.print(",");
    Serial.print(alsValue);
    Serial.print(",");
    Serial.print(whiteValue);
    Serial.print(",");
    Serial.println(luxValue, 2);
    
    lastReading = currentTime;
  }
}