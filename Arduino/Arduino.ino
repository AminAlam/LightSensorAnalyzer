#include <Wire.h>

// VEML6030 I2C address
#define VEML6030_I2C_ADDRESS 0x48

// VEML6030 register addresses
#define VEML6030_ALS_CONF_REG 0x00
#define VEML6030_ALS_WH_REG   0x01
#define VEML6030_ALS_WL_REG   0x02
#define VEML6030_POW_SAV_REG  0x03
#define VEML6030_ALS_REG      0x04
#define VEML6030_WHITE_REG    0x05
#define VEML6030_ALS_INT_REG  0x06

// Photoresistor pin
#define PHOTORESISTOR_PIN A6

class VEML6030 {
private:
  uint8_t i2c_address;
  
  void writeRegister(uint8_t reg, uint16_t value) {
    Wire.beginTransmission(i2c_address);
    Wire.write(reg);
    Wire.write(value & 0xFF);
    Wire.write((value >> 8) & 0xFF);
    Wire.endTransmission();
    delay(1); // Add small delay after write
  }
  
  uint16_t readRegister(uint8_t reg) {
    Wire.beginTransmission(i2c_address);
    Wire.write(reg);
    Wire.endTransmission(false);
    
    uint8_t bytesRead = Wire.requestFrom(i2c_address, (uint8_t)2);
    if (bytesRead != 2) {
      return 0xFFFF; // Return error value if read failed
    }
    
    uint16_t value = Wire.read();
    value |= Wire.read() << 8;
    return value;
  }
  
public:
  VEML6030(uint8_t addr = VEML6030_I2C_ADDRESS) : i2c_address(addr) {}
  
  bool begin() {
    Wire.begin();
    delay(100); // Add delay after I2C init
    
    // Configure the sensor
    // ALS_CONF: ALS integration time = 100ms, ALS gain = 1/8, ALS enable
    writeRegister(VEML6030_ALS_CONF_REG, 0x0000);
    delay(100);
    
    // Check if sensor is responding
    uint16_t conf = readRegister(VEML6030_ALS_CONF_REG);
    return (conf != 0xFFFF);
  }
  
  uint16_t readALS() {
    uint16_t value = readRegister(VEML6030_ALS_REG);
    if (value == 0xFFFF) {
      return 0; // Return 0 instead of error value for readings
    }
    return value;
  }
  
  uint16_t readWhite() {
    uint16_t value = readRegister(VEML6030_WHITE_REG);
    if (value == 0xFFFF) {
      return 0; // Return 0 instead of error value for readings
    }
    return value;
  }
  
  float getLux() {
    uint16_t alsValue = readALS();
    // Convert to lux (this conversion factor depends on gain and integration time)
    // For gain = 1/8 and integration time = 100ms
    return alsValue * 0.0036;
  }
};

VEML6030 lightSensor;
unsigned long lastReading = 0;
const unsigned long readingInterval = 1; // Read every 2ms (500Hz)

void setup() {
  Serial.begin(115200);
  
  // Wait for serial connection
  while (!Serial && millis() < 3000) { // Add timeout
    delay(10);
  }
  
  Serial.println("VEML6030 Light Sensor with Photoresistor Initializing...");
  
  if (!lightSensor.begin()) {
    Serial.println("ERROR: Could not initialize VEML6030 sensor!");
    while (1) {
      delay(1000);
    }
  }
  
  Serial.println("VEML6030 sensor initialized successfully!");
  Serial.println("timestamp,als_raw,white_raw,lux,photoresistor");
}

void loop() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastReading >= readingInterval) {
    uint16_t alsValue = lightSensor.readALS();
    uint16_t whiteValue = lightSensor.readWhite();
    float luxValue = lightSensor.getLux();
    int photoresistorValue = analogRead(PHOTORESISTOR_PIN);
    
    // Send data as CSV format: timestamp,als_raw,white_raw,lux,photoresistor
    Serial.print(currentTime);
    Serial.print(",");
    Serial.print(alsValue);
    Serial.print(",");
    Serial.print(whiteValue);
    Serial.print(",");
    Serial.print(luxValue, 2);
    Serial.print(",");
    Serial.println(photoresistorValue);
    
    lastReading = currentTime;
  }
}