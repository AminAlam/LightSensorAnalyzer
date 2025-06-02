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

class VEML6030 {
private:
  uint8_t i2c_address;
  
  void writeRegister(uint8_t reg, uint16_t value) {
    Wire.beginTransmission(i2c_address);
    Wire.write(reg);
    Wire.write(value & 0xFF);
    Wire.write((value >> 8) & 0xFF);
    Wire.endTransmission();
  }
  
  uint16_t readRegister(uint8_t reg) {
    Wire.beginTransmission(i2c_address);
    Wire.write(reg);
    Wire.endTransmission(false);
    
    Wire.requestFrom(i2c_address, (uint8_t)2);
    uint16_t value = Wire.read();
    value |= Wire.read() << 8;
    return value;
  }
  
public:
  VEML6030(uint8_t addr = VEML6030_I2C_ADDRESS) : i2c_address(addr) {}
  
  bool begin() {
    Wire.begin();
    
    // Configure the sensor
    // ALS_CONF: ALS integration time = 100ms, ALS gain = 1/8, ALS enable
    writeRegister(VEML6030_ALS_CONF_REG, 0x0000);
    delay(100);
    
    // Check if sensor is responding
    uint16_t conf = readRegister(VEML6030_ALS_CONF_REG);
    return (conf != 0xFFFF);
  }
  
  uint16_t readALS() {
    return readRegister(VEML6030_ALS_REG);
  }
  
  uint16_t readWhite() {
    return readRegister(VEML6030_WHITE_REG);
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
const unsigned long readingInterval = 4; // Read every 4ms (250Hz)

void setup() {
  Serial.begin(115200);
  
  // Wait for serial connection
  while (!Serial) {
    delay(10);
  }
  
  Serial.println("VEML6030 Light Sensor Initializing...");
  
  if (!lightSensor.begin()) {
    Serial.println("ERROR: Could not initialize VEML6030 sensor!");
    while (1) {
      delay(1000);
    }
  }
  
  Serial.println("VEML6030 sensor initialized successfully!");
  Serial.println("timestamp,als_raw,white_raw,lux");
}

void loop() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastReading >= readingInterval) {
    uint16_t alsValue = lightSensor.readALS();
    uint16_t whiteValue = lightSensor.readWhite();
    float luxValue = lightSensor.getLux();
    
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
  
  // delay(4);
}