/*
  ESP32 + MPU6050 data logger for smart-home hand gesture collection.

  Wiring:
    MPU6050 VCC -> ESP32 3V3
    MPU6050 GND -> ESP32 GND
    MPU6050 SDA -> ESP32 GPIO21
    MPU6050 SCL -> ESP32 GPIO22
    MPU6050 AD0 -> GND, I2C address 0x68

  Serial commands:
    PING       -> PONG
    START      -> begin streaming CSV-like DATA lines at 100 Hz
    STOP       -> stop streaming
    CALIBRATE  -> estimate gyro offset while the sensor is still

*/
#include <Arduino.h>
#include <Wire.h>

static const uint8_t MPU_ADDR = 0x68;
static const int SDA_PIN = 21;
static const int SCL_PIN = 22;
static const uint32_t SERIAL_BAUD = 115200;
static const uint32_t SAMPLE_RATE_HZ = 100;
static const uint32_t SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;

static const uint8_t REG_SMPLRT_DIV = 0x19;
static const uint8_t REG_CONFIG = 0x1A;
static const uint8_t REG_GYRO_CONFIG = 0x1B;
static const uint8_t REG_ACCEL_CONFIG = 0x1C;
static const uint8_t REG_ACCEL_XOUT_H = 0x3B;
static const uint8_t REG_PWR_MGMT_1 = 0x6B;
static const uint8_t REG_WHO_AM_I = 0x75;

static const float ACCEL_SCALE = 8192.0f;  // +/-4g
static const float GYRO_SCALE = 65.5f;     // +/-500 deg/s

bool streaming = false;
uint32_t sampleIndex = 0;
uint32_t nextSampleAt = 0;
float gyroOffsetX = 0.0f;
float gyroOffsetY = 0.0f;
float gyroOffsetZ = 0.0f;

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission(true);
}

uint8_t readRegister(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) {
    return 0xFF;
  }
  Wire.requestFrom(MPU_ADDR, (uint8_t)1, (uint8_t)true);
  return Wire.available() ? Wire.read() : 0xFF;
}

bool readRaw(int16_t &ax, int16_t &ay, int16_t &az,
             int16_t &temp, int16_t &gx, int16_t &gy, int16_t &gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  const uint8_t expected = 14;
  uint8_t received = Wire.requestFrom(MPU_ADDR, expected, (uint8_t)true);
  if (received != expected) {
    return false;
  }

  ax = (Wire.read() << 8) | Wire.read();
  ay = (Wire.read() << 8) | Wire.read();
  az = (Wire.read() << 8) | Wire.read();
  temp = (Wire.read() << 8) | Wire.read();
  gx = (Wire.read() << 8) | Wire.read();
  gy = (Wire.read() << 8) | Wire.read();
  gz = (Wire.read() << 8) | Wire.read();
  return true;
}

bool setupMpu6050() {
  writeRegister(REG_PWR_MGMT_1, 0x00);
  delay(100);

  uint8_t whoAmI = readRegister(REG_WHO_AM_I);
  if (whoAmI != 0x68 && whoAmI != 0x69) {
    Serial.print("ERROR,MPU6050_NOT_FOUND,WHO_AM_I=");
    Serial.println(whoAmI, HEX);
    return false;
  }

  writeRegister(REG_CONFIG, 0x03);        // DLPF about 44 Hz accel / 42 Hz gyro
  writeRegister(REG_SMPLRT_DIV, 0x09);    // 1 kHz / (1 + 9) = 100 Hz
  writeRegister(REG_GYRO_CONFIG, 0x08);   // +/-500 deg/s
  writeRegister(REG_ACCEL_CONFIG, 0x08);  // +/-4g
  return true;
}

void calibrateGyro() {
  const int samples = 500;
  long sx = 0;
  long sy = 0;
  long sz = 0;
  int valid = 0;

  Serial.println("CALIBRATING,keep_sensor_still");
  for (int i = 0; i < samples; i++) {
    int16_t ax, ay, az, temp, gx, gy, gz;
    if (readRaw(ax, ay, az, temp, gx, gy, gz)) {
      sx += gx;
      sy += gy;
      sz += gz;
      valid++;
    }
    delay(4);
  }

  if (valid > 0) {
    gyroOffsetX = (float)sx / valid;
    gyroOffsetY = (float)sy / valid;
    gyroOffsetZ = (float)sz / valid;
  }

  Serial.print("CALIBRATED,valid=");
  Serial.print(valid);
  Serial.print(",gx_offset=");
  Serial.print(gyroOffsetX, 3);
  Serial.print(",gy_offset=");
  Serial.print(gyroOffsetY, 3);
  Serial.print(",gz_offset=");
  Serial.println(gyroOffsetZ, 3);
}

void emitHeader() {
  Serial.println("HEADER,millis_ms,sample_index,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,temp_c");
}

void emitSample() {
  int16_t axRaw, ayRaw, azRaw, tempRaw, gxRaw, gyRaw, gzRaw;
  if (!readRaw(axRaw, ayRaw, azRaw, tempRaw, gxRaw, gyRaw, gzRaw)) {
    Serial.println("ERROR,READ_FAILED");
    return;
  }

  float ax = axRaw / ACCEL_SCALE;
  float ay = ayRaw / ACCEL_SCALE;
  float az = azRaw / ACCEL_SCALE;
  float gx = (gxRaw - gyroOffsetX) / GYRO_SCALE;
  float gy = (gyRaw - gyroOffsetY) / GYRO_SCALE;
  float gz = (gzRaw - gyroOffsetZ) / GYRO_SCALE;
  float tempC = (tempRaw / 340.0f) + 36.53f;

  Serial.print("DATA,");
  Serial.print(millis());
  Serial.print(',');
  Serial.print(sampleIndex++);
  Serial.print(',');
  Serial.print(ax, 5);
  Serial.print(',');
  Serial.print(ay, 5);
  Serial.print(',');
  Serial.print(az, 5);
  Serial.print(',');
  Serial.print(gx, 5);
  Serial.print(',');
  Serial.print(gy, 5);
  Serial.print(',');
  Serial.print(gz, 5);
  Serial.print(',');
  Serial.println(tempC, 2);
}

void handleCommand(const String &command) {
  String cmd = command;
  cmd.trim();
  cmd.toUpperCase();

  if (cmd == "PING") {
    Serial.println("PONG");
  } else if (cmd == "START") {
    streaming = true;
    sampleIndex = 0;
    nextSampleAt = micros();
    emitHeader();
    Serial.println("STARTED");
  } else if (cmd == "STOP") {
    streaming = false;
    Serial.println("STOPPED");
  } else if (cmd == "CALIBRATE") {
    bool wasStreaming = streaming;
    streaming = false;
    calibrateGyro();
    streaming = wasStreaming;
  } else if (cmd.length() > 0) {
    Serial.print("ERROR,UNKNOWN_COMMAND,");
    Serial.println(cmd);
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  delay(300);

  bool ok = setupMpu6050();
  Serial.print("READY,esp32_mpu6050_logger,rate_hz=");
  Serial.print(SAMPLE_RATE_HZ);
  Serial.print(",baud=");
  Serial.print(SERIAL_BAUD);
  Serial.print(",mpu=");
  Serial.println(ok ? "ok" : "error");
}

void loop() {
  while (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    handleCommand(command);
  }

  if (!streaming) {
    delay(1);
    return;
  }

  uint32_t now = micros();
  if ((int32_t)(now - nextSampleAt) >= 0) {
    nextSampleAt += SAMPLE_INTERVAL_US;
    emitSample();
  }
}
