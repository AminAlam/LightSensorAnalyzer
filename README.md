# 💡 Light Sensor Analyzer

A real-time light sensor analysis system that connects to Arduino-based VEML6030 sensors, providing live monitoring, signal analysis, and automated PDF report generation.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📸 Interface Preview

![Light Sensor Analyzer Interface](assets/screenshot.png)

## ✨ Features

- **Real-Time Monitoring**: Live LUX and ALS value visualization
- **Signal Analysis**: FFT analysis, dominant frequency detection, duty cycle calculation
- **Automated PDF Reports**: Professional reports with charts and statistical analysis
- **Modern Web Interface**: Responsive design with real-time updates
- **Multiple Time Windows**: Short-term (2s), medium-term (10s), and long-term (60s) analysis

## 🔧 Hardware Requirements

### Arduino Setup
- Arduino Uno/Nano or compatible
- VEML6030 Ambient Light Sensor
- USB cable for serial communication

### VEML6030 Wiring
```
Arduino    VEML6030
------     --------
VCC   →    VDD (3.3V)
GND   →    GND
A4    →    SDA
A5    →    SCL
```

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/LightSensorAnalyzer.git
cd LightSensorAnalyzer
```

### 2. Setup Python Environment
```bash
cd Backend
python -m venv venv

# Activate virtual environment
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
```

### 3. Arduino Setup
1. Connect VEML6030 sensor using wiring diagram above
2. Upload Arduino sketch from `Arduino/` directory
3. Verify serial output at 115200 baud shows CSV data
4. Note the COM port/device path

### 4. Start Application
```bash
# From Backend directory
python src/main.py
```

Web interface available at: `http://localhost:7001`

## 📱 Usage

1. **Connect**: Ensure Arduino with VEML6030 is connected via USB
2. **Launch**: Run the application and open web interface
3. **Record**: Set duration and click "🎥 Start Recording"
4. **Analyze**: View real-time charts and automatic PDF report generation

## 🛠️ Troubleshooting

**"No Arduino found"**
- Check USB connection and COM port
- Verify Arduino sketch is uploaded and running

**No data appearing**
- Check Arduino serial output (115200 baud)
- Verify VEML6030 wiring and power

**Web interface issues**
- Try `http://127.0.0.1:7001`
- Check port 7001 isn't blocked by firewall

## 📁 Project Structure

```
LightSensorAnalyzer/
├── Arduino/           # Arduino sensor sketches
├── Backend/           # Python Flask application
│   ├── src/
│   │   ├── main.py
│   │   └── templates/
│   └── requirements.txt
└── README.md
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for scientific measurement and analysis** 