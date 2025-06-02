# 💡 Light Sensor Analyzer

A comprehensive real-time light sensor analysis system that connects to Arduino-based VEML6030 sensors, providing live monitoring, signal analysis, and automated PDF report generation.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

## 📸 Interface Preview

![Light Sensor Analyzer Interface](assets/screenshot.png)

*Real-time light sensor monitoring with automated PDF report generation featuring a modern three-panel layout: report controls (left), live charts (center), and signal analysis (right).*

## ✨ Features

### 📊 Real-Time Monitoring
- **Live Data Visualization**: Real-time plotting of LUX and ALS (Ambient Light Sensor) values
- **FFT Analysis**: Frequency domain analysis with interactive charts
- **Multi-Window Analysis**: Short-term (2s), medium-term (10s), and long-term (60s) analysis windows
- **WebSocket Communication**: Ultra-fast data updates with minimal latency

### 📈 Signal Analysis
- **Dominant Frequency Detection**: Automatic identification of primary signal frequencies
- **Duty Cycle Calculation**: Percentage of time signal is above mean value
- **Period Analysis**: Signal period measurement in milliseconds
- **Rise/Fall Time**: Edge timing analysis for signal transitions
- **Statistical Metrics**: Mean, standard deviation, min/max values

### 📄 Automated Reporting
- **PDF Report Generation**: Comprehensive analysis reports with professional formatting
- **Signal Snapshots**: 2-second signal overview plots
- **Average Period Analysis**: Signal consistency analysis with error bounds
- **FFT Spectrum**: Frequency domain analysis plots
- **Measurements Table**: Detailed metrics with statistical uncertainties
- **Auto-Download**: Reports automatically download when recording completes
- **File Cleanup**: Automatic server-side file management

### 🎛️ User Interface
- **Modern Web Interface**: Responsive design works on desktop and mobile
- **Real-Time Progress**: Visual progress bars and status indicators
- **Flexible Recording**: Configurable recording duration (0.5 to 120 minutes)
- **Intuitive Controls**: Simple start/stop recording interface

## 🔧 Hardware Requirements

### Arduino Setup
- **Microcontroller**: Arduino Uno, Nano, or compatible board
- **Sensor**: VEML6030 Ambient Light Sensor
- **Connection**: I2C communication (SDA/SCL pins)
- **USB Cable**: For serial communication with computer

### VEML6030 Sensor Wiring
```
Arduino    VEML6030
------     --------
VCC   →    VDD (3.3V)
GND   →    GND
A4    →    SDA
A5    →    SCL
```

### Expected Arduino Output Format
The Arduino should send CSV data via serial at 115200 baud:
```
timestamp,als_raw,white_raw,lux_value
1234,156,89,12.34
1236,158,91,12.56
...
```

## 🚀 Software Installation

### Prerequisites
- **Python 3.8+** installed on your system
- **Arduino IDE** for programming the microcontroller
- **Git** for cloning the repository

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/veml6030-analyzer.git
cd veml6030-analyzer
```

### 2. Set Up Python Environment
```bash
cd Backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Arduino Setup
1. Connect VEML6030 sensor to Arduino as shown in wiring diagram
2. Upload the Arduino sketch from the `Arduino/` directory
3. Verify serial output at 115200 baud shows CSV format data
4. Note the COM port (Windows) or device path (macOS/Linux)

### 4. Start the Application
```bash
# From Backend directory
python src/main.py
```

The web interface will be available at: `http://localhost:7001`

## 📱 Usage

### Starting Real-Time Monitoring
1. **Connect Hardware**: Ensure Arduino with VEML6030 is connected via USB
2. **Launch Application**: Run `python src/main.py` from Backend directory
3. **Open Web Interface**: Navigate to `http://localhost:7001`
4. **Verify Connection**: Check that "Connected to sensor" appears in green

### Recording and Report Generation
1. **Set Duration**: Enter recording time in minutes (0.5 for testing, 30+ for analysis)
2. **Start Recording**: Click "🎥 Start Recording"
3. **Monitor Progress**: Watch real-time progress bar and time counters
4. **Automatic Completion**: Report generates and downloads automatically when done

### Understanding the Analysis
- **LUX Values**: Calibrated light intensity measurements
- **ALS Raw**: Raw sensor readings from ambient light detector
- **FFT Charts**: Frequency content showing periodic components
- **Analysis Windows**: Different time scales for various signal characteristics

## 📊 API Documentation

### Endpoints

#### Recording Control
- `POST /api/start-recording` - Start data recording
  ```json
  {
    "duration": 30.0  // Duration in minutes
  }
  ```

- `POST /api/stop-recording` - Stop current recording

#### Status and Data
- `GET /api/recording-status` - Get current recording status
- `GET /api/data` - Get latest sensor data and analysis
- `GET /api/download-report/<filename>` - Download generated PDF report

#### WebSocket Events
- `data_update` - Real-time sensor data and analysis updates
- `connect/disconnect` - Connection status events

## 📁 Project Structure

```
veml6030-analyzer/
├── Arduino/                    # Arduino sketches
│   └── veml6030_sensor/       # Main sensor sketch
├── Backend/                   # Python Flask application
│   ├── requirements.txt       # Python dependencies
│   ├── src/
│   │   ├── main.py           # Main Flask application
│   │   ├── report_generator.py # PDF report generation
│   │   └── templates/
│   │       └── index.html    # Web interface
│   └── venv/                 # Virtual environment (created)
├── Arduino_test/             # Test sketches and utilities
└── README.md                 # This file
```

## 🔬 Technical Details

### Signal Processing
- **Sampling Rate**: 1000 Hz configurable
- **Window Functions**: Hann windowing for FFT analysis
- **Peak Detection**: Scipy-based peak finding algorithms
- **Statistical Analysis**: Segmented analysis for uncertainty quantification

### Performance
- **Real-Time Updates**: ~50Hz web interface updates
- **Data Buffering**: Efficient circular buffers for continuous operation
- **Memory Management**: Automatic cleanup of temporary files
- **Multi-Threading**: Separate threads for serial reading, analysis, and web serving

### Report Features
- **Professional PDF Layout**: Multi-page reports with plots and tables
- **High-Resolution Plots**: 150 DPI matplotlib figures
- **Statistical Tables**: Mean ± standard deviation for all measurements
- **Frequency Analysis**: Complete FFT spectrum analysis

## 🛠️ Troubleshooting

### Common Issues

**"No Arduino found" Error**
- Check USB connection and drivers
- Verify correct COM port in Device Manager (Windows)
- Ensure Arduino sketch is uploaded and running

**Web Interface Not Loading**
- Check that port 7001 is not blocked by firewall
- Try `http://127.0.0.1:7001` instead of localhost
- Verify Python application started without errors

**No Data Appearing**
- Check Arduino serial output in Arduino IDE Serial Monitor
- Verify baud rate is set to 115200
- Ensure VEML6030 sensor is properly wired and powered

**Report Generation Fails**
- Check that `reports/` directory exists in `Backend/src/`
- Verify matplotlib backend is properly configured
- Ensure sufficient disk space for PDF generation

### Getting Help
1. Check the [Issues](https://github.com/yourusername/veml6030-analyzer/issues) page
2. Review Arduino serial output for data format
3. Check Python console for error messages
4. Verify all dependencies are installed correctly

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the Repository**
2. **Create a Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Changes**: `git commit -m 'Add amazing feature'`
4. **Push to Branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **VEML6030 Library**: Based on Arduino VEML6030 sensor libraries
- **Flask-SocketIO**: Real-time web communication
- **Plotly.js**: Interactive charting and visualization
- **ReportLab**: Professional PDF generation
- **SciPy**: Signal processing and analysis algorithms

---

**Made with ❤️ for scientific measurement and analysis**

*Ideal for research applications, LED testing, display analysis, ambient light studies, and educational projects.* 