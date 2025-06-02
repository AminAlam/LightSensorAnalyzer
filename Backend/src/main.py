import serial
import serial.tools.list_ports
import threading
import time
import json
import numpy as np
from scipy import signal
from collections import deque
from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
from report_generator import ReportGenerator
import os

class LightSensorAnalyzer:
    def __init__(self, port=None, baudrate=115200, window_size=1000):
        self.port = port
        self.baudrate = baudrate
        self.window_size = window_size
        self.sampling_rate = 1000  # Hz (based on 2ms interval in Arduino)
        
        # Data storage
        self.timestamps = deque(maxlen=window_size)
        self.als_values = deque(maxlen=window_size)
        self.white_values = deque(maxlen=window_size)
        self.lux_values = deque(maxlen=window_size)
        
        # Analysis windows
        self.window_1 = deque(maxlen=2*self.sampling_rate)  # 2 seconds at 20Hz
        self.window_2 = deque(maxlen=10*self.sampling_rate)  # 10 seconds at 20Hz
        self.window_3 = deque(maxlen=60*self.sampling_rate)  # 60 seconds at 20Hz
        
        self.serial_connection = None
        self.is_running = False
        self.data_lock = threading.Lock()
        
    def find_arduino_port(self):
        """Automatically find Arduino port"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(port.description)
            if 'Arduino' in port.description or 'USB' in port.description or 'Nano' in port.description:
                return port.device
        return None
    
    def connect_serial(self):
        """Connect to Arduino via serial"""
        if self.port is None:
            self.port = self.find_arduino_port()
            
        if self.port is None:
            print("No Arduino found. Please specify port manually.")
            return False
            
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            print(f"Connected to Arduino on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to {self.port}: {e}")
            return False
    
    def parse_serial_data(self, line):
        """Parse CSV data from Arduino"""
        try:
            parts = line.strip().split(',')
            if len(parts) == 4:
                timestamp = float(parts[0])
                als_raw = int(parts[1])
                white_raw = int(parts[2])
                lux = float(parts[3])
                return timestamp, als_raw, white_raw, lux
        except:
            pass
        return None
    
    def calculate_dominant_frequency(self, data):
        """Calculate dominant frequency using FFT"""
        if len(data) < 10:
            return 0
            
        # Apply window function to reduce spectral leakage
        windowed_data = np.array(data)
        
        # Perform FFT
        fft = np.fft.fftshift(np.fft.fft(windowed_data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data), 1/self.sampling_rate))

        fft = fft[len(fft)//2+1:]
        freqs = freqs[len(freqs)//2+1:]

        # Find dominant frequency (exclude DC component)
        magnitude = np.abs(fft)
        if len(magnitude) > 0:
            max_idx = np.argmax(magnitude)
            dominant_freq = abs(freqs[max_idx])
            return dominant_freq
        return 0
    
    def calculate_fft(self, data):
        """Calculate FFT of the light signal"""
        if len(data) < 10:
            return [], []
            
        windowed_data = np.array(data)
        fft = np.fft.fftshift(np.fft.fft(windowed_data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data), 1/self.sampling_rate))

        fft = fft[len(fft)//2+1:]
        freqs = freqs[len(freqs)//2+1:]

        # Find dominant frequency (exclude DC component)
        magnitude = np.abs(fft)
 
        # magnitude is in dB - safely handle log conversion
        try:
            # Add small epsilon to avoid log(0) = -inf
            magnitude_safe = np.maximum(magnitude, 1e-10)
            magnitude_db = 20 * np.log10(magnitude_safe)
            
            # Replace any remaining inf or nan values
            magnitude_db = np.nan_to_num(magnitude_db, nan=0.0, posinf=0.0, neginf=-100.0)
            
            return freqs, magnitude_db
        except:
            return [], []
    
    def calculate_duty_cycle(self, data):
        """Calculate duty cycle of the light signal"""
        if len(data) < 10:
            return 0
            
        data_array = np.array(data)
        mean_value = np.mean(data_array)
        
        # Count samples above mean as "high"
        high_samples = np.sum(data_array > mean_value)
        duty_cycle = (high_samples / len(data_array)) * 100
        return duty_cycle
    
    def calculate_period(self, data):
        """Calculate period of the light signal"""
        if len(data) < 10:
            return 0
        data = np.array(data)
        signal_length = len(data)

        peaks_indices, _ = signal.find_peaks(data)
        peaks_times_ms = peaks_indices / self.sampling_rate
        
        # Calculate period between first two peaks
        periods_diff = np.diff(peaks_times_ms)
        period = np.mean(periods_diff)*1000 # convert to ms
        if np.isnan(period):
            return 0
        return period
    
    def calculate_rise_fall_time(self, data):
        """Calculate rise time of the light signal"""
        if len(data) < 10:
            return 0, 0
        data = np.array(data)
        # find all the peaks
        peaks_indices, _ = signal.find_peaks(data)
        peak_values = data[peaks_indices]
        rise_times = []
        fall_times = []
        for i in range(len(peak_values)):
            threshold = peak_values[i] * 0.05
            try:
                rise_time_index = np.nonzero(data[:peaks_indices[i]] < threshold)[0][-1]
                fall_time_index = np.nonzero(data[peaks_indices[i]:] < threshold)[0][0] + peaks_indices[i]
                rise_times.append((peaks_indices[i]-rise_time_index)/self.sampling_rate)
                fall_times.append((fall_time_index-peaks_indices[i])/self.sampling_rate)
            except:
                pass

        rise_time = np.mean(rise_times)*1000 # convert to ms
        fall_time = np.mean(fall_times)*1000 # convert to ms
        return rise_time, fall_time


    def analyze_windows(self):
        """Analyze all windows and return results"""
        with self.data_lock:
            rise_time_w1, fall_time_w1 = self.calculate_rise_fall_time(self.window_1)
            rise_time_w2, fall_time_w2 = self.calculate_rise_fall_time(self.window_2)
            rise_time_w3, fall_time_w3 = self.calculate_rise_fall_time(self.window_3)
            results = {
                'window_1': {
                    'size': len(self.window_1),
                    'frequency': self.calculate_dominant_frequency(self.window_1),
                    'duty_cycle': self.calculate_duty_cycle(self.window_1),
                    'period': self.calculate_period(self.window_1),
                    'rise_time': rise_time_w1,
                    'fall_time': fall_time_w1
                },
                'window_2': {
                    'size': len(self.window_2),
                    'frequency': self.calculate_dominant_frequency(self.window_2),
                    'duty_cycle': self.calculate_duty_cycle(self.window_2),
                    'period': self.calculate_period(self.window_2),
                    'rise_time': rise_time_w2,
                    'fall_time': fall_time_w2
                },
                'window_3': {
                    'size': len(self.window_3),
                    'frequency': self.calculate_dominant_frequency(self.window_3),
                    'duty_cycle': self.calculate_duty_cycle(self.window_3),
                    'period': self.calculate_period(self.window_3),
                    'rise_time': rise_time_w3,
                    'fall_time': fall_time_w3
                }
            }
        return results
    
    def get_latest_data(self, n_points=1000):
        """Get latest data points for plotting"""
        with self.data_lock:
            timestamps = list(self.timestamps)[-n_points:]
            lux_values = list(self.lux_values)[-n_points:]
            als_values = list(self.als_values)[-n_points:]
        return timestamps, lux_values, als_values
    
    def read_serial_data(self):
        """Background thread to read serial data"""
        while self.is_running:
            try:
                if self.serial_connection and self.serial_connection.in_waiting:
                    line = self.serial_connection.readline().decode('utf-8')
                    parsed_data = self.parse_serial_data(line)
                    
                    if parsed_data:
                        timestamp, als_raw, white_raw, lux = parsed_data
                        
                        with self.data_lock:
                            self.timestamps.append(timestamp)
                            self.als_values.append(als_raw)
                            self.white_values.append(white_raw)
                            self.lux_values.append(lux)
                            
                            # Add to analysis windows
                            self.window_1.append(lux)
                            self.window_2.append(lux)
                            self.window_3.append(lux)
                
                time.sleep(0.01)  # Small delay
            except Exception as e:
                print(f"Serial read error: {e}")
                time.sleep(1)
    
    def start(self):
        """Start the analyzer"""
        if self.connect_serial():
            self.is_running = True
            self.thread = threading.Thread(target=self.read_serial_data)
            self.thread.daemon = True
            self.thread.start()
            return True
        return False
    
    def stop(self):
        """Stop the analyzer"""
        self.is_running = False
        if self.serial_connection:
            self.serial_connection.close()

# Flask web application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'light_sensor_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize analyzer and report generator
analyzer = LightSensorAnalyzer()
report_generator = ReportGenerator(analyzer)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """API endpoint to get current data"""
    timestamps, lux_values, als_values = analyzer.get_latest_data()
    analysis = analyzer.analyze_windows()
    
    # Calculate FFT for both ALS and LUX data
    lux_fft_freqs, lux_fft_magnitude = analyzer.calculate_fft(lux_values)
    als_fft_freqs, als_fft_magnitude = analyzer.calculate_fft(als_values)
    
    return jsonify({
        'timestamps': timestamps,
        'lux_values': lux_values,
        'als_values': als_values,
        'analysis': analysis,
        'lux_fft': {
            'frequencies': lux_fft_freqs.tolist() if hasattr(lux_fft_freqs, 'tolist') else lux_fft_freqs,
            'magnitude': lux_fft_magnitude.tolist() if hasattr(lux_fft_magnitude, 'tolist') else lux_fft_magnitude
        },
        'als_fft': {
            'frequencies': als_fft_freqs.tolist() if hasattr(als_fft_freqs, 'tolist') else als_fft_freqs,
            'magnitude': als_fft_magnitude.tolist() if hasattr(als_fft_magnitude, 'tolist') else als_fft_magnitude
        }
    })

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@app.route('/api/start-recording', methods=['POST'])
def start_recording():
    """Start recording data for report generation"""
    try:
        data = request.get_json()
        duration_minutes = float(data.get('duration', 30))  # Default 30 minutes
        
        success, message = report_generator.start_recording(duration_minutes)
        
        return jsonify({
            'success': success,
            'message': message
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error starting recording: {str(e)}'
        }), 400

@app.route('/api/stop-recording', methods=['POST'])
def stop_recording():
    """Stop recording prematurely"""
    try:
        success, message = report_generator.stop_recording()
        
        return jsonify({
            'success': success,
            'message': message
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error stopping recording: {str(e)}'
        }), 400

@app.route('/api/recording-status')
def recording_status():
    """Get current recording status"""
    try:
        status = report_generator.get_recording_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'error': f'Error getting recording status: {str(e)}'
        }), 400

@app.route('/api/download-report/<filename>')
def download_report(filename):
    """Download generated report and clean up file afterward"""
    try:
        reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
        file_path = os.path.join(reports_dir, filename)
        
        if os.path.exists(file_path) and filename.endswith('.pdf'):
            # Send file
            response = send_file(file_path, as_attachment=True)
            
            # Schedule file cleanup after response is sent
            def cleanup_file():
                try:
                    time.sleep(1)  # Small delay to ensure file is sent
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Cleaned up report file: {filename}")
                except Exception as e:
                    print(f"Error cleaning up file {filename}: {e}")
            
            import threading
            cleanup_thread = threading.Thread(target=cleanup_file)
            cleanup_thread.daemon = True
            cleanup_thread.start()
            
            return response
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 400

def background_data_update():
    """Send real-time updates to connected clients"""
    while True:
        if analyzer.is_running:
            try:
                timestamps, lux_values, als_values = analyzer.get_latest_data()
                analysis = analyzer.analyze_windows()
                
                # Calculate FFT for both ALS and LUX data
                lux_fft_freqs, lux_fft_magnitude = analyzer.calculate_fft(lux_values)
                als_fft_freqs, als_fft_magnitude = analyzer.calculate_fft(als_values)
                
                # Create plotly charts
                fig_lux = go.Figure()
                fig_lux.add_trace(go.Scatter(x=timestamps, y=lux_values, 
                                           mode='lines', name='Lux'))
                fig_lux.update_layout(title='Light Intensity (Lux)', 
                                    xaxis_title='Time (ms)', 
                                    yaxis_title='Lux')
                
                fig_als = go.Figure()
                fig_als.add_trace(go.Scatter(x=timestamps, y=als_values, 
                                           mode='lines', name='ALS Raw'))
                fig_als.update_layout(title='ALS Raw Values', 
                                    xaxis_title='Time (ms)', 
                                    yaxis_title='Raw Value')
                
                # Create FFT charts
                fig_lux_fft = go.Figure()
                if hasattr(lux_fft_freqs, '__len__') and len(lux_fft_freqs) > 0:
                    fig_lux_fft.add_trace(go.Scatter(x=lux_fft_freqs, y=lux_fft_magnitude, 
                                                   mode='lines', name='LUX FFT'))
                    fig_lux_fft.update_layout(title='LUX FFT Analysis', 
                                            xaxis_title='Frequency (Hz)', 
                                            yaxis_title='Magnitude (dB)')
                
                fig_als_fft = go.Figure()
                if hasattr(als_fft_freqs, '__len__') and len(als_fft_freqs) > 0:
                    fig_als_fft.add_trace(go.Scatter(x=als_fft_freqs, y=als_fft_magnitude, 
                                                   mode='lines', name='ALS FFT'))
                    fig_als_fft.update_layout(title='ALS FFT Analysis', 
                                            xaxis_title='Frequency (Hz)', 
                                            yaxis_title='Magnitude (dB)')
                
                # Prepare data with size limits to prevent payload issues
                lux_fft_data = {
                    'frequencies': lux_fft_freqs.tolist()[:500] if hasattr(lux_fft_freqs, 'tolist') and len(lux_fft_freqs) > 0 else [],
                    'magnitude': lux_fft_magnitude.tolist()[:500] if hasattr(lux_fft_magnitude, 'tolist') and len(lux_fft_magnitude) > 0 else []
                }
                
                als_fft_data = {
                    'frequencies': als_fft_freqs.tolist()[:500] if hasattr(als_fft_freqs, 'tolist') and len(als_fft_freqs) > 0 else [],
                    'magnitude': als_fft_magnitude.tolist()[:500] if hasattr(als_fft_magnitude, 'tolist') and len(als_fft_magnitude) > 0 else []
                }
                
                # Get recording status for real-time updates
                recording_status = report_generator.get_recording_status()
                
                data = {
                    'lux_chart': json.dumps(fig_lux, cls=plotly.utils.PlotlyJSONEncoder),
                    'als_chart': json.dumps(fig_als, cls=plotly.utils.PlotlyJSONEncoder),
                    'lux_fft_chart': json.dumps(fig_lux_fft, cls=plotly.utils.PlotlyJSONEncoder),
                    'als_fft_chart': json.dumps(fig_als_fft, cls=plotly.utils.PlotlyJSONEncoder),
                    'analysis': analysis,
                    'lux_fft': lux_fft_data,
                    'als_fft': als_fft_data,
                    'recording_status': recording_status
                }
                
                socketio.emit('data_update', data)
                
            except Exception as e:
                print(f"Error in background_data_update: {e}")
                # Send minimal data on error
                socketio.emit('data_update', {'error': 'Data processing error'})
        
        time.sleep(1.0)  # Update every second instead of 0.01

if __name__ == '__main__':
    # Start the analyzer
    if analyzer.start():
        print("Light sensor analyzer started successfully!")
        
        # Start background data updates
        update_thread = threading.Thread(target=background_data_update)
        update_thread.daemon = True
        update_thread.start()
        
        # Run the web application
        print("Starting web server on http://localhost:5000")
        socketio.run(app, host='0.0.0.0', port=7001, debug=False)
    else:
        print("Failed to start light sensor analyzer!")