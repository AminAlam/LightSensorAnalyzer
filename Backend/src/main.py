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

        self.raw_data_csv_file_path = os.path.join(os.path.dirname(__file__), 'sensor_raw_data.csv')
        # delete the file if it exists
        if os.path.exists(self.raw_data_csv_file_path):
            os.remove(self.raw_data_csv_file_path)
        
        # Data storage
        self.timestamps = deque(maxlen=window_size)
        self.als_values = deque(maxlen=window_size)
        self.white_values = deque(maxlen=window_size)
        self.lux_values = deque(maxlen=window_size)
        self.photoresistor_values = deque(maxlen=window_size)
        
        # Current analysis window size (configurable from frontend)
        self.current_window_size_seconds = 2  # Default to 2 seconds
        
        self.serial_connection = None
        self.is_running = False
        self.data_lock = threading.Lock()
        
    def find_arduino_port(self):
        """Automatically find Arduino port"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(port.description)
            if 'Arduino' in port.description or 'Nano' in port.description:
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
            if len(parts) == 5:
                timestamp = float(parts[0])
                als_raw = int(parts[1])
                white_raw = int(parts[2])
                lux = float(parts[3])
                photoresistor = int(parts[4])
                return timestamp, als_raw, white_raw, lux, photoresistor
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
        
    @staticmethod
    def convert_to_list(data):
        if hasattr(data, 'tolist'):
            return data.tolist()
        return data
    
    @staticmethod
    def list_converter(func):
        def wrapper(self, data):
            freqs, magnitude = func(self, data)
            return (LightSensorAnalyzer.convert_to_list(freqs), 
                   LightSensorAnalyzer.convert_to_list(magnitude))
        return wrapper
    
    @list_converter
    def calc_fft_data_return_list(self, data):
        return self.calculate_fft(data)
    
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
        """Calculate rise and fall times of the light signal using 10-90% method"""
        if len(data) < 10:
            return 0, 0
            
        data = np.array(data)
        # Normalize data to 0-1 range for consistent thresholding
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Find peaks with minimum height and distance requirements
        pos_peaks_indices, _ = signal.find_peaks(data_norm)
        neg_peaks_indices, _ = signal.find_peaks(-data_norm)
        
        if len(pos_peaks_indices) < 2 or len(neg_peaks_indices) < 2:  # Need at least 2 peaks for meaningful measurement
            return 0, 0
            
        rise_times = []
        fall_times = []
        
        for peak_idx in pos_peaks_indices:

            # find closest negative peak to this peak which is before the peak
            neg_peaks_indices_pre_pos_peak = neg_peaks_indices[neg_peaks_indices <= peak_idx]
            neg_peaks_indices_post_pos_peak = neg_peaks_indices[neg_peaks_indices >= peak_idx]
            if len(neg_peaks_indices_pre_pos_peak) == 0 or len(neg_peaks_indices_post_pos_peak) == 0:
                continue
            closest_pre_neg_peak_idx = neg_peaks_indices_pre_pos_peak[-1]
            closest_post_neg_peak_idx = neg_peaks_indices_post_pos_peak[0]

            if closest_pre_neg_peak_idx is None or closest_post_neg_peak_idx is None:
                continue

            rise_time = (peak_idx - closest_pre_neg_peak_idx) / self.sampling_rate
            fall_time = (closest_post_neg_peak_idx - peak_idx) / self.sampling_rate
            rise_times.append(rise_time)
            fall_times.append(fall_time)
            

        # Calculate mean times, handling empty lists
        rise_time = np.mean(rise_times) * 1000 if rise_times else 0  # convert to ms
        fall_time = np.mean(fall_times) * 1000 if fall_times else 0  # convert to ms

        return rise_time, fall_time

    def analyze_current_window(self):
        """Analyze the current window size and return results"""
        try:
            # Get data for the current window size
            window_size_seconds = self.get_window_size()
            timestamps, lux_values, photoresistor_values = self.get_latest_data_with_photoresistor_for_window(window_size_seconds)
            
            if not photoresistor_values or len(photoresistor_values) < 10:
                # Return empty results if not enough data
                return {
                    'current_window': {
                        'size': 0,
                        'frequency': 0.0,
                        'duty_cycle': 0.0,
                        'period': 0.0,
                        'rise_time': 0.0,
                        'fall_time': 0.0
                    }
                }
            
            # Calculate analysis metrics
            rise_time, fall_time = self.calculate_rise_fall_time(photoresistor_values)
            
            results = {
                'current_window': {
                    'size': len(photoresistor_values),
                    'frequency': self.calculate_dominant_frequency(photoresistor_values),
                    'duty_cycle': self.calculate_duty_cycle(photoresistor_values),
                    'period': self.calculate_period(photoresistor_values),
                    'rise_time': rise_time,
                    'fall_time': fall_time
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Error in analyze_current_window: {e}")
            return {
                'current_window': {
                    'size': 0,
                    'frequency': 0.0,
                    'duty_cycle': 0.0,
                    'period': 0.0,
                    'rise_time': 0.0,
                    'fall_time': 0.0
                }
            }

    def get_latest_data_with_photoresistor_for_window(self, window_size_seconds):
        """Get latest data for the specified window size in seconds"""
        try:
            data = np.genfromtxt(self.raw_data_csv_file_path, delimiter=',', skip_header=1, usecols=(0,1,2,3,4))
        except:
            return [], [], []
        
        if len(data) == 0:
            return [], [], []
        
        # Extract columns
        timestamps = data[:, 0]  # First column
        lux_values = data[:, 3]  # Fourth column (LUX calculated)
        photoresistor_values = data[:, 4]  # Fifth column (photoresistor)

        # Get data for the specified window size
        last_timestamp = timestamps[-1]
        first_timestamp = last_timestamp - window_size_seconds * 1000  # Convert to milliseconds

        # Filter data within the time window
        mask = timestamps >= first_timestamp
        time_stamps_filtered = timestamps[mask]
        lux_values_filtered = lux_values[mask]
        photoresistor_values_filtered = photoresistor_values[mask]

        if len(time_stamps_filtered) == 0:
            return [], [], []

        # Interpolate to consistent sampling rate
        target_samples = int(window_size_seconds * self.sampling_rate)
        timestamps_interpolated = np.linspace(first_timestamp, last_timestamp, target_samples)
        lux_values_interpolated = np.interp(timestamps_interpolated, time_stamps_filtered, lux_values_filtered)
        photoresistor_values_interpolated = np.interp(timestamps_interpolated, time_stamps_filtered, photoresistor_values_filtered)

        return timestamps_interpolated.tolist(), lux_values_interpolated.tolist(), photoresistor_values_interpolated.tolist()
    
    def get_latest_data(self, seconds=2):
        """Get latest data points for plotting charts"""
        try:
            data = np.genfromtxt(self.raw_data_csv_file_path, delimiter=',', skip_header=1, usecols=(0,1,2,3,4))
        except:
            return [], [], [], []
        if len(data) == 0:
            return [], [], [], []
        
        # Get timestamps and values
        timestamps = data[:, 0]  # First column
        als_values = data[:, 1]  # Second column
        lux_values = data[:, 3]  # Fourth column
        photoresistor_values = data[:, 4]  # Fifth column
        
        last_timestamp = timestamps[-1]
        first_timestamp = last_timestamp - seconds*1000

        time_stamps = timestamps[timestamps >= first_timestamp]
        lux_values = lux_values[timestamps >= first_timestamp]
        als_values = als_values[timestamps >= first_timestamp]
        photoresistor_values = photoresistor_values[timestamps >= first_timestamp]

        # interpolate the data to the sampling rate
        timestamps_interpolated = np.linspace(first_timestamp, last_timestamp, seconds*self.sampling_rate)
        lux_values_interpolated = np.interp(timestamps_interpolated, time_stamps, lux_values)
        als_values_interpolated = np.interp(timestamps_interpolated, time_stamps, als_values)
        photoresistor_values_interpolated = np.interp(timestamps_interpolated, time_stamps, photoresistor_values)
            
        return timestamps_interpolated.tolist(), lux_values_interpolated.tolist(), als_values_interpolated.tolist(), photoresistor_values_interpolated.tolist()
    
    def get_latest_data_with_photoresistor(self, n_points=1000):
        """Get latest data points including photoresistor values for report generation"""
        try:
            data = np.genfromtxt(self.raw_data_csv_file_path, delimiter=',', skip_header=1, usecols=(0,1,2,3,4))
        except:
            return [], [], [], []
        if len(data) == 0:
            return [], [], [], []
        
        # Get the last n_points or all data if less than n_points
        if len(data) > n_points:
            data = data[-n_points:]

        seconds = int(n_points/self.sampling_rate)
        
        # Extract columns
        timestamps = data[:, 0]  # First column
        als_values = data[:, 1]  # Second column (ALS raw)
        lux_values = data[:, 3]  # Fourth column (LUX calculated)
        photoresistor_values = data[:, 4]  # Fifth column (photoresistor)

        last_timestamp = timestamps[-1]
        first_timestamp = last_timestamp - seconds*1000

        time_stamps = timestamps[timestamps >= first_timestamp]
        lux_values = lux_values[timestamps >= first_timestamp]
        als_values = als_values[timestamps >= first_timestamp]
        photoresistor_values = photoresistor_values[timestamps >= first_timestamp]

        timestamps_interpolated = np.linspace(first_timestamp, last_timestamp, seconds*self.sampling_rate)
        lux_values_interpolated = np.interp(timestamps_interpolated, time_stamps, lux_values)
        als_values_interpolated = np.interp(timestamps_interpolated, time_stamps, als_values)
        photoresistor_values_interpolated = np.interp(timestamps_interpolated, time_stamps, photoresistor_values)

        return timestamps_interpolated.tolist(), lux_values_interpolated.tolist(), als_values_interpolated.tolist(), photoresistor_values_interpolated.tolist()
    
    def read_serial_data(self):
        """Background thread to read serial data and write to CSV"""
        buffer = ""  # Buffer to store partial lines
        last_timestamp = None  # Track the last timestamp we processed
        
        # Create/clear CSV file with headers
        with open(self.raw_data_csv_file_path, 'w') as f:
            f.write('timestamp,als_raw,white_raw,lux,photoresistor\n')
        
        while self.is_running:
            try:
                if self.serial_connection and self.serial_connection.in_waiting:
                    # Read all available bytes
                    raw_data = self.serial_connection.read(self.serial_connection.in_waiting)

                    try:
                        # Decode bytes to string and add to buffer
                        buffer += raw_data.decode('utf-8')
                    except UnicodeDecodeError:
                        # If decode fails, clear buffer and continue
                        buffer = ""
                        continue
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        parsed_data = self.parse_serial_data(line)
                        
                        if parsed_data:
                            timestamp, als_raw, white_raw, lux, photoresistor = parsed_data

                            
                            # Skip if we've already processed this timestamp
                            if last_timestamp is not None and timestamp <= last_timestamp:
                                continue
                                
                            with self.data_lock:
                                # Update in-memory data structures
                                self.timestamps.append(timestamp)
                                self.als_values.append(als_raw)
                                self.white_values.append(white_raw)
                                self.lux_values.append(lux)
                                self.photoresistor_values.append(photoresistor)
                                
                                # Write to CSV file
                                with open(self.raw_data_csv_file_path, 'a') as f:
                                    f.write(f'{timestamp},{als_raw},{white_raw},{lux},{photoresistor}\n')
                                
                                # Read all lines and keep only recent ones
                                with open(self.raw_data_csv_file_path, 'r') as f:
                                    lines = f.readlines()
                                if len(lines) > 10000:

                                    # Keep only last 10 seconds of data in CSV
                                    current_time = timestamp
                                    cutoff_time = current_time - (10 * 1000)  # 10 seconds in milliseconds
                                    line_index = 0
                                    # Keep header and recent data
                                    for line in lines[1:]:
                                        try:
                                            line_timestamp = float(line.split(',')[0])
                                            if line_timestamp >= cutoff_time:
                                                line_index = lines.index(line)
                                                break
                                        except (ValueError, IndexError):
                                            continue
                                    
                                    # Write back only recent data
                                    with open(self.raw_data_csv_file_path, 'w') as f:
                                        f.writelines(lines[line_index:])
                                
                            last_timestamp = timestamp  # Update last processed timestamp
                
                time.sleep(0.002)  # Small delay
            except Exception as e:
                print(f"Serial read error: {e}")
                buffer = ""  # Clear buffer on error
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

    def set_window_size(self, size_seconds):
        """Set the current analysis window size in seconds"""
        with self.data_lock:
            self.current_window_size_seconds = max(1, min(300, size_seconds))  # Clamp between 1-300 seconds
            print(f"Analysis window size set to {self.current_window_size_seconds} seconds")
    
    def get_window_size(self):
        """Get the current analysis window size in seconds"""
        return self.current_window_size_seconds


# Flask web application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'light_sensor_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize analyzer and report generator
analyzer = LightSensorAnalyzer()
report_generator = ReportGenerator(analyzer, socketio)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """API endpoint to get current data"""
    try:
        timestamps, lux_values, als_values, photoresistor_values = analyzer.get_latest_data()
        analysis = analyzer.analyze_current_window()
        
        # Calculate FFT for both ALS and LUX data
        # lux_fft_freqs, lux_fft_magnitude = analyzer.calc_fft_data_return_list(lux_values)
        # als_fft_freqs, als_fft_magnitude = analyzer.calc_fft_data_return_list(als_values)
        
        # Ensure all data is properly serializable
        def ensure_serializable(data):
            """Convert numpy arrays and other non-serializable data to basic Python types"""
            if hasattr(data, 'tolist'):
                return data.tolist()
            elif isinstance(data, (list, tuple)):
                return [ensure_serializable(item) for item in data]
            elif isinstance(data, dict):
                return {key: ensure_serializable(value) for key, value in data.items()}
            elif isinstance(data, (int, float, str, bool, type(None))):
                return data
            else:
                return str(data)  # Convert anything else to string
        
        return jsonify({
            'timestamps': ensure_serializable(timestamps),
            'lux_values': ensure_serializable(lux_values),
            'als_values': ensure_serializable(als_values),
            'photoresistor_values': ensure_serializable(photoresistor_values),
            'analysis': ensure_serializable(analysis),
            # 'lux_fft': {
            #     'frequencies': ensure_serializable(lux_fft_freqs),
            #     'magnitude': ensure_serializable(lux_fft_magnitude)
            # },
            # 'als_fft': {
            #     'frequencies': ensure_serializable(als_fft_freqs),
            #     'magnitude': ensure_serializable(als_fft_magnitude)
            # }
        })
    except Exception as e:
        return jsonify({'error': f'Error getting data: {str(e)}'}), 500

@app.route('/api/charts')
def get_charts():
    """API endpoint to get chart data (separate from real-time updates)"""
    try:
        timestamps, lux_values, als_values = analyzer.get_latest_data(seconds=2)

        # Calculate FFT for both ALS and LUX data (no limiting - show all points)
        lux_fft_freqs, lux_fft_magnitude = analyzer.calc_fft_data_return_list(lux_values)
        als_fft_freqs, als_fft_magnitude = analyzer.calc_fft_data_return_list(als_values)
        
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
        
        return jsonify({
            'lux_chart': json.dumps(fig_lux, cls=plotly.utils.PlotlyJSONEncoder),
            'als_chart': json.dumps(fig_als, cls=plotly.utils.PlotlyJSONEncoder),
            'lux_fft_chart': json.dumps(fig_lux_fft, cls=plotly.utils.PlotlyJSONEncoder),
            'als_fft_chart': json.dumps(fig_als_fft, cls=plotly.utils.PlotlyJSONEncoder)
        })
    except Exception as e:
        return jsonify({'error': f'Error generating charts: {str(e)}'}), 500

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

@app.route('/api/chart-data')
def get_chart_data():
    """API endpoint to get raw chart data for frontend plotting"""
    try:
        timestamps, lux_values, als_values, photoresistor_values = analyzer.get_latest_data(seconds=1)

        # Calculate FFT for all signals (LUX, ALS, and Photoresistor)
        lux_fft_freqs, lux_fft_magnitude = analyzer.calc_fft_data_return_list(lux_values)
        als_fft_freqs, als_fft_magnitude = analyzer.calc_fft_data_return_list(als_values)
        photoresistor_fft_freqs, photoresistor_fft_magnitude = analyzer.calc_fft_data_return_list(photoresistor_values)
        
        # Ensure all data is properly serializable
        def ensure_serializable(data):
            """Convert numpy arrays and other non-serializable data to basic Python types"""
            if hasattr(data, 'tolist'):
                data = data.tolist()
            
            if isinstance(data, (list, tuple)):
                result = []
                for item in data:
                    clean_item = ensure_serializable(item)
                    # Handle NaN and Infinity values
                    if isinstance(clean_item, float):
                        if not (clean_item == clean_item):  # NaN check
                            clean_item = 0.0
                        elif clean_item == float('inf') or clean_item == float('-inf'):
                            clean_item = 0.0
                    result.append(clean_item)
                return result
            elif isinstance(data, dict):
                result = {}
                for key, value in data.items():
                    result[str(key)] = ensure_serializable(value)
                return result
            elif isinstance(data, (int, float)):
                # Handle NaN and Infinity values
                if isinstance(data, float):
                    if not (data == data):  # NaN check
                        return 0.0
                    elif data == float('inf') or data == float('-inf'):
                        return 0.0
                return data
            elif isinstance(data, (str, bool, type(None))):
                return data
            else:
                return str(data)  # Convert anything else to string
        
        return jsonify({
            'timestamps': ensure_serializable(timestamps),
            'lux_values': ensure_serializable(lux_values),
            'als_values': ensure_serializable(als_values),
            'photoresistor_values': ensure_serializable(photoresistor_values),
            'lux_fft': {
                'frequencies': ensure_serializable(lux_fft_freqs),
                'magnitude': ensure_serializable(lux_fft_magnitude)
            },
            'als_fft': {
                'frequencies': ensure_serializable(als_fft_freqs),
                'magnitude': ensure_serializable(als_fft_magnitude)
            },
            'photoresistor_fft': {
                'frequencies': ensure_serializable(photoresistor_fft_freqs),
                'magnitude': ensure_serializable(photoresistor_fft_magnitude)
            }
        })
    except Exception as e:
        return jsonify({'error': f'Error getting chart data: {str(e)}'}), 500

@app.route('/api/set-window-size', methods=['POST'])
def set_window_size():
    """API endpoint to set the analysis window size"""
    try:
        data = request.get_json()
        window_size = data.get('window_size', 2)
        
        # Validate window size
        if not isinstance(window_size, (int, float)) or window_size < 1 or window_size > 300:
            return jsonify({
                'success': False,
                'message': 'Window size must be between 1 and 300 seconds'
            }), 400
        
        # Set the window size
        analyzer.set_window_size(window_size)
        
        return jsonify({
            'success': True,
            'message': f'Window size set to {window_size} seconds',
            'current_window_size': analyzer.get_window_size()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error setting window size: {str(e)}'
        }), 500

def background_data_update():
    """Send real-time updates to connected clients"""
    while True:
        if analyzer.is_running:
            try:
                # Only send analysis data and recording status via WebSocket
                # Chart data will be fetched separately via HTTP
                analysis = analyzer.analyze_current_window()
                recording_status = report_generator.get_recording_status()
                
                # Enhanced data serialization with NaN/Infinity handling
                def ensure_serializable(data):
                    """Convert numpy arrays and other non-serializable data to basic Python types"""
                    if hasattr(data, 'tolist'):
                        data = data.tolist()
                    
                    if isinstance(data, (list, tuple)):
                        result = []
                        for item in data:
                            clean_item = ensure_serializable(item)
                            # Handle NaN and Infinity values
                            if isinstance(clean_item, float):
                                if not (clean_item == clean_item):  # NaN check
                                    clean_item = 0.0
                                elif clean_item == float('inf') or clean_item == float('-inf'):
                                    clean_item = 0.0
                            result.append(clean_item)
                        return result
                    elif isinstance(data, dict):
                        result = {}
                        for key, value in data.items():
                            result[str(key)] = ensure_serializable(value)
                        return result
                    elif isinstance(data, (int, float)):
                        # Handle NaN and Infinity values
                        if isinstance(data, float):
                            if not (data == data):  # NaN check
                                return 0.0
                            elif data == float('inf') or data == float('-inf'):
                                return 0.0
                        return data
                    elif isinstance(data, (str, bool, type(None))):
                        return data
                    else:
                        return str(data)  # Convert anything else to string
                
                # Prepare lightweight data - only analysis and status
                clean_analysis = ensure_serializable(analysis)
                clean_recording_status = ensure_serializable(recording_status)
                
                # Send only lightweight analysis data via WebSocket
                data = {
                    'analysis': clean_analysis,
                    'recording_status': clean_recording_status,
                    'update_charts': True  # Signal frontend to fetch chart data
                }
                
                # Emit lightweight data
                socketio.emit('data_update', data)
                
            except Exception as e:
                print(f"Error in background_data_update: {e}")
                import traceback
                traceback.print_exc()
                # Send minimal error data
                try:
                    socketio.emit('data_update', {
                        'error': 'Data processing error', 
                        'message': str(e),
                        'analysis': {'current_window': {'size': 0, 'frequency': 0, 'duty_cycle': 0, 'period': 0, 'rise_time': 0, 'fall_time': 0}}
                    })
                except Exception as emit_error:
                    print(f"Failed to emit error data: {emit_error}")
        
        time.sleep(0.1)  # 10Hz update rate for analysis data

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