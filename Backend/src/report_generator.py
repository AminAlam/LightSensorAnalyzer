import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI issues
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import threading
import time
from collections import deque
from scipy import signal
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io


class ReportGenerator:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.is_recording = False
        self.recording_duration = 0
        self.recording_start_time = None
        self.recorded_data = {
            'timestamps': [],
            'lux_values': [],
            'als_values': [],
            'recording_start': None,
            'recording_end': None
        }
        self.recording_thread = None
        self.last_generated_report = None
        self.is_generating_report = False
        
    def start_recording(self, duration_minutes):
        """Start recording data for the specified duration"""
        if self.is_recording:
            return False, "Recording already in progress"
            
        self.recording_duration = duration_minutes * 60  # Convert to seconds
        self.recording_start_time = time.time()
        self.is_recording = True
        
        # Clear previous recording data
        self.recorded_data = {
            'timestamps': [],
            'lux_values': [],
            'als_values': [],
            'recording_start': datetime.now(),
            'recording_end': None
        }
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_data)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        return True, f"Started recording for {duration_minutes} minutes"
    
    def _record_data(self):
        """Background thread to record data"""
        while self.is_recording and (time.time() - self.recording_start_time) < self.recording_duration:
            try:
                # Get current data from analyzer
                timestamps, lux_values, als_values = self.analyzer.get_latest_data(n_points=1000)
                
                if timestamps and lux_values and als_values:
                    # Store the latest data point
                    self.recorded_data['timestamps'].extend(timestamps)  # Get last 10 points
                    self.recorded_data['lux_values'].extend(lux_values)
                    self.recorded_data['als_values'].extend(als_values)
                
                time.sleep(0.01)  # Record every 10ms
                
            except Exception as e:
                print(f"Recording error: {e}")
                
        # Recording finished
        self.is_recording = False
        self.recorded_data['recording_end'] = datetime.now()
        
        # Automatically generate report when recording completes
        try:
            self.is_generating_report = True
            print("Starting automatic report generation...")
            
            import os
            reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
            success, result = self.generate_pdf_report(reports_dir)
            if success:
                self.last_generated_report = result
                print(f"Recording completed and report generated: {os.path.basename(result)}")
            else:
                print(f"Recording completed but report generation failed: {result}")
                self.last_generated_report = None
        except Exception as e:
            print(f"Error auto-generating report: {e}")
            self.last_generated_report = None
        finally:
            self.is_generating_report = False
        
        print("Recording completed")
    
    def stop_recording(self):
        """Stop recording prematurely"""
        if self.is_recording:
            self.is_recording = False
            self.recorded_data['recording_end'] = datetime.now()
            
            # Start report generation immediately when manually stopped
            try:
                self.is_generating_report = True
                print("Recording stopped manually - Starting report generation...")
                
                import os
                reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
                success, result = self.generate_pdf_report(reports_dir)
                if success:
                    self.last_generated_report = result
                    print(f"Report generated after manual stop: {os.path.basename(result)}")
                else:
                    print(f"Report generation failed after manual stop: {result}")
                    self.last_generated_report = None
            except Exception as e:
                print(f"Error generating report after manual stop: {e}")
                self.last_generated_report = None
            finally:
                self.is_generating_report = False
                
            return True, "Recording stopped and report generated"
        elif self.is_generating_report:
            return False, "Recording stopped, report generation in progress"
        return False, "No recording in progress"
    
    def get_recording_status(self):
        """Get current recording status"""
        if not self.is_recording:
            return {
                'is_recording': False,
                'elapsed_time': 0,
                'remaining_time': 0,
                'progress': 0,
                'is_generating_report': self.is_generating_report,
                'report_ready': self.last_generated_report is not None and not self.is_generating_report,
                'report_filename': os.path.basename(self.last_generated_report) if self.last_generated_report else None
            }
            
        elapsed = time.time() - self.recording_start_time
        remaining = max(0, self.recording_duration - elapsed)
        progress = min(100, (elapsed / self.recording_duration) * 100)
        
        return {
            'is_recording': True,
            'elapsed_time': elapsed,
            'remaining_time': remaining,
            'progress': progress,
            'is_generating_report': False,
            'report_ready': False,
            'report_filename': None
        }
    
    def _extract_signal_period(self, data, timestamps):
        """Extract a single period from the signal"""
        if len(data) < 20:
            return data, timestamps
            
        data_array = np.array(data)
        
        # Find peaks to determine period
        peaks_indices, _ = signal.find_peaks(data_array, height=np.mean(data_array))
        
        if len(peaks_indices) < 2:
            # If no clear peaks, return first 2 seconds worth of data
            target_samples = min(len(data), int(2 * self.analyzer.sampling_rate))
            return data[:target_samples], timestamps[:target_samples]
        
        # Use the distance between first two peaks as one period
        period_start = peaks_indices[0]
        period_end = peaks_indices[1] if len(peaks_indices) > 1 else min(len(data), peaks_indices[0] + int(self.analyzer.sampling_rate))
        
        return data[period_start:period_end], timestamps[period_start:period_end]
    
    def _calculate_average_period(self, data):
        """Calculate average shape of signal periods with standard deviation"""
        data_array = np.array(data)
        
        # Find all peaks
        peaks_indices, _ = signal.find_peaks(data_array, height=np.mean(data_array))
        
        if len(peaks_indices) < 3:
            return data_array, np.zeros_like(data_array)
        
        # Extract periods between consecutive peaks
        periods = []
        periods_length = []
        
        for i in range(len(peaks_indices) - 1):
            period_data = data_array[peaks_indices[i]:peaks_indices[i+1]]
            periods.append(period_data)
            periods_length.append(len(period_data))
        
        avg_period_length = int(np.mean(periods_length))
        # Normalize all periods to the same length
        normalized_periods = []
        print(f'avg_period_length: {avg_period_length}, found {len(periods)} periods')
        for period in periods:
            if len(period) != avg_period_length:
                # Resample to min_period_length
                indices = np.linspace(0, len(period)-1, avg_period_length, dtype=int)
                normalized_periods.append(period[indices])
        
        # Calculate mean and std
        periods_array = np.array(normalized_periods)
        mean_period = np.mean(periods_array, axis=0)
        std_period = np.std(periods_array, axis=0)
        
        return mean_period, std_period
        
    def _create_analysis_table(self):
        """Create analysis table for the report"""
        if not self.recorded_data['lux_values']:
            return [['No data recorded', '', '']]
            
        # Calculate analysis on recorded data
        lux_data = self.recorded_data['lux_values']
        als_data = self.recorded_data['als_values']
        
        # Calculate metrics for multiple segments to get std
        segment_size = max(100, len(lux_data) // 10)  # Use 10 segments or minimum 100 samples
        lux_segments = [lux_data[i:i+segment_size] for i in range(0, len(lux_data), segment_size) if len(lux_data[i:i+segment_size]) >= segment_size//2]
        als_segments = [als_data[i:i+segment_size] for i in range(0, len(als_data), segment_size) if len(als_data[i:i+segment_size]) >= segment_size//2]
        
        # Calculate metrics for each segment
        lux_frequencies = [self.analyzer.calculate_dominant_frequency(seg) for seg in lux_segments]
        lux_duty_cycles = [self.analyzer.calculate_duty_cycle(seg) for seg in lux_segments]
        lux_periods = [self.analyzer.calculate_period(seg) for seg in lux_segments]
        lux_rise_times = []
        lux_fall_times = []
        
        for seg in lux_segments:
            rise_time, fall_time = self.analyzer.calculate_rise_fall_time(seg)
            lux_rise_times.append(rise_time)
            lux_fall_times.append(fall_time)
        
        als_frequencies = [self.analyzer.calculate_dominant_frequency(seg) for seg in als_segments]
        als_duty_cycles = [self.analyzer.calculate_duty_cycle(seg) for seg in als_segments]
        als_periods = [self.analyzer.calculate_period(seg) for seg in als_segments]
        als_rise_times = []
        als_fall_times = []
        
        for seg in als_segments:
            rise_time, fall_time = self.analyzer.calculate_rise_fall_time(seg)
            als_rise_times.append(rise_time)
            als_fall_times.append(fall_time)
        
        # Calculate means and stds
        lux_freq_mean, lux_freq_std = np.mean(lux_frequencies), np.std(lux_frequencies)
        lux_duty_mean, lux_duty_std = np.mean(lux_duty_cycles), np.std(lux_duty_cycles)
        lux_period_mean, lux_period_std = np.mean(lux_periods), np.std(lux_periods)
        lux_rise_mean, lux_rise_std = np.mean(lux_rise_times), np.std(lux_rise_times)
        lux_fall_mean, lux_fall_std = np.mean(lux_fall_times), np.std(lux_fall_times)
        
        als_freq_mean, als_freq_std = np.mean(als_frequencies), np.std(als_frequencies)
        als_duty_mean, als_duty_std = np.mean(als_duty_cycles), np.std(als_duty_cycles)
        als_period_mean, als_period_std = np.mean(als_periods), np.std(als_periods)
        als_rise_mean, als_rise_std = np.mean(als_rise_times), np.std(als_rise_times)
        als_fall_mean, als_fall_std = np.mean(als_fall_times), np.std(als_fall_times)
        
        # Basic statistics
        lux_mean = np.mean(lux_data)
        lux_std = np.std(lux_data)
        lux_min = np.min(lux_data)
        lux_max = np.max(lux_data)
        
        als_mean = np.mean(als_data)
        als_std = np.std(als_data)
        als_min = np.min(als_data)
        als_max = np.max(als_data)
        
        table_data = [
            ['Metric', 'LUX Signal (Mean ± Std)', 'ALS Signal (Mean ± Std)'],
            ['Mean Value', f'{lux_mean:.2f}', f'{als_mean:.2f}'],
            ['Standard Deviation', f'{lux_std:.2f}', f'{als_std:.2f}'],
            ['Minimum Value', f'{lux_min:.2f}', f'{als_min:.2f}'],
            ['Maximum Value', f'{lux_max:.2f}', f'{als_max:.2f}'],
            ['Dominant Frequency (Hz)', f'{lux_freq_mean:.2f} ± {lux_freq_std:.2f}', f'{als_freq_mean:.2f} ± {als_freq_std:.2f}'],
            ['Duty Cycle (%)', f'{lux_duty_mean:.2f} ± {lux_duty_std:.2f}', f'{als_duty_mean:.2f} ± {als_duty_std:.2f}'],
            ['Period (ms)', f'{lux_period_mean:.2f} ± {lux_period_std:.2f}', f'{als_period_mean:.2f} ± {als_period_std:.2f}'],
            ['Rise Time (ms)', f'{lux_rise_mean:.2f} ± {lux_rise_std:.2f}', f'{als_rise_mean:.2f} ± {als_rise_std:.2f}'],
            ['Fall Time (ms)', f'{lux_fall_mean:.2f} ± {lux_fall_std:.2f}', f'{als_fall_mean:.2f} ± {als_fall_std:.2f}'],
            ['Total Samples', str(len(lux_data)), str(len(als_data))],
            ['Recording Duration', f'{(self.recorded_data["recording_end"] - self.recorded_data["recording_start"]).total_seconds():.1f} seconds', '']
        ]
        
        return table_data
    
    def generate_pdf_report(self, output_dir="reports"):
        """Generate comprehensive PDF report"""
        if not self.recorded_data['lux_values']:
            return False, "No recorded data available"
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"light_sensor_report_{timestamp}.pdf"
        filepath = os.path.join(output_dir, filename)
        
        # try:
        # Create PDF document
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        story.append(Paragraph("Light Sensor Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Report information
        info_style = styles['Normal']
        info_text = f"""
        <b>Report Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br/>
        <b>Recording Start:</b> {self.recorded_data['recording_start'].strftime("%Y-%m-%d %H:%M:%S")}<br/>
        <b>Recording End:</b> {self.recorded_data['recording_end'].strftime("%Y-%m-%d %H:%M:%S")}<br/>
        <b>Recording Duration:</b> {(self.recorded_data['recording_end'] - self.recorded_data['recording_start']).total_seconds():.1f} seconds<br/>
        <b>Total Samples:</b> {len(self.recorded_data['lux_values'])}
        """
        story.append(Paragraph(info_text, info_style))
        story.append(Spacer(1, 20))
        
        # Create and save plots
        plot_files = self._create_plots(output_dir, timestamp)
        
        # Add plots to PDF
        story.append(Paragraph("Signal Overview (2-second snapshot)", styles['Heading2']))
        if plot_files['snapshot']:
            story.append(Image(plot_files['snapshot'], width=6*inch, height=4*inch))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Average Period Shape with Error Bounds", styles['Heading2']))
        if plot_files['average_period']:
            story.append(Image(plot_files['average_period'], width=6*inch, height=4*inch))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Frequency Domain Analysis (FFT)", styles['Heading2']))
        if plot_files['fft_analysis']:
            story.append(Image(plot_files['fft_analysis'], width=6*inch, height=4*inch))
        story.append(Spacer(1, 12))
        
        # Add measurements table
        story.append(Paragraph("Signal Measurements", styles['Heading2']))
        table_data = self._create_analysis_table()
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        story.append(table)
        
        # Build PDF
        doc.build(story)
        
        # Clean up temporary plot files
        for plot_file in plot_files.values():
            if plot_file and os.path.exists(plot_file):
                try:
                    os.remove(plot_file)
                except:
                    pass
        
        return True, filepath
            
        # except Exception as e:
        #     return False, f"Error generating PDF: {str(e)}"
    
    def _create_plots(self, output_dir, timestamp):
            """Create matplotlib plots for the PDF report"""
            plot_files = {
                'snapshot': None,
                'average_period': None,
                'fft_analysis': None
            }
        
        # try:
            lux_data = np.array(self.recorded_data['lux_values'])
            als_data = np.array(self.recorded_data['als_values'])
            timestamps = np.array(self.recorded_data['timestamps'])
            
            if len(lux_data) == 0:
                return plot_files
            
            # Plot 1: 2-second snapshot
            snapshot_samples = min(len(lux_data), int(2 * self.analyzer.sampling_rate))
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle('Signal Snapshot (2 seconds)', fontsize=16, fontweight='bold')
            
            time_axis = np.arange(snapshot_samples) / self.analyzer.sampling_rate
            
            ax1.plot(time_axis, lux_data[:snapshot_samples], 'b-', linewidth=1.5, label='LUX')
            ax1.set_ylabel('LUX Value', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_title('LUX Signal')
            
            ax2.plot(time_axis, als_data[:snapshot_samples], 'r-', linewidth=1.5, label='ALS')
            ax2.set_ylabel('ALS Raw Value', fontsize=12)
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_title('ALS Signal')
            
            plt.tight_layout()
            snapshot_file = os.path.join(output_dir, f'snapshot_{timestamp}.png')
            plt.savefig(snapshot_file, dpi=150, bbox_inches='tight')
            plt.close()
            plot_files['snapshot'] = snapshot_file
            
            # Plot 2: Average period with error bounds
            mean_lux, std_lux = self._calculate_average_period(lux_data)
            mean_als, std_als = self._calculate_average_period(als_data)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle('Average Period Shape with Standard Deviation', fontsize=16, fontweight='bold')
            
            time_avg = np.arange(len(mean_lux)) / self.analyzer.sampling_rate
            
            ax1.plot(time_avg, mean_lux, 'b-', linewidth=2, label='Mean LUX')
            ax1.fill_between(time_avg, mean_lux - std_lux, mean_lux + std_lux, 
                           alpha=0.3, color='blue', label='±1 STD')
            ax1.set_ylabel('LUX Value', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_title('Average LUX Period')
            
            time_avg_als = np.arange(len(mean_als)) / self.analyzer.sampling_rate
            ax2.plot(time_avg_als, mean_als, 'r-', linewidth=2, label='Mean ALS')
            ax2.fill_between(time_avg_als, mean_als - std_als, mean_als + std_als, 
                           alpha=0.3, color='red', label='±1 STD')
            ax2.set_ylabel('ALS Raw Value', fontsize=12)
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_title('Average ALS Period')
            
            plt.tight_layout()
            avg_period_file = os.path.join(output_dir, f'avg_period_{timestamp}.png')
            plt.savefig(avg_period_file, dpi=150, bbox_inches='tight')
            plt.close()
            plot_files['average_period'] = avg_period_file
            
            # Plot 3: Frequency Domain Analysis (FFT)
            lux_fft_freqs, lux_fft_magnitude = self.analyzer.calculate_fft(lux_data)
            als_fft_freqs, als_fft_magnitude = self.analyzer.calculate_fft(als_data)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle('Frequency Domain Analysis (FFT)', fontsize=16, fontweight='bold')
            
            if len(lux_fft_freqs) > 0 and len(lux_fft_magnitude) > 0:
                ax1.plot(lux_fft_freqs, lux_fft_magnitude, 'b-', linewidth=1.5, label='LUX FFT')
                ax1.set_ylabel('Magnitude (dB)', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                ax1.set_title('LUX Signal - FFT Analysis')
                ax1.set_xlim(0, min(max(lux_fft_freqs), 100))  # Limit to 100Hz for better visibility
            
            if len(als_fft_freqs) > 0 and len(als_fft_magnitude) > 0:
                ax2.plot(als_fft_freqs, als_fft_magnitude, 'r-', linewidth=1.5, label='ALS FFT')
                ax2.set_ylabel('Magnitude (dB)', fontsize=12)
                ax2.set_xlabel('Frequency (Hz)', fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                ax2.set_title('ALS Signal - FFT Analysis')
                ax2.set_xlim(0, min(max(als_fft_freqs), 100))  # Limit to 100Hz for better visibility
            
            plt.tight_layout()
            fft_file = os.path.join(output_dir, f'fft_{timestamp}.png')
            plt.savefig(fft_file, dpi=150, bbox_inches='tight')
            plt.close()
            plot_files['fft_analysis'] = fft_file
            
        # except Exception as e:
        #     print(f"Error creating plots: {e}")
        
            return plot_files 