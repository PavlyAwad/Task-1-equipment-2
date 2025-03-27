import pandas as pd
import numpy as np
import wfdb
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                            QHBoxLayout, QWidget, QFrame, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt, QUrl
from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg
from PyQt5.QtMultimedia import QSoundEffect
from scipy.signal import find_peaks, butter, filtfilt
import os
import glob
import sys
import numpy as np
from scipy.signal import welch
import scipy.signal as signal

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch

class ECGProcessor:
    def __init__(self, fs=500):
        self.fs = fs
        self.filter_lowcut = 0.5
        self.filter_highcut = 45
        self.filter_order = 3

        # Create filter coefficients
        nyquist = 0.5 * fs
        low = self.filter_lowcut / nyquist
        high = self.filter_highcut / nyquist
        self.b, self.a = butter(self.filter_order, [low, high], btype='band')

    def process_ecg(self, signal):
        """Filter and normalize ECG signal"""
        filtered = filtfilt(self.b, self.a, signal)
        normalized = (filtered - np.mean(filtered)) / (np.max(filtered) - np.min(filtered))
        return normalized

    def detect_r_peaks(self, signal):
        """Detect R-peaks using Pan-Tompkins algorithm"""
        diff_signal = np.diff(signal)
        squared_signal = diff_signal ** 2
        integrated_signal = np.convolve(squared_signal, np.ones(int(0.150 * self.fs)) / int(0.150 * self.fs), mode='same')
        peaks, _ = find_peaks(integrated_signal, height=np.mean(integrated_signal), distance=int(0.3 * self.fs))
        return peaks

    def calculate_hr(self, peaks):
        """Calculate heart rate from R-peaks"""
        if len(peaks) < 2:
            return 0
        rr_intervals = np.diff(peaks) / self.fs * 1000  # in ms
        hr = 60000 / np.mean(rr_intervals)  # beats per minute
        return hr



    def detect_arrhythmia_type(self, signal, peaks):
        """Enhanced arrhythmia detection with debugging"""
        if len(peaks) < 3:
            return "Normal Sinus Rhythm"

        # RR Intervals (in ms)
        rr_intervals = np.diff(peaks) / self.fs * 1000
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals)
        hr = 60000 / mean_rr  # Heart rate in bpm

        print(f"\n=== RR INTERVAL ANALYSIS ===")
        print(f"Mean RR: {mean_rr:.2f} ms, Std RR: {std_rr:.2f} ms")
        print(f"RMSSD: {rmssd:.2f}, NN50: {nn50:.2f}, HR: {hr:.2f} bpm")

        # Power Spectral Density (PSD) for AFib
        f, psd = welch(rr_intervals, fs=1, nperseg=len(rr_intervals))
        lf_power = np.sum(psd[(f >= 0.04) & (f <= 0.15)])
        hf_power = np.sum(psd[(f >= 0.15) & (f <= 0.4)])

        lf_hf_ratio = lf_power / (hf_power + 1e-6)  # Avoid division by zero

        print(f"\n=== POWER SPECTRAL DENSITY ===")
        print(f"LF Power: {lf_power:.4f}, HF Power: {hf_power:.4f}, LF/HF Ratio: {lf_hf_ratio:.4f}")

        # QRS Amplitude Calculation
        qrs_amplitudes = [
            np.max(signal[max(0, peak-10):min(len(signal), peak+10)]) -
            np.min(signal[max(0, peak-10):min(len(signal), peak+10)])
             for peak in peaks
                    ]
        avg_amplitude = np.mean(qrs_amplitudes)

        print(f"\n=== QRS AMPLITUDE ===")
        print(f"Avg QRS Amplitude: {avg_amplitude:.4f}")

        # 1. Low Voltage QRS
        #if avg_amplitude < 0.5 and 50 < hr < 100 and std_rr / mean_rr < 0.1:
            #return "Sinus Rhythm with Low Voltage"

        # # 2. Sinus Arrhythmia
        # if 0.1 < std_rr / mean_rr < 0.2 and 50 < hr < 100 and self._has_respiratory_pattern(rr_intervals):
        #     return "Sinus Arrhythmia"

        # 3. Atrial Fibrillation (AFib) Detection
        print("\n=== AFib Debugging ===")
        print(f"std_rr/mean_rr: {std_rr/mean_rr:.2f} (should be > 0.25)")
        print(f"RMSSD: {rmssd:.2f} (should be > 50)")
        print(f"NN50: {nn50:.2f} (should be > 0.5)")
        print(f"LF/HF Ratio: {lf_hf_ratio:.2f} (should be < 1.5)")
        p_wave_absent = self._p_wave_absence(signal, peaks)
        print(f"P-wave Absence: {p_wave_absent} (should be True)")

        afib_criteria = (
            std_rr / mean_rr > 0.25 and  # High RR interval variability
            rmssd > 50 and  # High beat-to-beat variability
            nn50 > 0.5 and  # More than 50% of RR intervals differ by >50ms
            lf_hf_ratio < 1.5 and  # Low LF/HF ratio
            self._p_wave_absence(signal, peaks)  # Absence of P-waves
        )

        if afib_criteria:
            print("\nAFib Detected!")
            return "Atrial Fibrillation"

        # 4. PVC Detection

        median_rr = np.median(rr_intervals)
        short_rr_indices = [i for i, rr in enumerate(rr_intervals) if rr < 0.75 * median_rr]

        # print(f"\n=== PVC DETECTION ===")
        # print(f"Median RR: {median_rr:.2f} ms")
        # print(f"Short RR Indices: {short_rr_indices}")

        for i in short_rr_indices:
            if i + 1 < len(rr_intervals):  # Ensure there's a next RR interval
                long_rr = rr_intervals[i + 1]  # Check the compensatory pause

                # print(f"Checking PVC at index {i}: Short RR = {rr_intervals[i]:.2f}, Long RR = {long_rr:.2f}")

                if long_rr > 1.4 * median_rr:  # Long pause after short beat
                    pvc_candidate = peaks[i + 1]

                    # Compare QRS duration
                    normal_qrs = np.median([self._qrs_duration(signal, [p]) for p in peaks[:-1]])
                    pvc_qrs = self._qrs_duration(signal, [pvc_candidate])

                    # print(f"Normal QRS Duration: {normal_qrs:.2f}, PVC QRS Duration: {pvc_qrs:.2f}")

                    if pvc_qrs > 1.2 * normal_qrs:
                        # print("PVC Detected!")
                        return "Premature Ventricular Contractions (PVC)"

        # 5. Bradycardia/Tachycardia
        if hr < 50:
            return "Sinus Bradycardia"
        elif hr > 100:
            return "Sinus Tachycardia"

        return "Normal Sinus Rhythm"


    #def _has_respiratory_pattern(self, rr_intervals):
      #  """Detect cyclic variation in sinus arrhythmia"""
       # if len(rr_intervals) < 6:
        #    return False
        #peaks, _ = find_peaks(rr_intervals, distance=3, prominence=np.std(rr_intervals)/2)
        #return len(peaks) >= 2





    def _p_wave_absence(self, signal, peaks):
        """Detect P-wave absence using bandpass filtering (4-8 Hz) and template matching."""
        if len(peaks) < 4:
            return False

        # Bandpass filter for P-wave (4-8 Hz)
        nyquist = 0.5 * self.fs
        low, high = 4 / nyquist, 8 / nyquist
        b, a = butter(3, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)

        p_wave_energies = []

        for i in range(1, len(peaks)):
            # Extract segment before the QRS complex (possible P-wave)
            start = max(0, peaks[i] - int(0.3 * self.fs))  # 300ms before QRS
            end = peaks[i] - int(0.05 * self.fs)  # 50ms before QRS

            p_wave_segment = filtered_signal[start:end]

            # Compute energy of the P-wave region
            p_wave_energy = np.sum(p_wave_segment ** 2)
            p_wave_energies.append(p_wave_energy)

        # AFib: If P-wave energy is consistently low (no distinct peaks)
        avg_energy = np.mean(p_wave_energies)
        return avg_energy < 0.01  # Threshold to detect absence of P-waves


    def _qrs_duration(self, signal, peaks):
        """Estimate QRS duration more precisely"""
        durations = []
        for peak in peaks:
            if peak > 20 and peak < len(signal) - 20:
                q_index = np.argmax(np.abs(np.diff(signal[peak-20:peak])))
                s_index = np.argmax(np.abs(np.diff(signal[peak:peak+20])))
                durations.append((s_index - q_index) * (1000 / self.fs))
        return np.mean(durations) if durations else 0



class AlarmManager:
    def __init__(self):
        self.alarms = {
            "Heart Rate": {"critical": (40, 120), "warning": (60, 100)},
            "Oxygen Saturation": {"critical": (90, None), "warning": (95, None)},
            "Body Temperature": {"critical": (35.0, 39.0), "warning": (36, 37.6)},
            "BP_Sys": {"critical": (80, 140), "warning": (90, 120)},
            "BP_Dia": {"critical": (50, 90), "warning": (60, 80)},
            "Respiratory Rate": {"critical": (10, 30), "warning": (12, 20)},
            "Arrhythmia": {"critical": True}
        }

    def check_alarm(self, name, value):
        if name == "BP":
            try:
                sys_val, dia_val = map(int, value.split("/"))
                sys_alarm = self._check_single("BP_Sys", sys_val)
                dia_alarm = self._check_single("BP_Dia", dia_val)
                return max(sys_alarm, dia_alarm)
            except:
                return "NORMAL"
        elif name == "Arrhythmia":
            # Different alarm levels for different arrhythmias
            if value in ["Tachycardic Atrial Fibrillation", "Atrial Fibrillation"]:
                return "CRITICAL"
            elif value in ["Sinus Bradycardia", "Sinus Tachycardia", "Premature Ventricular Contractions (PVC)"]:
                return "WARNING"
            return "NORMAL"
        else:
            try:
                return self._check_single(name, float(value))
            except:
                return "NORMAL"

    def _check_single(self, name, value):
        thresholds = self.alarms.get(name, {})
        crit_low, crit_high = thresholds.get("critical", (None, None))
        warn_low, warn_high = thresholds.get("warning", (None, None))

        if (crit_low is not None and value <= crit_low) or (crit_high is not None and value >= crit_high):
            return "CRITICAL"
        elif (warn_low is not None and value <= warn_low) or (warn_high is not None and value >= warn_high):
            return "WARNING"
        return "NORMAL"


class PatientMonitor(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize components
        self.ecg_processor = ECGProcessor()
        self.alarm_manager = AlarmManager()
        self.setup_audio()
        self.load_data()
        self.setup_ui()
        self.setup_timers()

    def setup_audio(self):
        """Initialize audio components"""
        self.alarm_sound = QSoundEffect()
        if os.path.exists("alarm.wav"):
            self.alarm_sound.setSource(QUrl.fromLocalFile("alarm.wav"))
            self.alarm_sound.setVolume(0.5)

    def load_data(self):
        """Load patient data and ECG records"""
        # Load patient vitals
        try:
            self.dataset = pd.read_csv("human_vital_signs_dataset_2024.csv")
            required_columns = ['Heart Rate', 'Respiratory Rate', 'Body Temperature',
                              'Oxygen Saturation', 'Systolic Blood Pressure',
                              'Diastolic Blood Pressure']
            for col in required_columns:
                if col not in self.dataset.columns:
                    raise ValueError(f"Missing required column: {col}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.create_simulated_data()

        # Load ECG records
        self.ecg_records = []
        try:
            record_files = glob.glob('Task3_Data/00349_hr.dat')
            for file in record_files:
                self.load_ecg_record(file)
        except Exception as e:
            print(f"Error loading ECG records: {e}")
            self.create_simulated_ecg()

        self.current_record = 0
        self.current_index = 0
        self.ecg_index = 0

    #def create_simulated_data(self):
        """Create simulated patient data"""
     #   self.dataset = pd.DataFrame({
      #      "Heart Rate": np.random.randint(60, 100, 100),
       #     "Respiratory Rate": np.random.randint(12, 20, 100),
        #    "Body Temperature": np.random.uniform(36.0, 37.5, 100),
        #    "Oxygen Saturation": np.random.uniform(95.0, 100.0, 100),
         #   "Systolic Blood Pressure": np.random.randint(110, 140, 100),
          #  "Diastolic Blood Pressure": np.random.randint(70, 90, 100),
           # "Risk Category": np.random.choice(["High Risk", "Low Risk"], 100)
        #})

    def load_ecg_record(self, file):
        """Load a single ECG record"""
        record_name = os.path.splitext(file)[0]
        record = wfdb.rdrecord(record_name)
        if record.p_signal is not None:
            ecg_signal = record.p_signal[:, 1] if record.p_signal.shape[1] > 1 else record.p_signal[:, 0]
            self.ecg_records.append(ecg_signal)

    #def create_simulated_ecg(self):
        """Create simulated ECG data"""
     #   t = np.linspace(0, 10, 5000)
      #  ecg_signal = np.sin(2*np.pi*1.2*t) + 0.25*np.sin(2*np.pi*5*t) + 0.1*np.random.randn(len(t))
       # self.ecg_records.append(ecg_signal)

    def setup_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Advanced Patient Monitor")
        self.setGeometry(100, 100, 1200, 800)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # ECG Plot Panel
        self.setup_ecg_panel(main_layout)

        # Vitals Panel
        self.setup_vitals_panel(main_layout)

        self.central_widget.setLayout(main_layout)

        # ECG plot settings
        self.fs = 500
        self.duration = 5
        self.ecg_window_size = self.fs * self.duration
        self.time_axis = np.linspace(0, self.duration, self.ecg_window_size)
        self.curve = self.ecg_plot.plot(self.time_axis,y=np.zeros(self.ecg_window_size), pen='#00FF00')

    def setup_ecg_panel(self, main_layout):
        """Setup the ECG display panel"""
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        self.ecg_label = QLabel("ECG Signal")
        self.ecg_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.ecg_label.setStyleSheet("color: white;")
        left_panel.addWidget(self.ecg_label)

        self.ecg_plot = pg.PlotWidget()
        self.ecg_plot.setBackground('k')
        self.ecg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ecg_plot.setLabel('left', 'Amplitude (mV)')
        # self.ecg_plot.setLabel('bottom', 'Time (s)')
        self.ecg_plot.setYRange(-2, 2)
        self.ecg_plot.showAxis('bottom', show=False)

        left_panel.addWidget(self.ecg_plot)

        main_layout.addLayout(left_panel, 2)

    def setup_vitals_panel(self, main_layout):
        """Setup the vitals display panel"""
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)

        self.vital_labels = {}
        self.units = {
            "Heart Rate": "bpm",
            "Oxygen Saturation": "%",
            "Body Temperature": "°C",
            "BP": "mmHg",
            "Respiratory Rate": "breaths/min",
            "Arrhythmia": ""
        }

        self.colors = {
            "NORMAL": "#00FF00",
            "WARNING": "#FFA500",
            "CRITICAL": "#FF0000"
        }

        for vital in ["Heart Rate", "Oxygen Saturation", "Body Temperature",
                     "BP", "Respiratory Rate", "Arrhythmia"]:
            self.add_vital_display(vital, right_panel)

        right_panel.addStretch()
        main_layout.addLayout(right_panel, 1)

    def add_vital_display(self, vital, panel):
        """Add a single vital sign display to the panel"""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("background-color: #222; border-radius: 5px;")

        frame_layout = QVBoxLayout()
        frame_layout.setContentsMargins(10, 10, 10, 10)
        frame_layout.setSpacing(5)

        label = QLabel(vital)
        label.setFont(QFont("Arial", 12, QFont.Bold))
        label.setStyleSheet("color: #EEE;")

        value_label = QLabel("--")
        value_label.setFont(QFont("Arial", 28, QFont.Bold))
        value_label.setAlignment(Qt.AlignCenter)

        status_label = QLabel("NORMAL")
        status_label.setFont(QFont("Arial", 10))
        status_label.setAlignment(Qt.AlignCenter)

        frame_layout.addWidget(label)
        frame_layout.addWidget(value_label)
        frame_layout.addWidget(status_label)
        frame.setLayout(frame_layout)

        self.vital_labels[vital] = {
            "value": value_label,
            "status": status_label,
            "frame": frame
        }

        panel.addWidget(frame)

    def setup_timers(self):
        """Initialize and start the timers"""
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.update_vitals)
        self.data_timer.start(2000)  # Update vitals every 2 seconds

        self.ecg_timer = QTimer()
        self.ecg_timer.timeout.connect(self.update_ecg)
        self.ecg_timer.start(20)  # Update ECG every 20ms

    def update_vitals(self):
        """Update all vital sign displays"""
        if len(self.dataset) == 0 or len(self.ecg_records) == 0:
            return

        patient_data = self.dataset.iloc[self.current_index % len(self.dataset)]
        current_ecg = self.ecg_records[self.current_record % len(self.ecg_records)]

        # Get ECG window
        window_start = self.ecg_index % len(current_ecg)
        window_end = (window_start + self.ecg_window_size) % len(current_ecg)
        ecg_window = current_ecg[window_start:window_end] if window_end > window_start else np.concatenate(
            (current_ecg[window_start:], current_ecg[:window_end]))

        # Process ECG
        processed_ecg = self.ecg_processor.process_ecg(ecg_window)
        peaks = self.ecg_processor.detect_r_peaks(processed_ecg)
        arrhythmia_type = self.ecg_processor.detect_arrhythmia_type(processed_ecg, peaks)
        ecg_hr = self.ecg_processor.calculate_hr(peaks)
        hr = ecg_hr if ecg_hr > 30 else int(patient_data["Heart Rate"])

        # Update displays
        self.update_vital_display("Heart Rate", f"{int(hr)}",
                                self.alarm_manager.check_alarm("Heart Rate", hr))
        self.update_vital_display("Oxygen Saturation", f"{float(patient_data['Oxygen Saturation']):.1f}",
                                self.alarm_manager.check_alarm("Oxygen Saturation", patient_data['Oxygen Saturation']))
        self.update_vital_display("Body Temperature", f"{float(patient_data['Body Temperature']):.1f}",
                                self.alarm_manager.check_alarm("Body Temperature", patient_data['Body Temperature']))
        self.update_vital_display("BP", f"{int(patient_data['Systolic Blood Pressure'])}/{int(patient_data['Diastolic Blood Pressure'])}",
                                self.alarm_manager.check_alarm("BP", f"{patient_data['Systolic Blood Pressure']}/{patient_data['Diastolic Blood Pressure']}"))
        self.update_vital_display("Respiratory Rate", f"{int(patient_data['Respiratory Rate'])}",
                                self.alarm_manager.check_alarm("Respiratory Rate", patient_data['Respiratory Rate']))

        # Update Arrhythmia with specific type
        is_abnormal = arrhythmia_type not in ["Normal Sinus Rhythm", "Sinus Arrhythmia (Normal Variant)"]
        self.update_vital_display("Arrhythmia", arrhythmia_type,
                                self.alarm_manager.check_alarm("Arrhythmia", arrhythmia_type))

        self.current_index += 1

        # Trigger alarms if needed
        self.check_alarm_conditions()

    def update_ecg(self):
        """Update the ECG plot"""
        if len(self.ecg_records) == 0:
            return

        current_ecg = self.ecg_records[self.current_record % len(self.ecg_records)]
        window_start = self.ecg_index % len(current_ecg)
        window_end = (window_start + self.ecg_window_size) % len(current_ecg)

        if window_end > window_start:
            ecg_window = current_ecg[window_start:window_end]
        else:
            ecg_window = np.concatenate((current_ecg[window_start:], current_ecg[:window_end]))

        processed_ecg = self.ecg_processor.process_ecg(ecg_window)
        self.curve.setData(self.time_axis, processed_ecg)
        self.ecg_index = (self.ecg_index + 10) % len(current_ecg)

    def check_alarm_conditions(self):
        """Check if any alarms should be triggered"""
        critical_statuses = [
            self.vital_labels["Heart Rate"]["status"].text(),
            self.vital_labels["Oxygen Saturation"]["status"].text(),
            self.vital_labels["Body Temperature"]["status"].text(),
            self.vital_labels["BP"]["status"].text(),
            self.vital_labels["Respiratory Rate"]["status"].text(),
            self.vital_labels["Arrhythmia"]["status"].text()
        ]

        if any(status == "CRITICAL" for status in critical_statuses):
            if self.alarm_sound.source().isValid():
                self.alarm_sound.play()

    def update_vital_display(self, name, value, status):
        """Update a single vital sign display"""
        unit = self.units.get(name, "")
        display_text = f"{value} {unit}" if unit else value

        self.vital_labels[name]["value"].setText(display_text)
        self.vital_labels[name]["status"].setText(status)

        color = self.colors.get(status, "#00FF00")
        self.vital_labels[name]["value"].setStyleSheet(f"color: {color};")
        self.vital_labels[name]["status"].setStyleSheet(f"color: {color};")

        border_color = {
            "CRITICAL": "#F00",
            "WARNING": "#FF0",
            "NORMAL": "#444"
        }.get(status, "#444")

        self.vital_labels[name]["frame"].setStyleSheet(
            f"background-color: #222; border: 2px solid {border_color}; border-radius: 5px;"
        )

    def closeEvent(self, event):
        """Clean up resources when closing"""
        if hasattr(self, 'alarm_sound') and self.alarm_sound.isPlaying():
            self.alarm_sound.stop()
        if hasattr(self, 'data_timer'):
            self.data_timer.stop()
        if hasattr(self, 'ecg_timer'):
            self.ecg_timer.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)

    # Set dark theme
    app.setStyle("Fusion")
    dark_palette = app.palette()
    dark_palette.setColor(dark_palette.Window, QColor(53, 53, 53))
    dark_palette.setColor(dark_palette.WindowText, Qt.white)
    dark_palette.setColor(dark_palette.Base, QColor(25, 25, 25))
    dark_palette.setColor(dark_palette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(dark_palette.ToolTipBase, Qt.white)
    dark_palette.setColor(dark_palette.ToolTipText, Qt.white)
    dark_palette.setColor(dark_palette.Text, Qt.white)
    dark_palette.setColor(dark_palette.Button, QColor(53, 53, 53))
    dark_palette.setColor(dark_palette.ButtonText, Qt.white)
    dark_palette.setColor(dark_palette.BrightText, Qt.red)
    dark_palette.setColor(dark_palette.Link, QColor(42, 130, 218))
    dark_palette.setColor(dark_palette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(dark_palette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)

    monitor = PatientMonitor()
    monitor.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()