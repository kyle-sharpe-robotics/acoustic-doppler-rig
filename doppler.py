import os
import csv
import threading
import curses
import time
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import spidev
import lgpio
from scipy import signal

# --- Hardware Configuration ---
BUZZER_PIN: int = 18
ADC_CS: int = 0  # SPI Device 0 (CE0)
SPI_BUS: int = 0  # SPI Bus 0
MICROPHONE_CHANNEL: int = 0
TONE_FREQ: float = 2800.0  # Hz
SAMPLE_RATE: int = 27000  # Target sampling rate (Hz)
BUFFER_SIZE: int = 2048  # FFT window size (power of 2)
VREF: float = 3.3  # Reference voltage (V)
SMOOTH_WINDOW_SIZE: int = 100  # For frequency smoothing

# --- Directories ---
os.makedirs("Data", exist_ok=True)
os.makedirs("Plots", exist_ok=True)

# --- Buzzer Controller (Fixed) ---
class BuzzerController:
    def __init__(self, pin: int, frequency: float) -> None:
        self.pin: int = pin
        self.frequency: float = frequency
        self.gpio_chip: Optional[int] = None

    def start(self) -> None:
        if self.gpio_chip is not None:
            return
        self.gpio_chip = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self.gpio_chip, self.pin)
        lgpio.tx_pwm(self.gpio_chip, self.pin, self.frequency, 50)  # 50% duty cycle

    def stop(self) -> None:
        if self.gpio_chip is not None:
            try:
                lgpio.gpio_write(self.gpio_chip, self.pin, 0)
                lgpio.gpio_free(self.gpio_chip, self.pin)
            finally:
                self.gpio_chip = None

    def close(self) -> None:
        if self.gpio_chip is not None:
            try:
                lgpio.gpiochip_close(self.gpio_chip)
            finally:
                self.gpio_chip = None

# --- ADC Reader (Fixed Command Byte) ---
class ADCReader:
    def __init__(self, bus: int, cs_pin: int, channel: int = 0) -> None:
        self.bus: int = bus
        self.cs_pin: int = cs_pin
        self.channel: int = channel
        self.spi: spidev.SpiDev = spidev.SpiDev()
        self.spi.open(self.bus, self.cs_pin)
        self.spi.max_speed_hz = 2_000_000  # 2 MHz (safe for MCP3208)
        self.spi.mode = 0b00  # SPI Mode 0,0

    def read(self) -> int:
        # Correct command byte: Start=1, SGL/DIFF=1, D2=0, D1/D0=channel
        cmd: int = 0xC0 | ((self.channel & 0x07) << 2)  # Shift by 2, not 3
        data: List[int] = self.spi.xfer2([cmd, 0, 0])
        return ((data[1] & 0x0F) << 8) | data[2]  # 12-bit result

    def close(self) -> None:
        self.spi.close()

# --- Frequency Detector (Simplified) ---
class FrequencyDetector:
    def __init__(self, buffer_size: int) -> None:
        self.buffer_size: int = buffer_size
        self.window: np.ndarray = np.hanning(buffer_size)

    def estimate_frequency(self, samples: np.ndarray, fs_used: float, expected_tone_hz: float = TONE_FREQ, search_band_hz: float = 10.0) -> Optional[float]:
        try:
            if fs_used <= 0.0:
                return None

            # Zero-pad to 16384 for better resolution
            padded_size = 16384
            padded = np.zeros(padded_size)
            padded[:len(samples)] = samples - np.mean(samples)

            # Create window matching padded size
            window = np.hanning(padded_size)
            windowed = padded * window

            fft_result = np.fft.rfft(windowed)
            freqs = np.fft.rfftfreq(padded_size, 1.0 / SAMPLE_RATE)
            magnitudes = np.abs(fft_result)

            lo = max(0, expected_tone_hz - search_band_hz)
            hi = expected_tone_hz + search_band_hz
            mask = (freqs >= lo) & (freqs <= hi)
            if not np.any(mask):
                peak_idx = np.argmax(magnitudes)
                return float(freqs[peak_idx])

            peak_idx_local = np.argmax(magnitudes[mask])
            peak_idx = np.where(mask)[0][peak_idx_local]

            noise_floor = np.median(magnitudes)
            if magnitudes[peak_idx] < noise_floor * 3:
                return None

            if peak_idx > 0 and peak_idx < len(freqs) - 1:
                x = freqs[peak_idx-1:peak_idx+2]
                y = magnitudes[peak_idx-1:peak_idx+2]
                p = np.polyfit(x - x[1], y, 2)
                delta = -p[1] / (2 * p[0])
                return float(x[1] + delta)
            return float(freqs[peak_idx])
        except Exception as e:
            print(f"Frequency detection error: {e}")
            return None

# --- Data Collector (Fixed Timestamps/Frequencies) ---
@dataclass
class TrialData:
    timestamps: List[float] = field(default_factory=list)
    frequencies: List[float] = field(default_factory=list)
    fs_used: List[float] = field(default_factory=list)

class DataCollector:
    def __init__(self) -> None:
        self.buzzer: BuzzerController = BuzzerController(BUZZER_PIN, TONE_FREQ)
        self.adc: ADCReader = ADCReader(SPI_BUS, ADC_CS, MICROPHONE_CHANNEL)
        self.detector: FrequencyDetector = FrequencyDetector(BUFFER_SIZE)
        self.running: bool = False
        self.paused: bool = False
        self.thread: Optional[threading.Thread] = None
        self.current_trial: TrialData = TrialData()
        self.trials: List[TrialData] = []
        self.trial_start_time: float = 0.0

    def _collect_one_window(self) -> Tuple[np.ndarray, float]:
        samples: np.ndarray = np.zeros(BUFFER_SIZE, dtype=np.int32)
        start_time: float = time.perf_counter()
        for i in range(BUFFER_SIZE):
            samples[i] = self.adc.read()
        end_time: float = time.perf_counter()
        dt: float = end_time - start_time
        fs_used: float = (BUFFER_SIZE - 1) / dt if dt > 0 else 0.0
        return samples, fs_used

    def collect(self) -> None:
        self.trial_start_time = time.time()
        self.current_trial = TrialData()
        while self.running:
            if self.paused:
                time.sleep(0.01)
                continue
            try:
                samples, fs_used = self._collect_one_window()
                freqs = np.fft.rfftfreq(padded_size, 1.0 / fs_used)
                if freq is not None and fs_used > 0:
                    elapsed: float = time.time() - self.trial_start_time
                    self.current_trial.timestamps.append(elapsed)
                    self.current_trial.frequencies.append(freq)
                    self.current_trial.fs_used.append(fs_used)
            except Exception as e:
                print(f"Collection error: {e}")

    def start(self) -> None:
        if not self.running:
            self.running = True
            self.paused = False
            self.buzzer.start()
            time.sleep(1.0)  # Buzzer warmup
            self.trial_start_time = time.time()
            self.current_trial = TrialData()
            self.thread = threading.Thread(target=self.collect, daemon=True)
            self.thread.start()
            print("Data collection started. Buzzer active.")

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def stop(self) -> None:
        self.running = False
        self.buzzer.stop()
        if self.thread is not None:
            self.thread.join()

    def save_trial(self) -> None:
        if not self.current_trial.timestamps:
            print("No data to save.")
            return
        trial_num: int = len(self.trials) + 1
        filename: str = f"Data/trial_{trial_num}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "frequency_hz", "fs_used_hz"])
            for ts, freq, fs in zip(
                self.current_trial.timestamps,
                self.current_trial.frequencies,
                self.current_trial.fs_used,
            ):
                writer.writerow([ts, freq, fs])
        self.trials.append(self.current_trial)
        print(f"Trial {trial_num} saved to {filename}.")
        self.current_trial = TrialData()

    def close(self) -> None:
        self.stop()
        self.buzzer.close()
        self.adc.close()

# --- Data Analyzer (Fixed Plotting) ---
class DataAnalyzer:
    @staticmethod
    def plot_trial(
        trial: TrialData,
        trial_num: int,
        smoothed: Optional[List[float]] = None,
    ) -> None:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(trial.timestamps, trial.frequencies, label="Raw", alpha=0.3)
            if smoothed is not None:
                plt.plot(trial.timestamps, smoothed, label="Smoothed", color="orange")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.title(f"Trial {trial_num}: Frequency vs. Time")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"Plots/trial_{trial_num}_frequency_vs_time.png", dpi=200)
            plt.close()
        except Exception as e:
            print(f"Plotting error: {e}")

    @staticmethod
    def smooth_frequencies(frequencies: List[float], window_size: int) -> List[float]:
        if len(frequencies) < window_size or window_size <= 1:
            return frequencies
        try:
            b = np.ones(window_size) / window_size
            return signal.filtfilt(b, 1.0, frequencies).tolist()
        except Exception:
            return frequencies

def main() -> None:
    # Initialize curses for non-blocking input
    stdscr = curses.initscr()
    curses.cbreak()
    curses.noecho()
    stdscr.keypad(True)
    stdscr.nodelay(True)  # Non-blocking key reads

    collector = DataCollector()
    analyzer = DataAnalyzer()
    stdscr.addstr(0, 0, "Commands: 1:start 2:pause 3:resume 4:stop 5:save 6:analyze 7:exit\n")

    try:
        while True:
            c = stdscr.getch()  # Non-blocking: returns -1 if no key pressed
            if c == -1:
                time.sleep(0.01)  # Reduce CPU usage
                continue

            if c == ord('1'):
                collector.start()
                stdscr.addstr(1, 0, "Data collection started.          \n")
            elif c == ord('2'):
                collector.pause()
                stdscr.addstr(1, 0, "Data collection paused.           \n")
            elif c == ord('3'):
                collector.resume()
                stdscr.addstr(1, 0, "Data collection resumed.          \n")
            elif c == ord('4'):
                collector.stop()
                stdscr.addstr(1, 0, "Data collection stopped.          \n")
            elif c == ord('5'):
                collector.save_trial()
                stdscr.addstr(1, 0, f"Trial saved. Total: {len(collector.trials)}\n")
            elif c == ord('6'):
                if not collector.trials and not collector.current_trial.timestamps:
                    stdscr.addstr(1, 0, "No data to analyze.              \n")
                    continue
                if collector.current_trial.timestamps:
                    smoothed = analyzer.smooth_frequencies(
                        collector.current_trial.frequencies,
                        SMOOTH_WINDOW_SIZE,
                    )
                    analyzer.plot_trial(
                        collector.current_trial,
                        len(collector.trials) + 1,
                        smoothed,
                    )
                for i, trial in enumerate(collector.trials):
                    smoothed = analyzer.smooth_frequencies(
                        trial.frequencies,
                        SMOOTH_WINDOW_SIZE,
                    )
                    analyzer.plot_trial(trial, i + 1, smoothed)
                stdscr.addstr(1, 0, "Analysis complete.               \n")
            elif c == ord('7'):
                collector.close()
                stdscr.addstr(1, 0, "Exiting...                        \n")
                break
            else:
                stdscr.addstr(1, 0, "Invalid command. Use 1-7.        \n")
            stdscr.refresh()
    finally:
        # Clean up curses
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()

if __name__ == "__main__":
    main()