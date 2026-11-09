# Acoustic Frequency Shift Measurement Rig (Doppler Effect)

A Raspberry Pi 5 data acquisition rig that measures the frequency drift of a
moving buzzer tone, demonstrating the Doppler effect in sound as an analogy
for cosmological redshift. Built as the project for PHY4000_350: Black
Holes, Big Bangs & the Cosmos (Algonquin College). The full report,
including background, procedure, results figures, and references, is in
`Report.pdf` in this repo.

## Hardware

- Raspberry Pi 5
- MCP3208 12-bit ADC over SPI (spidev, 2 MHz, mode 0, CE0)
- MAX4466 electret microphone amplifier
- KC-1201 passive buzzer (HiPi Sensor Kit 4.0), driven by hardware PWM via
  lgpio on GPIO 18 at 2800 Hz, 50 percent duty cycle
- 3.3 V reference

## How it works

Sampling and detection run in a background thread while a curses UI handles
commands (start, pause, resume, stop, save, analyze, exit).

Per window of 2048 samples:

1. Sample the MCP3208 one reading at a time over SPI.
2. Measure the actual elapsed time for the window and compute the actual
   sample rate for that window (nominal target 27000 Hz; measured 25-28 kHz).
3. Detrend, apply a Hann window, zero-pad to 16384 points, and compute the
   rFFT using the measured per-window sample rate for the frequency axis.
4. Search a +/-10 Hz band around the expected 2800 Hz tone for the peak.
5. Reject the window if the peak is below 3x the median noise floor.
6. Refine the peak with quadratic interpolation over the three bins around
   it.
7. Log timestamp, frequency, and measured sample rate to CSV
   (`Data/trial_N.csv`); plot raw and smoothed traces to
   `Plots/trial_N_frequency_vs_time.png` (smoothing: zero-phase filtfilt,
   window 100).

## Sampling rate

The initial target sampling rate of 44.1 kHz could not be achieved due to
the overhead of reading samples one by one in a Python loop over SPI. The
actual sampling rate varied between 25-28 kHz, introducing jitter. Rather
than assuming the nominal rate, the script measures the actual sample rate
for every window and uses that measured value for the FFT frequency axis.
The `fs_used_hz` column in each CSV is that measurement.

## Results and limitations

Expected Doppler shifts for hand motions were small (+/-4 to +/-8 Hz at
0.5-1 m/s), and the measured static-buzzer frequency still varied by roughly
+/-15-20 Hz between windows. Root cause was hardware, not software: the
MAX4466 amplifies up to 100 kHz and there was no analog low-pass filter
between the amplifier and the ADC, so high-frequency noise aliased into the
sampled band, and the 12-bit ADC limited dynamic range. Software attempts
(zero-padding, peak interpolation, noisereduce spectral gating, filtfilt
smoothing, narrower search band, smaller buffers) could not compensate.

Recommended hardware fixes: an RC low-pass filter between the MAX4466 and
the MCP3208, a higher-resolution ADC (e.g. 16-bit ADS1115), or DMA-based
sampling to remove Python loop overhead.

## Repo contents

- `doppler.py` - acquisition and analysis script
- `Report.pdf` - project report with
  background, results figures, root cause analysis, and references
- `requirements.txt`

## Requirements and usage

Runs on a Raspberry Pi 5 with the SPI interface enabled. Install
dependencies from `requirements.txt` (spidev and lgpio are Pi-only), then:

    python3 doppler.py

Use the number keys in the terminal UI: 1 start, 2 pause, 3 resume,
4 stop, 5 save trial, 6 analyze and plot, 7 exit.