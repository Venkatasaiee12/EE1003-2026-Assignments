import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

freqs = [30, 50, 70]

def sig(t):
    return sum(np.cos(2*np.pi*f*t) for f in freqs)

duration = 0.5
t_cont = np.linspace(0, duration, 2000, endpoint=False)
x_cont = sig(t_cont)

fig, ax = plt.subplots(2, 2, figsize=(12,6))

for i, fs in enumerate([150, 100]):
    t = np.arange(0, duration, 1/fs)
    x = sig(t)

    # FFT
    N = len(x)
    X = rfft(x)
    f = rfftfreq(N, 1/fs)
    amp = np.abs(X)/(N/2)
    amp[0] /= 2
    if N % 2 == 0:
        amp[-1] /= 2

# Time plot
    ax[0,i].plot(t_cont, x_cont, 'gray', label='Continuous Signal')
    ax[0,i].stem(t, x, basefmt=" ", label='Sampled Signal')
    ax[0,i].set_xlim(0, 0.1)
    ax[0,i].set_title(f"Time Domain (fs={fs} Hz)")
    ax[0,i].set_xlabel("Time (seconds)")
    ax[0,i].set_ylabel("Amplitude")
    ax[0,i].legend()

    # Frequency plot
    ax[1,i].stem(f, amp, basefmt=" ")
    ax[1,i].set_xlim(0, 80)
    ax[1,i].set_ylim(0, 2.2)
    ax[1,i].set_title("Frequency Spectrum")
    ax[1,i].set_xlabel("Frequency (Hz)")
    ax[1,i].set_ylabel("Amplitude")

plt.tight_layout()
plt.show()
