import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 8

n = 31

k = np.fft.fftfreq(n, d=1 / n)

F = np.abs(k)

f = np.fft.ifft(F)


fig, ax = plt.subplots(2, 1, figsize=(6, 6))
ax[0].bar(np.fft.fftshift(k), np.fft.fftshift(F), width=1)
ax[0].plot(np.fft.fftshift(k), np.fft.fftshift(F), color="r", lw=1)
ax[1].bar(np.arange(n) - n // 2, np.fft.fftshift(f.real), width=1)
ax[1].plot(np.arange(n) - n // 2, np.fft.fftshift(f.real), "r.-", lw=1, ms=2)
ax[0].set_title("ramp filter in frequency domain")
ax[1].set_title("ramp filter convolution kernel in spatial domain")
ax[0].set_xlabel("frequency $k$")
ax[1].set_xlabel("location $x$")

for axx in ax.ravel():
    axx.grid(ls=":")

fig.tight_layout()
fig.show()
