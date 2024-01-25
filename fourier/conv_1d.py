# %%
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams["font.size"] = 8

# %%
np.random.seed(1)

n = 255
x = np.arange(n)

sigs = np.array([2, 10, 20])


f = np.zeros(n)
f[n // 4 : (3 * n) // 4] = 1

f += np.random.normal(0, 0.05, n)

# %%

fig, ax = plt.subplots(
    3,
    sigs.shape[0] + 1,
    figsize=((sigs.shape[0] + 1) * 3, 3 * 2),
    sharey="row",
    sharex="row",
)

ax[0, 0].plot(x, f)
ax[0, 0].set_title("signal $f(x)$", fontsize="medium")
ax[1, 0].set_axis_off()
ax[2, 0].set_axis_off()

for i, sig in enumerate(sigs):
    kernel = np.exp(-((x - (n // 2)) ** 2) / (2 * sig**2))

    kernel_ft = np.fft.fftshift(np.abs(np.fft.fft(kernel, norm="forward")))
    kernel_ft /= kernel_ft.max()
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1 / n))

    trunc = int(n // 2 - 3.5 * sig)
    if not trunc % 2:
        trunc += 1

    kernel = kernel[trunc:-trunc]
    kernel /= kernel.sum()

    xk = np.arange(kernel.size) - kernel.size // 2

    f_conv = np.convolve(f, kernel, mode="same")

    ax[0, i + 1].plot(x, f_conv)
    ax[1, i + 1].plot(xk, kernel)
    ax[1, i + 1].set_xlim(-n // 2, n // 2)

    ax[2, i + 1].plot(freq, kernel_ft)

    ax[0, i + 1].set_title("convolved signal $f(x)$", fontsize="medium")
    ax[1, i + 1].set_title("convolution kernel $g(x)$", fontsize="medium")
    ax[2, i + 1].set_title("FT(convolution kernel $g(x)$)", fontsize="medium")

for axx in ax[:-1, :].ravel():
    axx.set_xlabel("location $x$")
for axx in ax[-1, :].ravel():
    axx.set_xlabel("frequency $k_x$")

fig.tight_layout()
fig.show()
