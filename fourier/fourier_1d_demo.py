"""
1D Fourier transform demo
=========================

foo bar
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 8

# %%
np.random.seed(1)

n = 16
x = np.arange(n)
k = np.arange(n // 2 + 1)

filt = np.exp(-(k**2) / (2 * 2**2))

f = np.zeros(n)
f[1] = 2
f[2:7] = 4
f[7] = 2
f[8:11] = 1
f[14] = 8

f += np.random.normal(0, 0.5, n)

# f = 2.5 * np.cos(6 * (2 * np.pi) * x / n) + 1
# f -= f.mean()

# f = np.exp(-((x - 8) ** 2) / (2 * 0.7**2))

# %%
F = np.fft.rfft(f, norm="forward")
F_amp = np.abs(F)
F_amp[1:-1] *= 2
F_phase = np.angle(F)
F_phase[F_amp < 1e-5] = 0

F_phase[0] = 0
F_phase[-1] = 0

# %%
fig, ax = plt.subplots(3, 1, figsize=(4, 5))
ax[0].bar(x, f, width=(x[1] - x[0]), edgecolor="k")
ax[1].bar(k, F_amp, width=(k[1] - k[0]), edgecolor="k", color=plt.cm.tab10(1))
ax[2].bar(k, F_phase, width=(k[1] - k[0]), edgecolor="k", color=plt.cm.tab10(1))
ax[2].set_ylim(-np.pi, np.pi)

ax[0].set_title("signal $f(x)$")
ax[1].set_title("cosine amplitudes")
ax[2].set_title("cosine phases")

ax[0].set_xlabel("location $x$")
ax[1].set_xlabel("spatial frequency $k$")
ax[2].set_xlabel("spatial frequency $k$")

fig.tight_layout()
fig.show()

# %%
fig2, ax2 = plt.subplots(n // 2 + 1, 3, figsize=(11, 8), sharex=True, sharey="col")

f_rec = np.zeros(n)

x2 = np.linspace(0, x[-1], 1000)

for i in k:
    tmp = F_amp[i] * np.cos(i * (2 * np.pi) * x / n + F_phase[i])
    f_rec += tmp

    tmp2 = F_amp[i] * np.cos(i * (2 * np.pi) * x2 / n + F_phase[i])

    ax2[i, 0].bar(
        x, np.cos(i * (2 * np.pi) * x / n), width=(x[1] - x[0]), edgecolor="k"
    )
    ax2[i, 0].plot(
        x2, np.cos(i * (2 * np.pi) * x2 / n), "-", lw=1, color=plt.cm.tab10(3)
    )

    ax2[i, 1].bar(x, tmp, width=(x[1] - x[0]), edgecolor="k")
    ax2[i, 1].plot(x2, tmp2, "-", lw=1, color=plt.cm.tab10(3))

    ax2[i, 2].bar(x, f_rec, width=(x[1] - x[0]), edgecolor="k")

    ax2[i, 0].set_title(
        f"basis function {i+1}: $\cos({i} \cdot 2\pi x / {n})$", fontsize="medium"
    )

    if i == 0 or i == n // 2:
        ax2[i, 1].set_title(
            f"${F_amp[i]:.2f} \cdot \cos({i} \cdot 2\pi x / {n})$",
            fontsize="medium",
        )
    else:
        ax2[i, 1].set_title(
            f"${F_amp[i]:.2f} \cdot \cos({i} \cdot 2\pi x / {n} {F_phase[i]:+.2f})$",
            fontsize="medium",
        )

    ax2[i, 2].set_title(f"sum of first ${i+1}$ cosines", fontsize="medium")

fig2.tight_layout()
fig2.show()

# %%

# fig3, ax3 = plt.subplots(n, 3, figsize=(11, 8), sharex=True, sharey="col")
#
# f_rec = np.zeros(n)
#
# for i in x:
#    tmp = np.zeros(n)
#    tmp[i] = 1
#    tmp2 = f[i] * tmp
#    f_rec += tmp2
#
#    ax3[i, 0].bar(x, tmp, width=(x[1] - x[0]), edgecolor="k")
#    ax3[i, 1].bar(x, tmp2, width=(x[1] - x[0]), edgecolor="k")
#    ax3[i, 2].bar(x, f_rec, width=(x[1] - x[0]), edgecolor="k")
#
#    ax3[i, 0].set_title(f"basis function {i+1}: $\Pi(x-{i})$", fontsize="medium")
#    ax3[i, 1].set_title(f"${f[i]:.2f} \cdot \Pi(x-{i})$", fontsize="medium")
#    ax3[i, 2].set_title(f"sum of first ${i+1}$ basis fct.", fontsize="medium")
#
# fig3.tight_layout()
# fig3.show()

# %%
fig4, ax4 = plt.subplots(n // 2 + 1, 3, figsize=(11, 8), sharex=True, sharey="col")

f_rec = np.zeros(n)

x2 = np.linspace(0, x[-1], 1000)

for i in k:
    tmp = filt[i] * F_amp[i] * np.cos(i * (2 * np.pi) * x / n + F_phase[i])
    f_rec += tmp

    tmp2 = filt[i] * F_amp[i] * np.cos(i * (2 * np.pi) * x2 / n + F_phase[i])

    ax4[i, 0].bar(
        x, np.cos(i * (2 * np.pi) * x / n), width=(x[1] - x[0]), edgecolor="k"
    )
    ax4[i, 0].plot(
        x2, np.cos(i * (2 * np.pi) * x2 / n), "-", lw=1, color=plt.cm.tab10(3)
    )

    ax4[i, 1].bar(x, tmp, width=(x[1] - x[0]), edgecolor="k")
    ax4[i, 1].plot(x2, tmp2, "-", lw=1, color=plt.cm.tab10(3))

    ax4[i, 2].bar(x, f_rec, width=(x[1] - x[0]), edgecolor="k")

    ax4[i, 0].set_title(
        f"basis function {i+1}: $\cos({i} \cdot 2\pi x / {n})$", fontsize="medium"
    )

    if i == 0 or i == n // 2:
        ax4[i, 1].set_title(
            f"${filt[i]:.2f} \cdot {F_amp[i]:.2f} \cdot \cos({i} \cdot 2\pi x / {n})$",
            fontsize="medium",
        )
    else:
        ax4[i, 1].set_title(
            f"${filt[i]:.2f} \cdot {F_amp[i]:.2f} \cdot \cos({i} \cdot 2\pi x / {n} {F_phase[i]:+.2f})$",
            fontsize="medium",
        )

    ax4[i, 2].set_title(f"sum of first ${i+1}$ cosines", fontsize="medium")

fig4.tight_layout()
fig4.show()

# %%
fig, ax = plt.subplots(2, 3, figsize=(12, 2 * 5 / 3), sharey="row")
ax[0, 0].bar(x, f, width=(x[1] - x[0]), edgecolor="k")
ax[1, 0].bar(k, F_amp, width=(k[1] - k[0]), edgecolor="k", color=plt.cm.tab10(1))

ax[0, 1].bar(x, f_rec, width=(x[1] - x[0]), edgecolor="k")
ax[1, 1].bar(k, filt * F_amp, width=(k[1] - k[0]), edgecolor="k", color=plt.cm.tab10(1))

ax[1, 2].bar(k, filt, width=(k[1] - k[0]), edgecolor="k", color=plt.cm.tab10(1))


ax[0, 0].set_title("signal $f(x)$")
ax[1, 0].set_title("cosine amplitudes")
ax[0, 1].set_title("freq. filtered signal $f(x)$")
ax[1, 1].set_title("filtered cosine amplitudes")
ax[1, 2].set_title("filter function")
for i in range(3):
    ax[0, i].set_xlabel("location $x$")
    ax[1, i].set_xlabel("spatial frequency $k$")

ax[0, 2].set_axis_off()

fig.tight_layout()
fig.show()
