# %%
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 8

# %%
np.random.seed(1)

n = 16
x = np.arange(n)
k = np.arange(n // 2 + 1)

X, Y = np.meshgrid(x, x)
KX, KY = np.meshgrid(k, k)

f = np.exp(-((X - 8) ** 2 + (Y - 8) ** 2) / (2 * 2**2))
f[1, 1] = 1
f[-2, :] = 0.5

# %%

fig, ax = plt.subplots(
    KX.shape[0], KX.shape[1], figsize=(8, 8), sharex=True, sharey=True
)

for i in k:
    for j in k:
        ax[i, j].imshow(
            np.cos(2 * np.pi * k[j] * X / n + 2 * np.pi * k[i] * Y / n),
            vmin=-1,
            vmax=1,
            cmap="Greys",
        )
        if i == 0:
            ax[i, j].set_title(f"$k_x$ = {k[j]}", fontsize="medium")
        if j == 0:
            ax[i, j].set_ylabel(f"$k_y$ = {k[i]}", fontsize="medium")

fig.tight_layout()
fig.show()

# %%

F = np.flip(
    np.fft.fftshift(np.abs(np.fft.fft2(f, norm="forward")))[
        : (n // 2 + 1), : (n // 2 + 1)
    ],
    (0, 1),
)

fig2, ax2 = plt.subplots(1, 2, figsize=(9, 5))
im0 = ax2[0].imshow(f, cmap="Greys", vmin=0, vmax=1)
im1 = ax2[1].imshow(F, cmap="Greys", vmin=0)

ax2[0].set_title("image $f(x,y)$")
ax2[1].set_title("cosine amplitudes $A_{k_x,k_y}$")

ax2[0].set_xlabel("location $x$")
ax2[0].set_ylabel("location $y$")

ax2[1].set_xlabel("frequency $k_x$")
ax2[1].set_ylabel("frequency $k_y$")

fig2.colorbar(im0, ax=ax2[0], location="bottom", fraction=0.04)
fig2.colorbar(im1, ax=ax2[1], location="bottom", fraction=0.04)

fig2.tight_layout()
fig2.show()
