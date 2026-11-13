"""θ and w of the cut-cell mountain runs (binary from mountain.jl) and the linear solution:
python3 plot_mountain.py gk_03h.bin poly_mountain.png TITLE"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

g, Rd, cp = 9.81, 287.0, 1004.0
U0, N, ths = 10.0, 0.01, 300.0


def read(path):
    with open(path, "rb") as f:
        nx, nz = np.fromfile(f, np.int64, 2)
        L, H, dx, dz, h0, a = np.fromfile(f, np.float64, 6)
        T, W, U = (np.fromfile(f, np.float64, nx * nz).reshape(nz, nx) for _ in range(3))
    x = (-L + (np.arange(nx) + 0.5) * dx) / 1e3
    z = (np.arange(nz) + 0.5) * dz / 1e3
    return x, z, T, W, U, dict(L=L, H=H, dx=dx, dz=dz, h0=h0, a=a)


def linear(x, z, h0, a):
    """Linear w and θ' over the Agnesi hill, nonhydrostatic, radiation condition, with the
    compressible growth of the amplitude sqrt(ρ0(0)/ρ0(z))."""
    n, Lx = 2 ** 15, 2000e3
    xs = (np.arange(n) - n // 2) * Lx / n
    k = 2 * np.pi * np.fft.fftfreq(n, Lx / n)
    hk = np.fft.fft(np.fft.ifftshift(h0 / (1 + (xs / a) ** 2)))
    l2 = (N / U0) ** 2 - k ** 2
    m = np.where(l2 > 0, np.sign(k) * np.sqrt(np.abs(l2)), 1j * np.sqrt(np.abs(l2)))
    Hs = Rd * ths / g
    W, T = np.zeros((len(z), len(x))), np.zeros((len(z), len(x)))
    for j, zz in enumerate(z * 1e3):
        eta = np.fft.fftshift(np.real(np.fft.ifft(hk * np.exp(1j * m * zz)))) * np.exp(zz / (2 * Hs))
        w = U0 * np.gradient(eta, xs)
        W[j] = np.interp(x * 1e3, xs, w)
        T[j] = -np.interp(x * 1e3, xs, eta) * ths * N ** 2 / g * np.exp(N ** 2 * zz / g)
    return W, T


def main(path, out, title):
    x, z, T, W, U, p = read(path)
    th0 = ths * np.exp(N ** 2 * z[:, None] * 1e3 / g)
    Wl, Tl = linear(x, z, p["h0"], p["a"])
    xmask, zmask = (x >= -10) & (x <= 15), z <= 9
    X, Z = np.meshgrid(x[xmask], z[zmask])
    sub = lambda A: np.ma.masked_invalid(A[np.ix_(zmask, xmask)])
    xt = np.linspace(-10, 15, 1000)
    zt = p["h0"] / (1 + (xt * 1e3 / p["a"]) ** 2) / 1e3
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    lev = np.arange(-2.0, 2.01, 0.2)
    for a_, A, name in ((ax[0, 0], W, "w cut cells"), (ax[0, 1], Wl, "w linear")):
        cf = a_.contourf(X, Z, np.clip(sub(A), lev[0], lev[-1]), lev, cmap="RdBu_r", extend="both")
        a_.contour(X, Z, sub(A), lev[lev != 0], colors="k", linewidths=0.4)
        a_.set_title(f"{name} [m/s], contours 0.2")
    fig.colorbar(cf, ax=ax[0, :], shrink=0.8)
    levt = np.arange(280, 340, 1.0)
    ax[1, 0].contour(X, Z, sub(T), levt, colors="k", linewidths=0.6)
    ax[1, 0].set_title("θ cut cells [K], contours 1 K")
    lp = np.arange(-6, 6.01, 0.5)
    ax[1, 1].contour(X, Z, sub(T - th0), lp[lp != 0], colors="r", linewidths=0.6)
    ax[1, 1].contour(X, Z, sub(Tl), lp[lp != 0], colors="k", linewidths=0.6, linestyles="dashed")
    ax[1, 1].set_title("θ' [K], cut cells (red) and linear (black dashed), contours 0.5 K")
    for a_ in ax.flat:
        a_.fill_between(xt, 0, zt, color="0.6")
        a_.set_xlabel("x [km]"); a_.set_ylabel("z [km]")
        a_.set_ylim(0, 9)
    fig.suptitle(title)
    fig.savefig(out, dpi=110)
    j = np.argmin(np.abs(z - 2.0))
    print(f"{path}: max|w| {np.nanmax(np.abs(W)):.3f} (linear {np.abs(Wl).max():.3f}), "
          f"w at z=2 km: min {np.nanmin(W[j]):.3f} max {np.nanmax(W[j]):.3f} (linear {Wl[j].min():.3f} {Wl[j].max():.3f})")


if __name__ == "__main__":
    main(*sys.argv[1:4])
