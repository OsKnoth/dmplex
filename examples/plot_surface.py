"""Surface pressure (and temperature of the lowest layer) of the baroclinic wave from the text files
of `baroclinic_wave.jl --out=prefix` (`lon lat p_s T` per column, any dycore), one row per file:

    python3 plot_surface.py out.png "title" poly5_day08.txt "poly, day 8" mpas5_day08.txt "mpas, day 8" ...
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

out, title, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
files, names = pairs[0::2], pairs[1::2]
fig, axes = plt.subplots(len(files), 2, figsize=(14, 2.9 * len(files) + 0.6), squeeze=False, constrained_layout=True)
lev_p = np.arange(920, 1032, 4.0)
lev_t = np.arange(220, 310, 4.0)
for row, (f, name) in zip(axes, zip(files, names)):
    d = np.loadtxt(f)
    lon, lat, ps, T = np.mod(d[:, 0], 360), d[:, 1], d[:, 2] / 100, d[:, 3]
    # periodic copies so that the triangulation covers 0..360
    lon = np.concatenate([lon, lon - 360, lon + 360]); lat = np.tile(lat, 3); ps = np.tile(ps, 3); T = np.tile(T, 3)
    keep = (lon > -20) & (lon < 380) & (lat > 0)
    tri = mtri.Triangulation(lon[keep], lat[keep])
    cp = row[0].tricontourf(tri, ps[keep], lev_p, cmap="viridis", extend="both")
    row[0].tricontour(tri, ps[keep], lev_p, colors="k", linewidths=0.3)
    row[0].set_title(f"{name}: surface pressure [hPa], min {d[:, 2].min() / 100:.1f}")
    ct = row[1].tricontourf(tri, T[keep], lev_t, cmap="RdBu_r", extend="both")
    row[1].set_title(f"{name}: temperature of the lowest layer [K]")
    for a in row:
        a.set_xlim(0, 360); a.set_ylim(10, 80); a.set_ylabel("latitude")
    fig.colorbar(cp, ax=row[0]); fig.colorbar(ct, ax=row[1])
for a in axes[-1]:
    a.set_xlabel("longitude")
fig.suptitle(title)
fig.savefig(out, dpi=100)
print("wrote", out)
