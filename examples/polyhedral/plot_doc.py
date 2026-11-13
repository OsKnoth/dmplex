"""Figures of docs/polyhedral/polyhedral_doc.tex from the data of figure_data.jl and the Galewsky runs
(OUT=file of galewsky_poly.jl): python3 plot_doc.py docfig figures galewsky_out_dir"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

D, OUT, GAL = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)


def polygons(path):
    vals, polys = [], []
    for line in open(path):
        a = np.array(line.split(), float)
        vals.append(a[0])
        polys.append(a[1:].reshape(-1, 3))
    return np.array(vals), polys


def orthographic(lon0, lat0):
    lon0, lat0 = np.radians(lon0), np.radians(lat0)
    v = np.array([np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)])
    e = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    n = np.cross(v, e)
    return v, lambda P: np.stack([P @ e, P @ n], axis=1)


def sphere(ax, polys, values=None, view=(30, 30), cmap="viridis", norm=None, edge="k", lw=0.3):
    v, proj = orthographic(*view)
    keep = [i for i, P in enumerate(polys) if P.mean(axis=0) @ v > 0]
    pc = PolyCollection([proj(polys[i] / np.linalg.norm(polys[i], axis=1)[:, None]) for i in keep],
                        edgecolors=edge, linewidths=lw)
    if values is None:
        pc.set_facecolor("0.92")
    else:
        pc.set_array(values[keep]); pc.set_cmap(cmap); pc.set_norm(norm) if norm else None
    ax.add_collection(pc)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, lw=0.8))
    ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02); ax.set_aspect("equal"); ax.axis("off")
    return pc


# meshes
fig, ax = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
for a, name, title in zip(ax, ("hexagons", "triangles", "cubed"),
                          ("hexagons, level 3 (642 cells)", "triangles, level 3 (1280 cells)", "cubed sphere, 8 × 8 (384 cells)")):
    _, P = polygons(f"{D}/mesh_{name}.txt")
    sphere(a, P)
    a.set_title(title)
fig.savefig(f"{OUT}/poly_meshes.png", dpi=130)

# solid-body rotation
fig, ax = plt.subplots(1, 2, figsize=(9, 4.4), constrained_layout=True)
for a, p in zip(ax, (1, 2)):
    q, P = polygons(f"{D}/rotation_p{p}.txt")
    pc = sphere(a, P, q, view=(0, 10), cmap="viridis", edge="none")
    pc.set_clim(0, 1)
    a.set_title(f"degree {p}: max {q.max():.3f}, min {q.min():.1e}")
fig.colorbar(pc, ax=ax, shrink=0.8)
fig.savefig(f"{OUT}/poly_rotation.png", dpi=100)

# cut cells
fig, ax = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
xz = lambda P: [p[:, [0, 2]] for p in P]
f, P = polygons(f"{D}/cut_fraction.txt")
pc = PolyCollection(xz(P), edgecolors="k", linewidths=0.4, cmap="magma", norm=LogNorm(1e-4, 1))
pc.set_array(np.clip(f, 1e-4, 1)); ax[0].add_collection(pc)
fig.colorbar(pc, ax=ax[0], label="volume / volume of the prism")
ax[0].set_title(f"cut cells of the wall y = 0, 16 × 8 prisms: volume fractions (smallest here {f.min():.1e})")
c, P = polygons(f"{D}/cut_merged.txt")
rng = np.random.default_rng(2)
colors = rng.uniform(0.35, 1.0, (int(c.max()) + 1, 3))
pc = PolyCollection(xz(P), facecolors=colors[c.astype(int)], edgecolors="k", linewidths=0.4)
ax[1].add_collection(pc)
ax[1].set_title("after merging the cells below half of their prism (one colour per cell)")
e, P = polygons(f"{D}/cut_steady.txt")
m = np.abs(e).max()
pc = PolyCollection(xz(P), edgecolors="none", cmap="RdBu_r")
pc.set_array(e); pc.set_clim(-m, m); ax[2].add_collection(pc)
fig.colorbar(pc, ax=ax[2], label="q(t = 1/2) - q(0)")
ax[2].set_title("steady transport q = exp(2ψ), 32 × 16, degree 2: error after t = 1/2")
xs = np.linspace(-1, 1, 1000)
for a in ax:
    a.plot(xs, 0.03 + 0.25 * np.exp(-(xs / 0.4) ** 2) * np.cos(np.pi * xs / 0.5) ** 2, "k", lw=1.2)
    a.set_xlim(-1, 1); a.set_ylim(0, 1 if a is ax[2] else 0.6); a.set_aspect("equal")
    a.set_xlabel("x"); a.set_ylabel("z")
fig.savefig(f"{OUT}/poly_cut.png", dpi=130)

# Galewsky: relative vorticity at day 6
cases = (("hex_perot_p1", "hexagons level 7, Perot"), ("cub_perot_p1", "cubed sphere 200 × 200, Perot"),
         ("tri_linear_p1", "triangles level 7, linear fit"), ("hex_perot_p0", "hexagons level 7, Perot, unperturbed"))
fig, ax = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)
lev = np.linspace(-1.6e-4, 1.6e-4, 33)
for a, (tag, title) in zip(ax, cases):
    d = np.loadtxt(f"{GAL}/{tag}.txt")
    lon, lat, z = np.degrees(d[:, 0]), np.degrees(d[:, 1]), d[:, 2]
    s = lat > 5
    cf = a.tricontourf(lon[s], lat[s], np.clip(z[s], lev[0], lev[-1]), lev, cmap="RdBu_r", extend="both")
    a.set_ylim(10, 80); a.set_xlim(-180, 180)
    a.set_title(f"{title} ({len(z)} cells): ζ at day 6")
    a.set_ylabel("latitude")
ax[-1].set_xlabel("longitude")
fig.colorbar(cf, ax=ax, shrink=0.6, label="ζ [1/s]")
fig.savefig(f"{OUT}/poly_galewsky.png", dpi=110)
print("done")
