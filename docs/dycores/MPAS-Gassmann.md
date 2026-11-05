# MPAS-Gassmann dycore: hexagonal C grid, ICON-IAP form (`MPASGassmannNonhydrostatic`)

Source: [src/Nonhydrostatic/MPAS/Gassmann/](../../src/Nonhydrostatic/MPAS/Gassmann/), with the TRiSK
operators of [src/Nonhydrostatic/MPAS/Operators.jl](../../src/Nonhydrostatic/MPAS/Operators.jl).
Shared with the other dycores: the vertical grid, the column solver, the C-grid operators, the RK3
driver and the test cases ([src/Nonhydrostatic/Common/](../../src/Nonhydrostatic/Common/)). Formerly
`TRiSKNonhydrostatic` (still an alias).

The ICON-IAP model of Gassmann (2013) on the Voronoi meshes of MPAS. The time stepping is still
the Wicker–Skamarock RK3 of the other dycores; ICON-IAP's own predictor–corrector with its
vertical solve (Gassmann and Herzog 2008) is planned as an option, as is the standard MPAS-A dycore
(MPAS-Orig, split-explicit) as a separate model.

## Discretization

Dry compressible Euler equations, shallow atmosphere, flat ground, on the extruded C grid of a
Voronoi (hexagon–pentagon) mesh. Lorenz staggering: `vn` at the edges × layers, `w` at the cells ×
interfaces, `ρ`, `ρθ` at the cells × layers (Gassmann 2013).

- **Horizontal, per layer:** the TRiSK operators (Thuburn et al. 2009, Ringler et al. 2010) with ρ
  in place of the height: PV flux `Σ w_{e,e'} l_{e'} ρ_{e'} vn_{e'} q_{e,e'} / d_e`, cell kinetic
  energy, divergence and gradient. The energy exchange between kinetic and potential energy is
  exact.
- **Momentum advection:** 3D vector-invariant (`ω × u` and `∇K` with `K = K_h + K_w`), vertical
  coupling by the mass fluxes as in ICON-IAP.
- **Vertical:** implicit per column (tridiagonal for w, linearized Exner pressure), θ at the
  interfaces third-order upwind after a Crank–Nicolson vertical advection (ICON-IAP
  `ltheta_up_vert`).
- **Time stepping:** Wicker–Skamarock RK3 with a HEVI split, horizontal mass fluxes forward–backward.
- **Diffusion:** Smagorinsky (ICON-IAP `diff_diss_hori`, with heating), optional hyperdiffusion.

### ICON-IAP and MPAS horizontal operators

The horizontal operators of MPAS are options of the same model:

| option | ICON-IAP (default) | MPAS-like |
|---|---|---|
| `pv_flux` | `:iap`, rhombus PV of one edge (`i_cori_method = 4`) | `:trisk`, `(q_e + q_{e'})/2` |
| `trisk_weights` | `:iap`, kite fractions of planar triangles | `:exact`, exact spherical kites |
| `kinetic_energy` | `:cell` | `:mixed` (3/8 cell + 5/8 vertex, Skamarock et al. 2012) |

Only the horizontal operators correspond to MPAS. MPAS-A itself uses a split-explicit acoustic
step, flux-form transport, APVM and terrain-following coordinates; that is the planned MPAS-Orig.

## Results

Baroclinic wave (Ullrich et al. 2014), 30 layers, minimum surface pressure (hPa):

| Day | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|
| hexagon 5, dt 300 s | 992.6 | 983.5 | 968.4 | 946.9 | 921.8 |
| hexagon 6, dt 300 s | 991.66 | 982.25 | 963.13 | 940.43 | 925.06 |
| hexagon 7, dt 150 s | 990.2 | 975.8 | 957.5 | 949.5 | 931.4 |

The GPU and CPU runs of hexagon 6 agree; mass is conserved to 4e-16.

Thermal wind (`thermal_wind`, steady), hexagon 4, dt 300 s, 5 days: max |w| 6.7e-4 m/s,
max |θ′| 0.35 K, max |Δp_s| 0.35 hPa.

## Time per step on one A100 (30 layers)

| Version | hexagon 6 (40962 cells) | hexagon 7 (163842 cells) |
|---|---|---|
| original | 19.1 ms | 99.5 ms |
| ρ_e and ρ_e·vn once per stage | 13.6 ms | 59.8 ms |
| plus Hilbert ordering | 12.5 ms | 49.2 ms |
| plus Exner pressure of the base state once per step, edge-first Int32 TRiSK tables, blocks of 32 points × 8 levels | 9.1 ms | 27.3 ms |
| plus the pentadiagonal θ solve with rows built on the fly | 7.7 ms | 25.1 ms |

On hexagon 7, `vertical_solve` takes 5.1 ms, `vn_tendency` 4.5 ms and `theta_interface` 2.3 ms.

## Run

```sh
julia --project examples/baroclinic_wave.jl 5 10 300 30
sbatch --account=bb1143 examples/gpu/dycores.sbatch      # ICON and TRiSK on one A100
julia --project examples/thermal_wind.jl trisk 4 5 300
```
