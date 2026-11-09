# MPAS-Gassmann dycore: hexagonal C grid, ICON-IAP form (`MPASGassmannNonhydrostatic`)

Source: [src/Nonhydrostatic/MPAS/Gassmann/](../../src/Nonhydrostatic/MPAS/Gassmann/), with the TRiSK
operators of [src/Nonhydrostatic/MPAS/Operators.jl](../../src/Nonhydrostatic/MPAS/Operators.jl).
Shared with the other dycores: the vertical grid, the column solver, the C-grid operators, the RK3
driver and the test cases ([src/Nonhydrostatic/Common/](../../src/Nonhydrostatic/Common/)). Formerly
`TRiSKNonhydrostatic` (still an alias).

The ICON-IAP model of Gassmann (2013) on the Voronoi meshes of MPAS, with ICON-IAP's
predictor–corrector and its pentadiagonal vertical solve (Gassmann and Herzog 2008); the
Wicker–Skamarock RK3 of the other dycores and the tridiagonal vertical solve remain options. The
standard MPAS-A dycore (MPAS-Orig, split-explicit) is planned as a separate model.

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
- **Vertical:** implicit per column with the linearized Exner pressure. `vertical_solver =
  :five_band` (default, ICON-IAP `divergent_modes_5band`): the unknown is the vertical mass flux
  `m = (ρ̄ w⁰ + ρ⁰ w)/2`, time-centred in the products of the new density and the old `w` and
  vice versa, the vertical kinetic energy `w⁰ w/2` implicit, so `∂K_w/∂z` couples five mass fluxes:
  one pentadiagonal system per column. `:tridiagonal`: one tridiagonal system for `w`, `K_w`
  explicit. θ at the interfaces third-order upwind after a Crank–Nicolson vertical advection
  (ICON-IAP `ltheta_up_vert`). ICON-IAP's divergence damping in the sponge (`div_updamp_coeff`, 0 by
  default) is replaced by the Rayleigh damping of `w`.
- **Time stepping:** `time_stepping = :iap` (default): ICON-IAP's two stages, both from the old state
  over the whole step, the second with the mean slow tendencies of both, the horizontal pressure
  gradient with the Exner pressure off-centred in time, `(R/c_p − ½) π^{n−1} + (3/2 − R/c_p) π^n`.
  `:wicker_skamarock`: RK3 with a HEVI split. Horizontal mass fluxes forward–backward in both.
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

(RK3, tridiagonal vertical solve.) The GPU and CPU runs of hexagon 6 agree; mass is conserved to 4e-16.

Hexagon 5, dt 300 s, CPU, minimum surface pressure at day 10 and time per step:

| time stepping | vertical solve | p_s day 10 | per step |
|---|---|---|---|
| ICON-IAP (default) | five-band (default) | 922.11 hPa | 334 ms |
| ICON-IAP | tridiagonal | 922.06 hPa | 321 ms |
| RK3 | tridiagonal | 921.79 hPa | 394 ms |

With dt 600 s every combination blows up on the first day (the horizontal acoustic limit of the
explicit horizontal step).

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
julia --project examples/baroclinic_wave.jl 5 10 300 30            # --rk3, --tridiagonal for the options
sbatch --account=bb1143 examples/gpu/dycores.sbatch      # ICON and TRiSK on one A100
julia --project examples/thermal_wind.jl trisk 4 5 300
```
