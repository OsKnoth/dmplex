# ICON dycore: triangular C grid, predictor–corrector (`ICONNonhydrostatic`)

Source: [src/Nonhydrostatic/ICON/](../../src/Nonhydrostatic/ICON/). Shared with the other
dycores: the vertical grid, the column solver, the C-grid operators and the test cases
([src/Nonhydrostatic/Common/](../../src/Nonhydrostatic/Common/)).

## Discretization

Dry compressible Euler equations, shallow atmosphere, flat ground, on the extruded triangular C grid
of ICON (Zängl et al. 2015): `vn` at the edges × layers, `w` at the cells (circumcentres) ×
interfaces, `ρ`, `ρθ` at the cells × layers.

**Velocity tendencies** (`mo_velocity_advection`), per layer with the operators of
`ICONShallowWater`:

- tangential velocity `vt` by vector RBF (4 edges), kinetic energy `K_e = (vn² + vt²)/2`, cell
  kinetic energy with the bilinear weights `e_bln_c_s`;
- `∂vn/∂t = (f + ζ_e) vt − [K_e(a₁ − a₂) + a₂K_r − a₁K_l] − w_e ∂vn/∂z`, the kinetic-energy gradient
  of ICON (`coeff_gradekin`), ζ_e the mean of the two vertex vorticities, `vn` at the interfaces by
  `wgtfac_e`;
- `∂w/∂t = −Σ_e b_{i,e}(vn ∂w/∂n + vt ∂w/∂t)_e − w ∂w/∂z`, `w` at the vertices by area weights.

**Fast waves** (`solve_nh`):

- horizontal pressure gradient `−c_p θ_e ∂π/∂n`;
- mass flux `ρ_e vn` and `ρθ` flux `ρ_e θ_e vn`, with `ρ_e`, `θ_e` from Miura's second-order upwind
  reconstruction (upwind cell plus its Green–Gauss gradient at the backtrajectory point
  `x_e − (Δt/2)(vn n + vt t)`);
- vertically implicit per column (off-centring 0.65 = 0.5 + `vwind_offctr`), linearized Exner
  pressure: the column solver shared with the other dycores, with ρ and θ at the interfaces from
  ICON (below).

**Time stepping** (`step!`), the two-time-level predictor–corrector of ICON (`solve_nh`,
`istep = 1, 2`), the only time stepper of this dycore:

1. *Predictor:* velocity tendencies at n: with the default `itime_scheme = 4` those of the last
   corrector (computed at the predictor state of the previous step; recomputed only in the first
   step), with `itime_scheme = 6` recomputed every step; pressure gradient of the Exner pressure
   extrapolated in time, `π^n + exner_expol (π^n − π^{n−1})` (`exner_expol = 1/3`); `ρ_e`, `θ_e`
   at n; `vn* = vn^n + Δt (T^n − c_p θ_e ∂π/∂n)`; fluxes of `vn*` and the column solve with ρ, θ of
   n linearly interpolated to the interfaces give `w*, ρ*, ρθ*`.
2. *Corrector:* velocity tendencies at the predictor state, weighted 0.25 / 0.75
   (`veladv_offctr = 0.25`); the same pressure gradient and `ρ_e`, `θ_e`;
   `vn^{n+1} = vn^n + Δt (0.25 T^n + 0.75 T* − c_p θ_e ∂π/∂n)` plus the divergence damping of the
   predictor state; fluxes of `vn^{n+1}` and a second column solve with the weighted w tendency
   and ρ, θ at the interfaces from the time average `0.6 x^n + 0.4 x*` (`rhotheta_offctr = −0.1`),
   upwind biased along the vertical backtrajectory over Δt/2 with `w*`.

**Divergence damping** (ICON defaults `divdamp_order = 4`, `divdamp_type = 3`): fourth order of
the 3D divergence of the predictor state, `−k₄(z) Ā² ∇(∇⋅∇(∇⋅vn* + ∂w*/∂z))`, with ICON's height
profile `k₄(z)`: `divdamp_fac = 0.0025` up to 32.5 km, linear to 0.004 at 40 km, quadratic through
`divdamp_fac3`, `divdamp_fac4` at 60 and 80 km. `divdamp_type = 2` damps the 2D divergence,
`divdamp_type = 32` blends from 3D to 2D between 12.5 and 17.5 km,
`divdamp_order = 2` or `24` adds the second-order term `divdamp_fac_o2 Ā ∇D` (ICON uses it only in
the spin-up of real-data runs; in test cases `divdamp_fac_o2 = 0`).

**Diffusion** (`mo_nh_diffusion`, `hdiff_order = 5`, ICON defaults; implemented but not yet
the default, see below), after every `ndyn_substeps = 5`
dynamics steps over the advective time step `5 Δt`, as ICON calls it after the dynamics substeps:

- velocity vectors at the vertices by vector RBF (Gaussian kernel, `rbf_vec_scale_v = 0.5`), the
  edge normal and tangent projected onto the vertex planes;
- Smagorinsky coefficient at the edges from the deformation on the quadrilateral of the edge (its
  two vertices and the opposite vertices of its two cells), `K = s(z) Δt_a |D| − k₄ ndyn_substeps/4`,
  limited by `1/8 − 4k`, with the height profile `s(z)` of `hdiff_smag_fac…fac4` (0.015 below 32.5
  km);
- `vn += A_e (K ∇²vn − A_e k ∇⁴vn)` with the background coefficient
  `k = min(1/128, ndyn_substeps/(3·64·hdiff_efdt_ratio))`, `hdiff_efdt_ratio = 36`;
- `w −= k_w A² ∇⁴w`, `k_w = min(1/48, ndyn_substeps/(36·hdiff_w_efdt_ratio))`, plus ICON's `∇²w`
  above `damping_height`, `n2w ≤ 1/12`. For the highest mode, a checkerboard of `A∇²` with
  eigenvalue −ρ, one call multiplies w by `1 − k_w ρ² − n2w ρ`. ICON's `k_w = 5/540` makes
  `k_w ρ² = 1` on equilateral triangles (ρ = 10.4): the mode is removed without overshoot. On
  `create_spherical_mesh` ρ ≈ 11.45 and the factor is −0.2 to −0.74 near the top; a negative
  factor flips w against ρ and θ every call, and this drove the checkerboard of w up within a
  day (R2B4, w only: NaN after 365 steps). So `k_w ≤ 1/ρ²` and `n2w ≤ (1 − k_w ρ²)/ρ`, with the
  Gershgorin bound `ρ = 2 max Σ l/d`; on R2B4 `k_w` drops from 0.0093 to 0.0073 and the `∇²w`
  term vanishes;
- `θ += A ∇⋅(K ∇θ)` at constant ρ, with `K` raised over grid-point cold pools in the lowest two
  layers.

`hdiff_order = −1`, the default for now, switches it off. `hyperdiffusion = ν₄` adds the former
`−ν₄ L(L(vn))` with the C-grid vector Laplacian to every dynamics step; the example uses it.

**Open problem:** with ICON's diffusion the baroclinic wave on R2B4 (dt 150 s) blows up by day 2,
also with the jww settings (`hdiff_efdt_ratio = 10`, `hdiff_smag_fac = 0.15`). The growing mode is
grid scale (`A|∇²w|/|w| ≈ 4–6`) and starts at 26–30° latitude in both hemispheres, around the
pentagons of the icosahedron, in the mid troposphere; without any diffusion a slower mode grows
near the ground under the jet. `divdamp_type = 2` stabilizes the w-only diffusion, `divdamp_type
= 32`, `itime_scheme = 6`, `ndyn_substeps = 1` or a smaller `k_w` do not. Likely causes, not yet
separated: the irregular cells around the pentagons (ICON's grids are spring optimized), and ρθ
instead of ICON's prognostic Exner pressure and θ_v.

The quadrilateral stencils (deformation, `∇²vn`) are exact only where the quadrilateral is a
rhombus. On the icosahedral meshes of `create_spherical_mesh`, which are not spring optimized as
ICON's grids, a solid-body rotation shows a deformation of up to 0.17/a at the most distorted edges
and below 0.01/a where the quadrilateral is a rhombus (tested).

**Differences to ICON:** ρθ is prognostic and π follows from the equation of state (ICON carries
`exner` and `θ_v` and updates `exner` with the linearized equation); no reference state (with flat
ground its horizontal gradient vanishes); no tracers or physics (`ndyn_substeps` only sets when the
diffusion is called); no nest boundaries; serial.

## Results

Rest state at rest (w ~ 1e-14), steady jet and thermal wind steady (triangles 4: max |w| 1.7e-3 and
8e-4 m/s over 2 and 3 days), mass to round-off (tested). Time step about ICON's dynamics substep:
150 s on R2B4, 75 s on R2B5, 37.5 s on R2B6 (300 s on R2B4 is unstable).

Baroclinic wave (Ullrich et al. 2014), 30 layers, minimum surface pressure (hPa), with the
fourth-order diffusion `hyperdiffusion` (R2B5 and R2B6 with the stepper before the
`rhotheta_offctr` interfaces and the 3D divergence damping; on R2B4 these changed day 10 by 1.2 hPa):

| Day | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|
| R2B4 (triangles 5, 20480 cells, ≈ 158 km), dt 150 s | 994.39 | 986.12 | 970.81 | 952.66 | 925.97 |
| R2B5 (triangles 6, 81920 cells, ≈ 79 km), dt 75 s | 992.63 | 981.41 | 961.49 | 939.03 | 918.16 |
| R2B6 (triangles 7, 327680 cells, ≈ 39 km), dt 37.5 s | 991.88 | 979.15 | 958.49 | 937.32 | 917.30 |

Without diffusion, R2B4 (dt 150 s) reaches 993.4 hPa at day 6 and blows up at day 7.

## Time per step on one A100 (30 layers)

| mesh | cells | edges | dt | time per step | 10 days |
|---|---|---|---|---|---|
| R2B5 | 81920 | 122880 | 75 s | 5.2 ms | 60 s |
| R2B6 | 327680 | 491520 | 37.5 s | 18.8 ms | 433 s |

For comparison, TRiSK in the same job: hexagon 6 (40962 cells, 122880 edges, dt 300 s) 7.4 ms
per step and 21 s for 10 days; hexagon 7 (163842 cells, dt 150 s) 24.5 ms and 141 s. Per step,
ICON with the same number of edges is faster (two velocity-tendency evaluations but no RK stages);
per simulated day it is slower because its time step is four times smaller.

## Run

```sh
julia --project examples/baroclinic_wave.jl 5 10 150 30 --icon
sbatch --account=bb1143 examples/gpu/dycores.sbatch      # ICON and TRiSK on one A100
julia --project examples/thermal_wind.jl icon 4 5 300
```
