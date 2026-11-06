# ICON dycore: triangular C grid, predictor–corrector (`ICONNonhydrostatic`)

Source: [src/Nonhydrostatic/ICON/](../../src/Nonhydrostatic/ICON/). Shared with the other
dycores: the vertical grid, the column solver, the C-grid operators and the test cases
([src/Nonhydrostatic/Common/](../../src/Nonhydrostatic/Common/)).

## Discretization

Dry compressible Euler equations, shallow atmosphere, flat ground, on the extruded triangular C grid
of ICON (Zängl et al. 2015): `vn` at the edges × layers, `w` at the cells (circumcentres) ×
interfaces, `ρ`, `ρθ` and the prognostic Exner pressure `exner` at the cells × layers. The
defaults of `ICONNonhydrostatic` are those of ICON's namelists (`nonhydrostatic_nml`,
`diffusion_nml`, `dynamics_nml`), and the model uses ICON's constants (`mo_physical_constants`:
`ICON_RD = 287.04`, `ICON_CPD = 1004.64`, `ICON_GRAV = 9.80665`, radius 6.371229e6 m,
Ω = 7.29212e-5 s⁻¹); the analytic test cases keep those of dmplex (Rd = 287, g = 9.80616), as
the DCMIP and JW formulas do inside ICON.

**Velocity tendencies** (`mo_velocity_advection`), per layer with the operators of
`ICONShallowWater`:

- tangential velocity `vt` by vector RBF (4 edges), kinetic energy `K_e = (vn² + vt²)/2`, cell
  kinetic energy with the bilinear weights `e_bln_c_s`;
- `∂vn/∂t = (f + ζ_e) vt − [K_e(a₁ − a₂) + a₂K_r − a₁K_l] − w_e ∂vn/∂z`, the kinetic-energy gradient
  of ICON (`coeff_gradekin`), ζ_e the mean of the two vertex vorticities, `vn` at the interfaces by
  `wgtfac_e`, at the ground by the quadratic extrapolation `wgtfacq_e`;
- `∂w/∂t = −Σ_e b_{i,e}(vn ∂w/∂n + vt ∂w/∂t)_e − w ∂w/∂z` with ICON's second-order `∂w/∂z` on
  unequal layers (`coeff1_dwdz`, `coeff2_dwdz`), `w` at the vertices by area weights.

**Fast waves** (`solve_nh`):

- horizontal pressure gradient `−c_p θ_e ∂π/∂n` of the Exner pressure extrapolated in time;
- mass flux `ρ_e v̄` and `ρθ` flux `ρ_e θ_e v̄`, with `ρ_e`, `θ_e` from Miura's second-order upwind
  reconstruction (upwind cell plus its Green–Gauss gradient at the backtrajectory point
  `x_e − (Δt/2)(vn n + vt t)`);
- `v̄` is ICON's averaged normal velocity (`e_flx_avg`, `z_vn_avg`, `flux_average`): five weights
  per edge (the edge and the other two edges of each of its cells) such that the divergence of
  `v̄` is the cell average `C` of the divergence of `vn`. `C` (`c_bln_avg`) gives the cell
  `divavg_cntrwgt = 0.5` and its three neighbours the weights that reproduce linear fields,
  iteratively corrected to conserve the area integral. `D v̄ = C D vn` fixes the outer weights;
  the central one follows in each of the two cells, and ICON takes the mean; finally the weights
  are scaled to reproduce a constant vector. It damps the grid-scale divergence of the triangles
  (that of a random field to 41 % on R2B4) and keeps smooth winds (solid-body rotation to 1e-4).
  Without it the model was unstable at ICON's time steps (see below);
- vertically implicit per column, as `solve_nh`: the Exner pressure linearized around `π^n`,
  `Δπ = −z_beta Δ_z(θ_j ρ_j w_j)` per layer with `z_beta = Δt R π^n/(c_v (ρθ)^n Δz)`; the vertical
  pressure gradient split around ICON's reference atmosphere (`h_scal_bg = 10 km`,
  `t0sl_bg = 288.15 K`, `del_t_bg = 75 K`), `(1 − β) θ_j ∂π'/∂z + θ'_j ∂π_ref/∂z` explicit with
  the analytic `∂π_ref/∂z = −g/(c_p θ_ref)` and `β θ_j ∂π'/∂z` implicit, `β = 0.5 + vwind_offctr`;
  ρ and θ linearly interpolated to the interfaces (`wgtfac_c`); then
  `θ = (ρθ)^n((π/π^n − 1) c_v/R + 1)/ρ`. Klemp's Rayleigh damping of w above `damping_height`
  (`rayleigh_type = 2`, `rayleigh_w = rayleigh_coeff (1 − tanh(3.8 (z_top − z)/(z_top − z_damp)))`)
  after the solve.

**Time stepping** (`step!`), the two-time-level predictor–corrector of ICON (`solve_nh`,
`istep = 1, 2`), the only time stepper of this dycore:

1. *Predictor:* velocity tendencies at n: with the default `itime_scheme = 4` those of the last
   corrector (computed at the predictor state of the previous step), all recomputed in the first
   step and that of vn in the first of every `ndyn_substeps` steps (ICON `l_recompute`); with
   `itime_scheme = 6` recomputed every step; pressure gradient of the Exner pressure
   extrapolated in time, `π^n + exner_expol (π^n − π^{n−1})` (`exner_expol = 1/3`); `ρ_e`, `θ_e`
   at n; `vn* = vn^n + Δt (T^n − c_p θ_e ∂π/∂n)`; fluxes of `vn*` and the column solve with ρ, θ of
   n give `w*, ρ*, π*, ρθ*`.
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

**Diffusion** (`mo_nh_diffusion`, ICON's default `hdiff_order = 5`), after every
`ndyn_substeps = 5` dynamics steps over the advective time step `Δt_a = 5 Δt`, as ICON calls it
after the dynamics substeps:

- velocity vectors at the vertices by vector RBF (Gaussian kernel, `rbf_vec_scale_v = 0.5`), the
  edge normal and tangent projected onto the vertex planes;
- Smagorinsky coefficient at the edges from the deformation on the quadrilateral of the edge (its
  two vertices and the opposite vertices of its two cells), `K = s(z) Δt_a |D| − k₄ ndyn_substeps/4`,
  limited by `1/8 − 4k`, with the height profile `s(z)` of `hdiff_smag_fac…fac4` (0.015 below 32.5
  km);
- `vn += A_e (K ∇²vn − A_e k ∇⁴vn)` with the background coefficient
  `k = min(1/128, ndyn_substeps/(3·64·hdiff_efdt_ratio))`, `hdiff_efdt_ratio = 36`
  (`itype_vn_diffu = 1`);
- `w −= k_w A² ∇⁴w`, `k_w = min(1/48, ndyn_substeps/(36·hdiff_w_efdt_ratio))`, plus ICON's `∇²w`
  between the lowest interface above `damping_height` and the top;
- `θ += A ∇⋅(K ∇θ)` at constant ρ, with `K` raised over grid-point cold pools in the lowest two
  layers, and the Exner pressure updated with it, `π ← π (1 + R/c_v (θ_new/θ − 1))`.

`hdiff_order = −1` switches it off. `hyperdiffusion = ν₄` adds `−ν₄ L(L(vn))` with the C-grid
vector Laplacian to every dynamics step.

The quadrilateral stencils (deformation, `∇²vn`) are exact only where the quadrilateral is a
rhombus. On the icosahedral meshes of `create_spherical_mesh`, which are not spring optimized as
ICON's grids, a solid-body rotation shows a deformation of up to 0.17/a at the most distorted edges
and below 0.01/a where the quadrilateral is a rhombus (tested). The ICON grid files
(`icon_grid_00xx_R02Byy_G.nc`) are read with `read_grid` and give exactly ICON's geometry.

**Differences to ICON:** flat ground only (ICON's `jabw` test has the JW orography with
terrain-following levels; `dcmip_bw_11` is flat); the vn diffusion `itype_vn_diffu = 1` (ICON's
jabw10d uses 2); no extra CFL diffusion (`lextra_diffu`); no tracers or physics (`ndyn_substeps`
only sets when the diffusion is called); no nest boundaries; serial.

## Comparison with Fortran ICON

Fortran ICON 2026.10 (GPU build on Levante) with the namelist of its jabw10d experiment on the
flat DCMIP 2016 baroclinic wave `dcmip_bw_11` (R2B5 `icon_grid_0051_R02B05_G.nc`, 64 SLEVE levels
to 35 km, `icon_vertical_grid`), with the separate vn tendencies `ddt_vn_*` in the output; our
model started from ICON's state (`read_icon_state`, cells and edges matched by position).

One step from ICON's state, the difference to ICON relative to ICON's own change in the step:

| | dtime 120 s, `ndyn_substeps = 1` | dtime 576 s, 5 substeps |
|---|---|---|
| w | 0.1 % | 0.1 % |
| ρ | 0.08 % | 0.4 % |
| θ_v | 0.1 % | 0.6 % |
| exner | 0.16 % | 0.9 % |
| vn | 1.6 % | 12 % (the vn diffusion, `itype_vn_diffu`) |

The tendencies of vn on ICON's state: pressure gradient to 3e-5, advection and Coriolis to 0.9 %
(ICON's mixes in the corrector), divergence damping to 5 % of ICON's `ddt_vn_*`. Two
differences to ICON were found this way and removed: the flux averaging `e_flx_avg`, missing
before, and the constants. ICON's DCMIP initial state is hydrostatic with the constants of the
DCMIP routine (Rd = 287, g = 9.80616); with ICON's g = 9.80665 it keeps a vertical force of
0.87e-3 m/s², with g = 9.80616 one of 1.36e-3 m/s², which made our w 1.5 times ICON's.

Ten days of `dcmip_bw_11` with the jabw10d namelist (R2B5, 64 levels, dtime 576 s), our model
started from ICON's initial state, `examples/icon_jabw.jl --init --compare` on one A100:

| | Fortran ICON | ICONNonhydrostatic |
|---|---|---|
| minimum p_s at day 10 | 936.86 hPa | 936.84 hPa |
| max \|w\| at day 10 | 0.077 m/s | 0.078 m/s |
| p_s difference at day 10 | | rms 0.10 hPa, max 1.84 hPa |
| time for 10 days, 1 A100 | 81.2 s (time loop 79.6 s, 10.6 ms per dynamics step) | 112 s (14.9 ms per dynamics step with the compilation, 13.5 ms without) |
| time for 10 days, 4 A100 | 33.5 s | |

Fortran ICON on 1 A100 (one compute and one I/O process, `exp.dcmip10d_1gpu.run`) gives the same
result as on 4 A100, bit for bit.

### Time per section

Both models timed in ICON's sections of `solve_nh` on one A100, with the GPU synchronized at
the start and end of each section: Fortran ICON with `timers_level = 11` (`!$ACC WAIT` in the
timers, 1.6 % slower than with 10), ours with `--timer` (`section_timer!`, TimerOutputs), after
the first advective step. The sections are grouped as in ICON: veltend the velocity tendencies;
cellcomp θ, π and the extrapolated π; vnupd ρ_e, θ_e, the pressure gradient, the vn update and
the divergence damping; edgecomp the averaged vn and the fluxes; vimpl the flux divergence and
the column solve (in ICON the boundaries of cellcomp, vnupd and edgecomp lie somewhat
differently: the interface values are in cellcomp, part of the divergence damping in edgecomp).
Time per dynamics step, 10 days of the R2B5 wave (7500 steps):

| section | Fortran ICON | ICONNonhydrostatic |
|---|---|---|
| veltend | 1.84 ms | 1.57 ms |
| cellcomp | 1.14 ms | 0.39 ms |
| vnupd | 1.99 ms | 2.15 ms |
| edgecomp | 1.51 ms | 0.76 ms |
| vimpl | 3.40 ms | 6.99 ms (column solve 5.98, divergence 0.55, interfaces 0.48) |
| rest of nh_solve | 0.17 ms | 0.62 ms (copies of the state and tendencies) |
| nh_hdiff | 0.48 ms | 0.99 ms |
| step | 10.8 ms | 13.5 ms |

The difference is the column solve, `icon_vertical_solve_kernel!` with one thread per
column (20480 threads on R2B5) that does all of the column sequentially: the explicit ρ and π,
the coefficients, the Thomas sweeps, the update of ρ, π, ρθ. ICON computes all but the
sweeps in kernels over cells × levels. The workgroup size is not the cause (32, 64, 128 instead
of the default: 3.96 ms per call instead of 2.99 ms).

The minimum surface pressure of our model by day: 999.67, 999.58, 999.41, 999.00, 998.19,
996.48, 992.74, 983.99, 961.83, 936.84 hPa (days 1 to 10).

Without the flux averaging the model was unstable at ICON's time steps: on R2B4 (dtime 1152 s)
within 50–80 steps, at 480 s after 800 steps with the w diffusion and the 3D divergence damping;
more divergence damping only delayed it. With it both run stably (600 and 1500 steps, max |w|
below 1e-3 m/s).

## Earlier results

With the scheme before ICON's numerics (ρθ prognostic, the column solver of MPAS-Gassmann, no flux
averaging), the constants of dmplex and `create_spherical_mesh`:

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
# ICON's jabw10d settings; from a Fortran ICON state and compared with it at the end
julia --project examples/icon_jabw.jl grid/icon_grid_0013_R02B04_G.nc 10 1152 64 35000
sbatch --account=bb1143 examples/gpu/icon_jabw.sbatch icon_grid_0051_R02B05_G.nc --init=icon.nc --compare=icon.nc
sbatch --account=bb1143 examples/gpu/icon_jabw.sbatch icon_grid_0051_R02B05_G.nc --timer   # time per section
```
