# ICON dycore: triangular C grid, predictor–corrector (`ICONNonhydrostatic`)

Source: [src/Nonhydrostatic/ICON/](../../src/Nonhydrostatic/ICON/). Shared with the other
dycores: the vertical grid, the column solver and the test cases (`src/Nonhydrostatic/`).

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
- vertically implicit per column (off-centring 0.65 = 0.5 + `vwind_offctr`), centred θ and ρ at the
  interfaces, linearized Exner pressure: the column solver shared with TRiSK.

**Time stepping** (`step!`), the two-time-level predictor–corrector of ICON:

1. *Predictor:* velocity tendencies at n; pressure gradient of the Exner pressure extrapolated in
   time, `π^n + exner_expol (π^n − π^{n−1})` (`exner_expol = 1/3`); `ρ_e`, `θ_e` at n;
   `vn* = vn^n + Δt (T^n − c_p θ_e ∂π/∂n)`; fluxes of `vn*` and the column solve give `w*, ρ*, ρθ*`.
2. *Corrector:* velocity tendencies at the predictor state, weighted 0.25 / 0.75
   (`veladv_offctr = 0.25`); the same pressure gradient and `ρ_e`, `θ_e`;
   `vn^{n+1} = vn^n + Δt (0.25 T^n + 0.75 T* − c_p θ_e ∂π/∂n)` plus the fourth-order divergence
   damping of `vn*`, `−divdamp_fac Ā² ∇(∇⋅∇(∇⋅vn*))` (`divdamp_fac = 0.0025`); fluxes of `vn^{n+1}`
   and a second column solve with the weighted w tendency.

**Diffusion:** fourth-order `−ν₄ L(L(vn))` with the C-grid vector Laplacian, in place of ICON's
`mo_nh_diffusion`. The example scales it like ICON's coefficient (∝ A²/Δt ∝ Δx³):
ν₄ = 5e14 m⁴/s · (Δx / 79 km)³. Without it the baroclinic wave blows up after days 3–7, as ICON does
without horizontal diffusion.

**Differences to ICON:** ρθ is prognostic and π follows from the equation of state (ICON carries
`exner` and `θ_v` and updates `exner` with the linearized equation); no reference state (with flat
ground its horizontal gradient vanishes); no dynamics substeps, tracers or physics; the 2D
divergence in the divergence damping; serial.

## Results

Rest state at rest (w ~ 1e-14), steady jet and thermal wind steady (triangles 4: max |w| 1.7e-3 and
8e-4 m/s over 2 and 3 days), mass to round-off (tested). Time step about ICON's dynamics substep:
150 s on R2B4, 75 s on R2B5, 37.5 s on R2B6 (300 s on R2B4 is unstable).

Baroclinic wave (Ullrich et al. 2014), 30 layers, minimum surface pressure (hPa), with the
fourth-order diffusion:

| Day | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|
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
