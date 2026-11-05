# Plan: 3D kite model as a direct product

The nonhydrostatic compressible Euler equations on the extruded kite mesh. The spaces are tensor
products of the 2D kite spaces of `NonlinearKiteShallowWater` with 1D spaces on the vertical
grid, in the split-order form of Witt, Bendall & Shipton (arXiv:2603.16571), with Lorenz staggering
as in ICON and `TRiSKNonhydrostatic`. The 3D dual-cell method of Codecasa, Kapidani, Schöberl &
Wess (arXiv:2604.13921, Maxwell on tetrahedra with barycentric duals) is not used: its dual cells
cut through the columns, so there is no column structure for a HEVI solver.

## Spaces

Horizontal factors (2D kite spaces, order k):
- H_h = `KiteRadau(k)`: Q_k on the kites with Gauss–Radau nodes, broken across primal edges.
  Diagonal mass matrix.
- U_h = `KiteBrokenRT(k)`: broken H(div) on the dual cells. Block-diagonal mass matrix, one block
  per dual cell.

Vertical factors (order v on the layers of `VerticalGrid`): DG_v per layer, CG_{v+1} continuous
across the interfaces.

| Field | Space | Lowest order (k = 1, v = 0) |
|---|---|---|
| ρ | H_h ⊗ DG_v | Q2 Radau × constant per layer |
| ρθ | H_h ⊗ DG_v (Lorenz: with ρ) | as ρ |
| w | H_h ⊗ CG_{v+1} | Q2 Radau × linear, values at the interfaces |
| u_h | U_h ⊗ DG_v | RT1 kite × constant per layer |

Together (u_h, w) ∈ V1 = (U_h ⊗ DG_v) ⊕ (H_h ⊗ CG_{v+1}) and ρ, ρθ ∈ V2 = H_h ⊗ DG_v: the
divergence maps V1 into V2, so the discrete continuity equation is exact per kite column.

**Lorenz staggering.** θ lives in the layers with ρ, and the pressure gradient at the w points
interpolates θ to the interfaces. This is the layout of ICON and of `TRiSKNonhydrostatic`, with its
computational mode in the vertical; the upwind θ at the interfaces of `theta_interface_kernel!`
(ICON-IAP `ltheta_up_vert`) carries over. Higher-order reconstruction of θ inside the DG_v layers is
possible later. That is the reason for Lorenz rather than Charney–Phillips (θ with w), which the
split-order paper uses.

## Mass matrices and columns

The Radau nodes of H_h are both the dofs and the quadrature points, so the horizontal mass
matrix of ρ, ρθ, w is diagonal. The vertical terms (∂_z of the mass and ρθ fluxes, the vertical
pressure gradient, buoyancy) involve no horizontal derivative, so they decouple into **independent
columns, one per horizontal Radau node**. For v = 0 a column is the same tridiagonal system in w,
after eliminating ρ and ρθ, as in `vertical_solve_kernel!`. For v > 0 it is a block-banded system
of width v + 1.

The horizontal momentum has the mass matrix (dual-cell block of U_h) ⊗ (layer block of DG_v), so
it can be inverted block-diagonally, which allows explicit horizontal terms.

## Time stepping (HEVI)

As in `TRiSKNonhydrostatic`: Wicker–Skamarock RK3. In each stage:
1. **Explicit horizontal terms**, per layer, with the 2D kite operators of
   `NonlinearKiteShallowWater` and ρ in place of h:
   - mass and ρθ fluxes in U_h;
   - vorticity in the continuous kite space and the PV term;
   - kinetic energy;
   - horizontal pressure gradient c_p θ ∇π, tested with U_h.
2. **Implicit vertical terms**, per Radau column: w, ρ, ρθ at t + τ, linearized around the Exner
   pressure of the base state. π0 and κπ0/ρθ0 are computed once per step.
3. **Vertical advection of u_h and w**, explicit, in the ICON-IAP mass-flux form.

## Steps

Done (`src/Nonhydrostatic/KiteNonhydrostatic.jl`, tests in `test/extruded.jl`): steps 1 and 2 at
lowest vertical order, serial. The state is `(u, w, ρ, ρθ)` as arrays (dofs × levels).

1. **Extruded kite spaces** (done: `zero_state`, `project_layers`, `project_velocity`,
   `mass_divergence!`; mass to round-off, column sums exact): dof numbering of H_h ⊗ DG_v, H_h ⊗ CG_{v+1} and U_h ⊗ DG_v on top of
   `create_kite_mesh` and `VerticalGrid`, with layer-innermost or point-innermost layouts for the
   GPU (blocks of points × levels, as `BLOCK` in `TRiSKNonhydrostatic`). Tests: dimensions, and the
   divergence V1 → V2 exact per column.
2. **Column solver for v = 0** (done: `column_points`, `initial_state` with `balanced_column`,
   `vertical_stage!`, `vertical_step!`; rest state w < 1e-12 with `:upwind` and `:centred`, column
   mass to round-off; `:iap` leaves rest as in `TRiSKNonhydrostatic`): reuse `vertical_solve_kernel!` and `theta_interface_kernel!` with
   one column per Radau node. Tests: hydrostatic rest state at rest, and a column at rest staying
   at rest with upwind θ.
3. **Horizontal terms per layer** (done without vertical advection: `horizontal_tendency!`, `stage!`,
   `step!`; F = Π_U(ρu), Fθ = Π_U(θF), pressure gradient −c_p Π_U(θ∇π) with ∇π = M_u⁻¹Bᵀπ/a, which
   makes the energy exchange exact (residual 1e-16 of the summed products); rest state at rest.
   The sub-cell instability of the rest state, see below, came from the quadrature of the velocity
   mass and is fixed) from the kite shallow-water operators. Tests: mass conservation
   to round-off, energy exchange between the horizontal and vertical terms.
4. **Test cases:**
   - inertia-gravity wave (Skamarock–Klemp on the plane with `create_periodic_plane`, the
     Baldauf–Brdar sphere version);
   - the baroclinic wave, compared with `TRiSKNonhydrostatic` on the same hexagon mesh.
5. **GPU** with KernelAbstractions, as the 2D kite model.
6. **Higher vertical order** (v = 1) and the θ reconstruction in the layers.

## Sub-cell instability on non-affine kites (2026-11-03, solved)

**Cause:** the velocity mass `M_U` was assembled with the default rule of the space (degree 6), the
projections `Π_U(h v)` with the 3×3 rule of `KiteQuadrature` (degree 4). On non-affine kites the
metric `JᵀJ/det J` is rational, so the two differ and `Π_U(c v) = c M_U⁻¹ R v ≠ c v` for a constant
`c` (1.4e-2 on hexagon 3, 1e-4 on the cubed sphere from the curvature). Then the ρθ flux
`Π_U(θ̄ F) = θ̄ P F` is not `θ̄` times the mass flux `F`: the horizontal fluxes create sub-cell `θ′`
(entropy) at rest. The energy stays conserved, but the Casimir `∫ ρ f(θ)` that makes the stratified
rest state stable does not, and the mode lives on the released potential energy. On affine kites
(cubed sphere) the integrand is a polynomial, both rules are exact and `P = I`.

**Fix:** `NonlinearKiteShallowWater(...; mass_degree)`, and `KiteNonhydrostatic` passes the degree
of the projections, so `Π_U(c v) = c v` to round-off (tested). The noise on the rest state now
decays on hexagons 3 and 4.

**Still open:** the unperturbed baroclinic wave on hexagon 3 grows from day 1 in the jet (33–39°,
levels 6–7; w 1.8 m/s and Δu 2 m/s at 48 h), while `TRiSKNonhydrostatic` (hexagon 4) stays bounded
for 48 h also without Smagorinsky diffusion (w ≈ 1e-2 m/s). Step 3b, the vertical coupling of the
vector-invariant form (`vertical_advection=:vector_invariant`, default: `-η w` in the `u` equation,
`η ⋅ ṽ - ∂K/∂z` in the `w` equation, exactly adjoint), slows it (w 0.9 m/s at 48 h) but does not
remove it.

The growing jet mode (2026-11-03, open), unperturbed baroclinic wave, dt 150 s, 10 layers:

- **Spatial:** the same growth at dt 50 and 150 s; faster on finer meshes (max|w| at 36 h:
  hexagon 2 0.03, hexagon 3 0.17, hexagon 4 5 m/s).
- **Pictures** (hexagon 3 and cubed sphere 10, 42 h): node-scale noise in w and θ′ (up to ±20 K in
  the upper levels) in the jet at 30–55°, also inside the primal cells; somewhat stronger next to
  primal edges that the flow crosses (40 % more w than next to edges along the flow); on the cubed
  sphere strongest at the cube corners and edges. Not specific to the icosahedral grid.
- **Pairs** (hexagon 3, max|w| at 42 h): RT1 × Q2 (default) 0.55 m/s, RT1 × Q1 1.7, RT1 × Q0
  0.6 (unbalanced from the start, bounded); Wess et al., Q2 × dual Radau 2: 1.6 m/s,
  **Q1 × dual Radau 1: 0.024 m/s, bounded** (TRiSK hexagon 4: 0.01). Velocity choice
  `KiteNonhydrostatic(...; velocity=:dual_radau)` with the exact velocity mass in the rule of the
  projections (`velocity_lumped=false`), so `Π_U(c v) = c v` also holds there.
- **Consistency at t = 0:** ρθ̇ converges at first order and is not concentrated at primal edges;
  the w tendency of the balanced state is 3e-14; u̇ converges in the lower layers, in the upper
  layers it is set by the vertical resolution (second order in Δz, from the discrete hydrostatic
  integration of the initial state, as for TRiSK).
- **Central θ trace on the primal faces:** B pairs the traces of the broken θ from both sides with
  the single-valued normal component of Π_U(θF), and the jump [θ] never enters. The upwind trace,
  see the next section, is the DG-type stabilization.

## Upwind θ trace on the primal faces (2026-11-04)

`KiteNonhydrostatic(...; horizontal_theta=:upwind)` adds the DG upwind correction
`-∫_e η_i |F⋅n|/2 (θ^K - θ^E)` on every primal face to the ρθ flux (`upwind_theta!`, see
[KITE3D_DISCRETIZATION.md](KITE3D_DISCRETIZATION.md)). It conserves ρθ, vanishes for constant θ and
dissipates the θ variance (tested). Serial CPU loop for now.

![max |w| on the steady jet](images/kite3d_jet_growth.png)

**Steady jet** (unperturbed baroclinic wave, 10 stretched layers), max |w|:

| pair, mesh | central | upwind θ |
|---|---|---|
| RT1 × Q2, hexagon 3, 42 h | 0.55 m/s, growing | **0.029 m/s** |
| RT1 × Q2, hexagon 4 (dt 60 s), 6 h / 12 h | 4.1 m/s at 6 h | **0.0025 / 0.004 m/s** |
| Q2 × dual Radau 2, hexagon 3, 42 h | 1.6 m/s | 0.45 m/s, still growing |
| Q1 × dual Radau 1, hexagon 3, 42 h | 0.024 m/s | – |
| Q1 × dual Radau 1, hexagon 4, 12 h | 0.012 m/s | 0.011 m/s |

The growth with resolution is gone for RT1 × Q2: on hexagon 4 the upwind run stays at the TRiSK
level. The node-scale pattern is 20 times weaker but still visible near the pentagons at 50–60°:

![w at 10.6 km after 42 h, central and upwind θ](images/kite3d_jet_w.png)

**Perturbed baroclinic wave**, 30 layers, hexagon 3 (8322 columns, dt 150 s), Q1 × dual Radau 1,
central θ: minimum p_s 989.2 (day 6), 985.1 (7), 981.4 (8), 978.2 (9), 976.6 hPa (day 10); mass to
2e-16. The wave develops but is far too weak (TRiSK hexagon 5: 921.8 hPa at day 10), as expected for
about 100 km columns of first order. The same pair on hexagon 4 grew from 0.03 to 0.25 m/s between
day 1 and 1.5 (w noise, before the wave), so it is not stable at higher resolution either; the run
was stopped. RT1 × Q2 with upwind θ on hexagon 3 is running.

**Thermal wind** (`thermal_wind`, new steady state in exact gradient-wind balance, 30 layers,
5 days, `examples/thermal_wind.jl`): this is the hardest test so far.

![max |w| and θ′ on the thermal wind](images/kite3d_thermal_wind.png)

| model | max \|w\| day 1 | day 2 | day 3 | max \|θ′\| |
|---|---|---|---|---|
| TRiSK, hexagon 4 | 2.6e-3 | 1.2e-3 | 8.4e-4 | 0.35 K at day 5, steady |
| kite RT1 × Q2, central | 1.5e-2 | 1.8 | crashes (p < 0) | 470 K at day 2 |
| kite RT1 × Q2, upwind θ | 7.3e-4 | 1.3e-2 | 0.30 | 31 K at day 3.5 |
| kite dual Radau 1 × Q1, central | 2.7e-3 | 8.4e-3 | 9.5e-2 | 94 K at day 5 |

All kite variants grow, the upwind trace only delays it by about a day. θ′ is largest in the
tropics and subtropics (0–45°, where the wind is strongest), through the whole column (27 K at
15 km for dual Radau 1).

**One layer, θ = 1, p = g(ρθ)²/2** (`equation_of_state=:shallow_water`, the shallow-water test of
the compressible code, 2026-11-04): the horizontal tendencies of one layer agree with
`NonlinearKiteShallowWater` to round-off (u̇ 4e-14, ρ̇ exactly, ρθ̇ = ρ̇ to 3e-15; tested), so the
2D → 3D step of the horizontal terms is correct. But TC2 through the 3D step blows up within two
days, while the 2D model runs: θ − 1 grows from round-off by about 1e9 per day. The one-layer model is
thermal shallow water, and θ′ is unstable there, i.e. already horizontally, without any vertical
coupling:

| one layer, random θ′ = 1e-8 | max \|θ − 1\| after ¼ … 1 day |
|---|---|
| at rest, hexagon 3 | 5e-8, bounded |
| TC2, hexagon 2 | 1.3e-7 → 5e-5 (×7 per ¼ day) |
| TC2, hexagon 3 | 1.8e-6 → 3.5e-2 → NaN (×150 per ¼ day) |
| TC2, cubed sphere 8 | 1.8e-7 → 1.4e-5 (×4 per ¼ day) |

The growth needs the flow and scales like 0.3 |u|/Δx: an advective instability of θ′, the same
signature as the 3D jet mode. The conserved energy bounds u and ρθ but not ρ and θ separately. The
upwind θ trace (×1000 per day instead of 1e9) and the upwind traces of ρ and ρθ
(`horizontal_theta=:upwind_density`, same rate) slow it but do not remove it: most of the mode lies
inside the primal cells, where ρ and θ are continuous. The instantaneous Casimir rate
d/dt Σ M ρ(θ−1)² is zero for smooth θ′ and small for random θ′ (2e-3), so the mode is a coupled
u–ρ–θ′ mode through the pressure term −gρ∇θ′ − gθ′∇ρ, not passive advection.

**Passive tracer** (`tracer=true`, `ρq` transported as ρθ: horizontally `B Π_U(q F)`, vertically
with the mass flux of the column solve; for q = 1 the equations of ρ, tested). One layer, TC2,
hexagon 3:

| | max \|θ − 1\|, max \|q − 1\| after ¼ / ½ / ¾ / 1 day |
|---|---|
| θ = q = 1 exactly, central | identical: 1.4e-13, 1.6e-11, 1.9e-9, 2.2e-7 |
| random θ′, q′ = 1e-8, central | θ′ 1.8e-6 / q′ 2.1e-6, …, 3.5e-2 / 3.4e-2, NaN |
| random θ′, q′ = 1e-8, upwind | θ′ 1.1e-7 / q′ 1.2e-7, …, 1.1e-5 / 2.3e-5 |

The passive tracer grows exactly like θ, although it does not feed back on the flow and its
equation is linear in q′. So the pressure coupling is not the cause: **the transport of a mixing
ratio on the kite height space, `ρq̇ = M_h⁻¹ B Π_U(q F)/a` with the nodal `q = ρq/ρ` and
`ρ̇ = M_h⁻¹ B F/a`, is unstable in a flow**, already in 2D. With q = 1 it keeps ρq = ρ exactly, but
q = 1 is an unstable fixed point. In 3D (steady jet, hexagon 3) q − 1 grows from 1e-14 to 2e-12 in
one day. The fix belongs to the tracer transport (and then θ), e.g. a stabilized (SUPG or upwind DG
with interior terms) or a skew-symmetric form of the advection of q.

**Pure transport with a prescribed velocity** (TC2 solid-body rotation, SSPRK3, one day, ρ
perturbed by 1e-8; the zonal ρ is a steady solution): the kite continuity operator
`ρ ↦ M_h⁻¹ B Π_U(ρ u)` is itself unstable, independent of the time step (dt = 0.1 and 0.3 Δx/u0):

| mesh | max \|ρ − ρ(0)\| after ¼ / ½ / ¾ / 1 day |
|---|---|
| hexagon 2 | 0.05, 0.44, 3.3, 24 |
| hexagon 3 | 0.16, 25, 3e3, 4e5 |
| cubed sphere 8 | 0.03, 0.09, 0.14, 0.20 |

Its eigenvalues have max Re λ ≈ 0.7 u0/Δx on hexagons and 0.28 u0/Δx on the cubed sphere (projected
flux; the pointwise flux of the poster form 4.3–4.8 and 1.25); the symmetric part of the operator in the
ρ-weighted norm is 40 % of the skew part. In shallow water and in the dycore the gravity-wave coupling
to the momentum (the conserved energy) keeps ρ bounded; a transported quantity without that feedback
(q, θ′) sees the bare operator. Neither form is the skew-symmetric DG central flux
`∫ρ u⋅∇η − ∫{ρ} u⋅n [η]`. The natural fix for q and θ is a DG transport with the kite mass flux,
`⟨ρq̇, η⟩ = Σ_K ∫_K q F⋅∇η − Σ_F ∫_F q̂ F⋅n [η]` with the upwind (or central) trace `q̂`: for q = 1 it is
`B F` (consistency), and with upwinding it is stable in the ρ-weighted norm.

**Source of the positive eigenvalues** (dense eigenvalues of the height transport with the
prescribed TC2 velocity u_h; RT1 × Q2, `eig_source.jl`, `eig_tracer_dg.jl` in the scratchpad):

| max Re λ / (u0/Δx) | hexagon 2 | cubed sphere 4 | triangles 2 |
|---|---|---|---|
| projected flux `B Π_U(ρu)`, degree 4 | 0.700 | 0.278 | 1.473 |
| projected flux, degree 10 | 0.695 | 0.277 | 1.409 |
| pointwise flux (interior trace), degree 4 and 10 | 4.757 | 1.245 | 2.428 |
| DG central `∫ρu⋅∇η − ∫{ρ}u⋅n[η]` | 0.001 | 0.000 | 0.002 |
| DG upwind | 0.000 | 0.000 | 0.000 |

- **Not under-integration:** degree 10 changes nothing (the pointwise form is exact with degree 4).
- **Pointwise:** the interior trace ρ_K on every primal face decouples the primal cells. On inflow
  faces it is the downwind value, so ½ d/dt ∫ρ² gets +½∫|u⋅n|ρ² there (anti-upwind). The
  symmetric part is 75–93 % of the skew part.
- **Projected:** ⟨ρ̇, η⟩ = ∫ ρ u⋅g_η with g_η = M_U⁻¹Bᵀη, the discrete gradient of η in U,
  including the primal-face jumps of η. These are spread over the dual-cell mass blocks.
  - So ⟨ρ̇,η⟩ + ⟨η̇,ρ⟩ = ∫u⋅(ρ g_η + η g_ρ) ≠ ∫u⋅∇(ρη) = 0.
  - In effect, the trace of ρ in F⋅n is a weighting set by the mass blocks, not the mean.
  - This is a projection (consistency) defect at the grid scale: max Re λ ∝ 1/Δx, independent
    of the quadrature, and worst on the irregular kites of triangles.
- **DG central** is skew up to the discrete divergence of u_h (symmetric part 2–4e-4 of the skew part).
- **Tracer with the frozen kite flux F = Π_U(ρu):** `x = ρq ↦ M_h⁻¹[Σ∫q F⋅∇η − Σ∫q̂ F⋅n[η]]`, q = x/ρ.
  - With q = 1 it reproduces B F to 1e-12 on all three meshes.
  - max Re λ/(u0/Δx) is 0.001 / 0.000 / 0.002 for the central trace and 0 for upwind.
  - The current `B Π_U(qF)` gives 0.70 / 0.28 / 1.47.
  - So the DG tracer form removes the instability and keeps ρq = ρ for q = 1.

**DG transport in the model** (`horizontal_theta=:dg` central, `:dg_upwind` upwind trace;
2026-11-04):

- **ρθ and ρq:** `Σ_K ∫ θ F⋅∇η − Σ_F ∫ θ̂ F⋅n [η]` with the kite mass flux F (`dg_divergence!`).
- **Pressure gradient:** its adjoint (`dg_gradient!`), so πᵀ(ρθ̇) = Fᵀ(u̇) exactly and the
  energy exchange stays exact.
- **Horizontal advection of w:** the same defect. η = ∂u/∂z − ∇w used the projected gradient
  M_u⁻¹Bᵀw, whose eigenvalues are identical to those of the ρ transport (0.700 / 0.278 / 1.473).
  - It is now DG in advective form, `−Σ∫(u⋅∇w)ψ + Σ_F∫(u⋅n)[w]{ψ}`, with the upwind penalty for
    `:dg_upwind` (`w_advection!`).
  - Its adjoint goes into the momentum equation (`w_advection_flux!`; the adjoint holds to 1e-17).
  - The eigenvalues drop to 0.0005 / 0.0000 / 0.0015.
- ρ keeps `B F`.
- **Tests:** adjoint, consistency (q = 1 gives B F), conservation and ρq = ρ, on all three meshes.

| one layer, TC2, random θ′ = 1e-8 | hexagon 3 | cubed sphere 12 | triangles 3 |
|---|---|---|---|
| `B Π_U(θF)` | NaN in a day | 3e-3 | NaN in a day |
| `:dg`, `:dg_upwind` | 6e-8 / 2e-8, constant | the same | the same |

The thermal shallow-water instability is gone on all meshes. In 3D it is not:

| thermal wind, max \|w\| at day 0.5 / 1.5 / 3 | hexagon 3 | cubed sphere 12 | triangles 3 |
|---|---|---|---|
| `:dg_upwind` | 5.0e-5 / 1.7e-4 / 2.9e-2 | 3.6e-5 / 3.0e-3 / 0.20 | 1.4e-4 / 0.14 / NaN (day 2.5) |
| `:dg` | 7.7e-5 / 6.9e-4 / 6.7e-2 | | |

With and without the DG advection of w these agree to three digits. The old upwind θ on
hexagon 3 gave 3.9e-4 / – / 0.30, and TRiSK and ICON stay below 3e-3.

**Third source: the momentum and continuity transport in supercritical flow.** Ablations on
triangles 3 (max |w| at day 1, 3.5e-3 for the baseline):

- **No vertical advection:** 3.5e-3.
- **Centred vertical θ:** 3.5e-3.
- **dt 75 s:** the same through day 0.5.
- **Barotropic isothermal state in solid-body rotation (no shear):** 9.5e-3. With 5 layers of
  1 km it gives 1.2e-2 at day 1; **one layer stays at rest**.

So the growth needs neither shear nor vertical advection, but more than one layer. The high vertical
modes have internal wave speeds NH/(mπ) of a few m/s, below the wind. One layer with a uniform flow
(1, 0.3) on the periodic unit square, no rotation, `:dg_upwind`, random 1e-8:

| 16 × 16, max \|δρ\|/h0 up to t = 8 | c = 3 | c = 1.5 | c = 1.0 | c = 0.5 |
|---|---|---|---|---|
| quadrilaterals | 1e-7 | 6e-7 | 4e-6 | NaN by t = 0.3 |
| triangles | 3e-7 | 2e-6 | NaN by t = 0.8 | NaN by t = 0.2 |

Eigenvalues of the linearized one-layer tendency on 6 × 6 (`eig_plane.jl`):

| max Re λ/(\|u\|/Δx) | full, c = 0.5 | full, c = 3 | momentum block (ρ frozen) | continuity block (u frozen) |
|---|---|---|---|---|
| quadrilaterals | 3.49 | 0.000 | 0.77 | 0.77 |
| triangles | 6.08 | 0.007 | 5.48 | 5.49 |

- **Both transports are unstable on their own:** the vector-invariant momentum advection (PV flux
  and ∇K) and the continuity `B Π_U(ρu)`.
- **Subcritical flow (c > |u|):** the fast gravity-wave coupling and the conserved energy keep the
  coupled system neutral. The 2D tests and the 3D one-layer tests are all subcritical.
- **Supercritical flow:** the perturbation can draw energy from the mean flow, and the coupled rate
  3.49 on quadrilaterals matches the observed growth (×260 per 0.1).
- **In 3D:** every column has supercritical vertical modes, so this is the remaining 3D
  instability.

**Baroclinic channel** (`examples/baroclinic_channel.jl`, `baroclinic_channel`, Ullrich, Reed and
Jablonowski 2015; mirrored, doubly periodic box 40000 × 12000 km on the f-plane, 30 layers to 30 km;
`:dg_upwind`, dt 300 s, mass to 1e-15):

| 40 × 12 cells (1000 km) | min p_s day 4 / 8 / 10 / 12 | max \|w\| days 0.5–4 | s/step |
|---|---|---|---|
| quadrilaterals (12000 columns) | 999.6 / 997.3 / 990.9 / 982.4 hPa | 3.5e-3 → 1e-3 | 1.9 |
| triangles (18240 columns) | 973.8 at day 4.5, then 972–973 to day 10.5 | 1.7e-2 → 1.9 m/s by day 2 | 2.9 |

- **Quadrilaterals:** the kites are affine and the growth rates small, so the wave develops
  smoothly, though slowly at 1000 km. It is not yet compared with a reference solution.
- **Triangles:** the run is dominated by the grid-scale mode from day 1.

## DG momentum advection (2026-11-05, solves the supercritical instability)

`horizontal_advection=:dg` (`KiteAdvection.jl`; requires `horizontal_theta` `:dg` or `:dg_upwind`)
replaces the vector-invariant momentum and the projected continuity by DG transport with the
pointwise velocity u:

- **Momentum, advective form** with the ρ-weighted velocity mass (Cholesky per dual-cell block):
  `∫ρ u̇·v = −N(u; v) − ∫ρ f (r̂×u)·v − c_p (Σ∫ρθ v·∇π − Σ (ρθ)^ (v·n)[π]) − (vertical ∂u/∂z)`,
  `N(u; v) = Σ_K ∫ρ v·(u·∇)u − Σ_F ρ̂ {u·n} [u]·{v} + α Σ_F ρ̂ |u·n|/2 [u]·[v]` over all four
  faces of every kite (primal and dual edges), ρ̂ the mean. α = 1 (upwind) with `:dg_upwind`.
- **ρ, ρθ, ρq:** `dg_divergence!` with u, as ρθ before. Mass stays exact per kite.
- **Volume term:** `(u·∇)u dA = Σ_k û_k ∂_k u dξ`, with `∂_k (P û) = ∂_k P û + P ∂_k û` and `∂_k P`
  of the contravariant Piola factor by a Richardson difference of the cell map.

The central part is skew in the energy norm diag(ρ M_u, g M_h) up to the projection of K. Without
the upwind term the eigenvalues of the one-layer tendency, uniform flow, c = 0.5, 6 × 6
(`eig_model.jl`), are:

| max Re λ/(\|u\|/Δx), c = 0.5 | vector invariant | `:dg`, α = 0 | `:dg`, α = 1 |
|---|---|---|---|
| quadrilaterals | 3.49 | 0.0000 | 0 |
| triangles | 6.08 | 0.014 (non-affine kites) | 0 |

The DG advection has the CFL limit of DG with RK3, about 0.1 per kite (0.13 failed in the one-layer
plane test). The 3D runs here are at about 0.02. It costs about 20 % more per step than the vector-invariant form.

**Thermal wind**, `:dg_upwind` with `:dg` advection, 30 layers, dt 150 s (before: 2.9e-2 / 0.20 / NaN at day 3):

| max \|w\| at day 0.5 / 1 / 3 | hexagon 3 | cubed sphere 12 | triangles 3 |
|---|---|---|---|
| `:dg` advection | 4.5e-5 / 6.6e-5 / 1.3e-4 | 2.6e-5 / 4.0e-5 / 1.1e-4 | 4.9e-5 / 7.1e-5 / 1.5e-4 |

max |θ′| stays at 2e-2 K and max |Δp_s| at 5e-2 hPa on all three meshes, about 10× below TRiSK
on hexagon 4.

**Baroclinic channel** (`:dg` advection; k = 10 is the amplitude of zonal wavenumber 10 of p_s in the
channel, linear QG growth rate 6.59e-6 1/s):

| | min p_s day 4 / 8 / 10 / 12 | k = 10 growth days 3–12 | max \|w\| day 12 | s/step |
|---|---|---|---|---|
| 40 × 12 quadrilaterals, dt 300 s | 999.6 / 997.7 / 993.2 / 979.8 hPa | – | 2.6e-2 | 2.1 |
| 40 × 12 triangles, dt 300 s | 999.6 / 997.8 / 993.5 / 980.7 hPa | – | 3.9e-2 | 3.3 |
| 60 × 18 quadrilaterals, dt 200 s | 999.7 / 997.7 / 993.0 / 978.7 hPa | 5.9–6.6e-6 1/s (6.58e-6 at day 11) | 2.7e-2 | 4.5 |
| 60 × 18 triangles, dt 200 s | 999.7 / 997.8 / … | 5.8–6.0e-6 1/s (to day 8) | | 7.1 |

The triangle run that the vector-invariant form ruined from day 1 now follows the quadrilateral
one to 1 hPa. The growth of the k = 10 wave matches the QG rate within 10 % from day 3 on, and
within 0.2 % at its maximum. Ullrich, Reed and Jablonowski (2015) see the wave after about 6 days
and the breaking near day 13.5 at 100 km; at 667 km it is a little slower.

**Spherical baroclinic wave** (Ullrich et al. 2014, `baroclinic_wave`), RT1 × Q2, `:dg_upwind`, `:dg`
advection, 30 layers, dt 150 s, minimum p_s (hPa):

| day | 6 | 7 | 8 | 9 | 10 | max \|w\| day 10 | s/step |
|---|---|---|---|---|---|---|---|
| hexagon 3 (23682 columns) | 993.5 | 984.3 | 968.3 | 941.6 | 925.8 | 0.11 | 3.9 |
| cubed sphere 12 (21600) | 993.4 | 984.1 | 966.9 | 942.7 | 926.3 | 0.13 | 3.7 |
| triangles 3 (24320) | 993.4 | 984.7 | 967.3 | 936.0 | 925.4 | 0.11 | 4.3 |
| TRiSK, hexagon 6 (40962 cells) | 991.7 | 982.3 | 963.1 | 940.4 | 925.1 | | |

The three meshes agree to 1 hPa at day 10 and to TRiSK at a similar resolution. Mass is
conserved to 1e-15. The old Q1 × dual Radau 1 run on hexagon 3 gave 976.6 hPa with max |w| ≈ 1 m/s.

**Open problems of the kite model:**

1. The supercritical instability is solved by `:dg` advection (above). The constructor defaults
   (`:central` θ, `:vector_invariant`) are kept for the tests; the examples use `:dg_upwind` with
   `:dg`. Not yet energy conserving: the kinetic energy of the advective form differs from
   the K of the pressure term by the projection of K, and the upwind term supplies the dissipation.
2. The DG and upwind terms are CPU loops; GPU kernels and distribution are still to do.
3. Coarse columns of first order give a far too weak baroclinic wave; resolution and order must
   go up together, which needs the GPU.

Record of the search:

The full step (horizontal explicit, columns implicit) is unstable on the hexagon and triangle meshes:
small noise on a stratified rest state grows by about ×6 every 2 h (hexagon 3, 10 layers), the
unperturbed baroclinic wave blows up within 16 h. What is known:

- **Semi-discrete:** the same growth at dt = 15, 50, 75, 150 s; the explicit acoustic limit is
  between 250 and 300 s on hexagon 3.
- **Not the vertical solver alone:** `vertical_step!` with the same noise stays at |w| ~ 1e-13;
  `:centred`, `:upwind` θ and β = 0.5, 0.65, 1 all grow.
- **Not the horizontal terms alone:** one layer (no vertical motion) decays, two layers stay bounded.
- **Not rotation:** the same growth with Ω = 0.
- **Not the vertical mass of w:** the consistent CG1 mass (tridiagonal, now the default) and the
  lumped one grow alike.
- **The 2D kite shallow-water model** runs on planes of all three grid types (quadrilaterals,
  triangles, hexagons), and the one-layer 3D model is stable, so the horizontal pair alone is fine.
- **Mesh:** grows on hexagons (plain and spring-optimized) and faster on triangles, largest near
  the pentagons; bounded on the cubed sphere, whose kites are affine (squares). The kites of
  hexagons and triangles are not affine.
- **Order of the height space:** `KiteRadau(0)` (one column per primal cell, as TRiSK) is stable,
  `KiteRadau(1)` (13 columns per hexagon) grows slowly (×2 per 2 h), `KiteRadau(2)` (37 columns) fast.
- **Structure:** 99 % of the mode is inside the primal cells (zero cell mean), largest in the
  lowest layers.
- **Energy:** the total energy stays constant (it decreases slightly by the off-centring) while
  the kinetic energy grows by 10³: the mode releases potential and internal energy, so the
  conserved energy is not a sufficient stability bound for these sub-cell modes. `TRiSKNonhydrostatic`
  on the same vertical grid and noise stays bounded without any diffusion.

Further tests (same noise test, hexagon 3, 10 layers, unless noted):

- **Resolution:** hexagon 0 and 1 (also with a/8 and a/16, so with cells of the size of hexagons
  3–4) stay bounded; hexagon 2 grows ×1.3 per hour, hexagon 3 ×2.4, hexagon 4 ×6. The rate roughly
  doubles per refinement (∝ 1/Δx), and the mode needs the irregular hexagons of the finer meshes,
  not only pentagons (hexagon 0 has only pentagons). A dense eigen-analysis on a small mesh
  therefore finds nothing.
- **Frequency:** low. At the strongest column w changes sign once in 2 h (e-folding 1.3 h);
  the horizontal kinetic energy is 200 times the vertical one.
- **Stratification restrains it:** isentropic (N = 0) ×2.5 per 30 min, the baroclinic-wave profile
  ×170 in 6 h, isothermal (T = 250 K) ×2.6 in 6 h. One layer stays stable in all cases.
- **Linear in θ:** θ replaced by its layer mean in Π_U(θ·) of the pressure gradient and the ρθ flux
  gives the same growth, as the linearization at rest says.
- **Not the Lorenz computational mode:** θ′ of the mode is smooth in the vertical (neighbouring
  layers correlate +0.25, interface values as large as for noise). Structure at the strongest
  column: warm lowest layer (θ′ > 0, ρ′ < 0) under downward motion in levels 2–4, the opposite at the
  neighbouring nodes of the cell, so the motion is driven by the sub-cell pressure gradient.
- **Not the lumped height mass:** the exact mass of the height space (37 × 37 per primal cell) in
  the horizontal divergence grows slightly faster. On these meshes the Radau lumping equals row-sum
  lumping (diag(exact)/lumped 0.88–0.98).
- **No ghosts:** owned and local dofs agree in serial.

Next: Arnoldi on the linearized step map of hexagon 3 (matrix-free, finite differences of the step,
about 50 products) for the unstable eigenvalues and eigenvectors, then the energy conversion terms of
the leading mode.
