# Kite dycore: compatible finite elements (`KiteNonhydrostatic`)

Source: [src/Nonhydrostatic/Kite/](../../src/Nonhydrostatic/Kite/). Discretization:
[KITE_FEM.md](../KITE_FEM.md) (short), [KITE3D_DISCRETIZATION.md](../KITE3D_DISCRETIZATION.md)
(complete); status and the record of the tests: [KITE3D_PLAN.md](../KITE3D_PLAN.md).

## Discretization

A direct product of the 2D kite spaces of `NonlinearKiteShallowWater` with the layers: ρ and ρθ in
`KiteRadau(2)` ⊗ DG0 (37 columns per hexagon), w in `KiteRadau(2)` ⊗ CG1, u_h in `KiteBrokenRT(1)`
⊗ DG0 (or the dual Radau pair of Wess et al.). Per layer the horizontal terms are those of the
energy-conserving kite shallow-water model with ρ in place of h; the vertical terms use the column
solver of the TRiSK dycore, one column per Radau node. Wicker–Skamarock RK3, HEVI.

## Status

| test | result |
|---|---|
| rest state, mass, energy exchange of the horizontal terms, adjoint vertical coupling | exact to round-off (tested) |
| steady jet, RT1 × Q2 with upwind θ on the primal faces | bounded, at the TRiSK level also on hexagon 4 |
| steady jet, central θ | node-scale θ mode, growing faster with resolution |
| perturbed baroclinic wave, hexagon 3, Q1 × dual Radau 1 | p_s 976.6 hPa at day 10 (too weak at 100 km) |
| thermal wind | **all variants grow** (upwind θ delays it by about a day) |
| thermal shallow water (one layer, TC2), DG θ (`:dg_upwind`) | θ′ bounded on hexagons, cubed sphere, triangles |
| thermal wind, DG θ and DG advection of w | still grows on all three meshes (supercritical vertical modes, see below) |
| baroclinic channel, f-plane, 40 × 12 quadrilaterals | wave develops, min p_s 982.4 hPa at day 12, mass to 1e-15 |
| baroclinic channel, 40 × 12 triangles | grid-scale mode from day 1 |

![max |w| on the steady jet](../images/kite3d_jet_growth.png)

![max |w| and θ′ on the thermal wind](../images/kite3d_thermal_wind.png)

**Open:** the horizontal transport of ρ and u is unstable in supercritical flow (wind faster than
the gravity waves): the eigenvalues of the one-layer tendency in a uniform flow reach 3.5 |u|/Δx on
quadrilaterals and 6 on triangles once c < |u|. In 3D the high vertical modes are always
supercritical. The fix is a DG transport of ρ with its adjoint in the momentum equation. The model
is serial and runs on the CPU only (the DG and upwind terms are CPU loops), so there is **no GPU
timing yet**. Details: [KITE3D_PLAN.md](../KITE3D_PLAN.md).

## Run

```sh
julia --project examples/thermal_wind.jl kite 3 5 150 rt dg_upwind hexagon   # or cubed, triangles
julia --project examples/baroclinic_channel.jl 40 12 300 quadrilateral dg_upwind
```
