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

![max |w| on the steady jet](../images/kite3d_jet_growth.png)

![max |w| and θ′ on the thermal wind](../images/kite3d_thermal_wind.png)

**Open:** the node-scale mode on flows with strong cross-edge wind (see the plan): the upwind θ
also in the pressure gradient, the same for ρ. The model is serial and runs on the CPU only (the
upwind θ term is a CPU loop), so there is **no GPU timing yet**.

## Run

```sh
julia --project examples/thermal_wind.jl kite 3 5 150 rt upwind
```
