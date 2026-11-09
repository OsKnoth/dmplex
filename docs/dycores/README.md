# Nonhydrostatic dycores

Horizontal discretizations of the dry compressible Euler equations on the same vertical grid, the
same implicit column solver and the same test cases (`src/Nonhydrostatic/`):

| dycore | grid | time stepping | status | GPU (A100) |
|---|---|---|---|---|
| [ICON](ICON.md) (`ICONNonhydrostatic`), ICON's numerics, namelist defaults and constants | triangular C grid | predictor–corrector of ICON | as Fortran ICON: DCMIP wave R2B5, p_s 936.84 hPa at day 10 (ICON 936.86) | 10.0 ms per step on R2B5 with 64 levels (Fortran ICON 10.8 ms) |
| [MPAS-Gassmann](MPAS-Gassmann.md) (`MPASGassmannNonhydrostatic`, formerly `TRiSKNonhydrostatic`), ICON-IAP form, MPAS operators as options | hexagonal C grid | ICON-IAP predictor–corrector, five-band vertical solve (RK3 as option) | baroclinic wave to hexagon 7, p_s 931.4 hPa at day 10 | 24.5 ms per step on hexagon 7 |
| MPAS-Orig (planned) | hexagonal C grid | split-explicit RK3 with acoustic substeps (MPAS-A) | – | – |
| [Kite](Kite.md) (`KiteNonhydrostatic`) | kite finite elements | Wicker–Skamarock RK3, HEVI | baroclinic wave on hexagons, cubed sphere and triangles, p_s 925.4–926.3 hPa at day 10 (CPU, serial) | not yet |

Source layout:

```
src/Nonhydrostatic/
  Common/    Constants (constants, kernel launcher with blocks of 32 points × 8 levels), VerticalGrid,
             ColumnSolver (θ at the interfaces, vertical mass of w, tridiagonal column solve),
             CGridOperators (shared by ICON and MPAS), TimeStepping (Wicker–Skamarock RK3), TestCases
  ICON/      ICONNonhydrostatic (with flux_average), VelocityAdvection, SolveNH, Diffusion, Stepping,
             ICONState (read_icon_state: the state of a Fortran ICON run)
  MPAS/      Operators (TRiSK operators on Voronoi meshes)
    Gassmann/  MPASGassmannNonhydrostatic, Tendencies, Diffusion, Stepping
  Kite/      KiteNonhydrostatic, KiteAdvection
```

Accuracy, effective resolution and cost of Kite against ICON: [Kite-vs-ICON](Kite-vs-ICON.md).

Test cases: `baroclinic_wave`, `jablonowski_williamson`, `thermal_wind`, `baroclinic_channel`
(`Common/TestCases.jl`).

Orography: ICON and MPAS-Gassmann have terrain-following levels (`surface_height`,
`terrain_following`); Kite is flat (plan in [OROGRAPHY_PLAN](../OROGRAPHY_PLAN.md)).

Planned: ICON's `itype_vn_diffu = 2`, orography for Kite, and MPAS-Orig.
