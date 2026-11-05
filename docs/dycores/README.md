# Nonhydrostatic dycores

Horizontal discretizations of the dry compressible Euler equations on the same vertical grid, the
same implicit column solver and the same test cases (`src/Nonhydrostatic/`):

| dycore | grid | time stepping | status | GPU (A100) |
|---|---|---|---|---|
| [ICON](ICON.md) (`ICONNonhydrostatic`) | triangular C grid | predictor–corrector of ICON | baroclinic wave to R2B6, p_s 917.3 hPa at day 10 | 18.8 ms per step on R2B6 |
| [MPAS-Gassmann](MPAS-Gassmann.md) (`MPASGassmannNonhydrostatic`, formerly `TRiSKNonhydrostatic`), ICON-IAP form, MPAS operators as options | hexagonal C grid | Wicker–Skamarock RK3, HEVI | baroclinic wave to hexagon 7, p_s 931.4 hPa at day 10 | 24.5 ms per step on hexagon 7 |
| MPAS-Orig (planned) | hexagonal C grid | split-explicit RK3 with acoustic substeps (MPAS-A) | – | – |
| [Kite](Kite.md) (`KiteNonhydrostatic`) | kite finite elements | Wicker–Skamarock RK3, HEVI | baroclinic wave on hexagons, cubed sphere and triangles, p_s 925.4–926.3 hPa at day 10 (CPU, serial) | not yet |

Source layout:

```
src/Nonhydrostatic/
  Common/    Constants (constants, kernel launcher with blocks of 32 points × 8 levels), VerticalGrid,
             ColumnSolver (θ at the interfaces, vertical mass of w, tridiagonal column solve),
             CGridOperators (shared by ICON and MPAS), TimeStepping (Wicker–Skamarock RK3), TestCases
  ICON/      ICONNonhydrostatic, VelocityAdvection, SolveNH, Diffusion, Stepping
  MPAS/      Operators (TRiSK operators on Voronoi meshes)
    Gassmann/  MPASGassmannNonhydrostatic, Tendencies, Diffusion, Stepping
  Kite/      KiteNonhydrostatic, KiteAdvection
```

Test cases: `baroclinic_wave`, `thermal_wind`, `baroclinic_channel` (`Common/TestCases.jl`).

Planned: ICON closer to the standard code (prognostic Exner pressure and `θ_v` with ICON's own
vertical solve, reference state, substeps, `divdamp_order = 24`, `mo_nh_diffusion`, with the
predictor–corrector as default and the RK3 as an option), the ICON-IAP time stepper for
MPAS-Gassmann, and MPAS-Orig.
