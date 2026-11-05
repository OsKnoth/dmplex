# Nonhydrostatic dycores

Three horizontal discretizations of the dry compressible Euler equations on the same vertical
grid, the same implicit column solver and the same test cases (`src/Nonhydrostatic/`):

| dycore | grid | time stepping | status | GPU (A100) |
|---|---|---|---|---|
| [ICON](ICON.md) (`ICONNonhydrostatic`) | triangular C grid | predictor–corrector of ICON | baroclinic wave to R2B6, p_s 917.3 hPa at day 10 | 18.8 ms per step on R2B6 |
| [TRiSK](TRiSK.md) (`TRiSKNonhydrostatic`), ICON-IAP and MPAS forms | hexagonal C grid | Wicker–Skamarock RK3, HEVI | baroclinic wave to hexagon 7, p_s 931.4 hPa at day 10 | 24.5 ms per step on hexagon 7 |
| [Kite](Kite.md) (`KiteNonhydrostatic`) | kite finite elements | Wicker–Skamarock RK3, HEVI | baroclinic wave on hexagons, cubed sphere and triangles, p_s 925.4–926.3 hPa at day 10 (CPU, serial) | not yet |

`Columns.jl` holds the shared parts: physical constants, the kernel launcher with blocks of
32 points × 8 levels, θ at the interfaces, the vertical mass of w and the tridiagonal column solve.
Test cases: `baroclinic_wave`, `thermal_wind` (`BaroclinicWave.jl`).
