# Staggered kite finite elements: paper draft

[kite.tex](kite.tex) (compiled: [kite.pdf](kite.pdf)) describes the kite method, from the 2D
shallow-water model (`NonlinearKiteShallowWater`) to the 3D compressible model
(`KiteNonhydrostatic`), with the stability analysis of the horizontal transport and the 3D results.

```sh
latexmk -pdf kite.tex
```

## Figures

The figures in [figures/](figures/) (PDF, for the paper) and their 300 dpi PNG versions in
[../images/](../images/) (for the GitHub pages) are made by the scripts in [scripts/](scripts/). They
need, besides `dmplex`, the packages `Plots` and `NearestNeighbors`. The fields are the finite-element
fields evaluated at 8 × 8 points inside every kite and interpolated to a regular grid, not the nodal
values.

| figure | runs (data in `scripts/data/`) | plot |
|---|---|---|
| `kite_galewsky` | `galewsky_run.jl hexagon 5 6` (80 min) | `fig_2d.jl galewsky ...` |
| `kite_bickley` | `bickley_run.jl quadrilateral 64 40` (30 min) | `fig_2d.jl bickley ...` |
| `kite_spectra` | `eig_spectra.jl` | `plot_series.jl` |
| `kite3d_thermal_wind` | `examples/thermal_wind.jl kite 3 3 150 rt dg_upwind <mesh> dg` and `... vector_invariant` | `plot_series.jl` |
| `kite3d_channel_growth` | `examples/baroclinic_channel.jl 60 14 200 <cells> dg_upwind true dg` | `plot_series.jl` |
| `kite3d_channel_quadrilateral`, `_triangle` | `channel_run.jl 60 14 200 <cells>` (snapshots at days 8–14) | `fig_channel.jl` |
| `kite3d_baroclinic_wave_ps`, `_T` | `baroclinic_wave_run.jl <mesh> <res> 10 150` on hexagon 3, cubed 12, triangles 3 (7 h each) | `fig_bw.jl`, `plot_bw.jl` |
| `kite3d_jet_growth`, `kite3d_jet_w` | steady-jet runs of KITE3D_PLAN.md | `fig_jet.jl` |
