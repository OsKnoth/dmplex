# Status and next steps

Working notes for continuing development (on another machine or in a new session).

## State (October 2026)

- Meshes: icosahedral, cubed sphere, hexagonal (exact Voronoi), HEALPix, periodic planes; ICON/MPAS readers;
  cell centers from the orthogonal dual; kite grids.
- Finite elements of any order (Lagrange, DG, RT, BDM on quads, Nédélec); kite elements `KiteRadau`,
  `KiteBrokenRT`, `KiteBrokenBDM`, `KiteDualRadau` (Wess et al.); lumped and block-diagonal mass solvers.
- Shallow water: mixed RT×DG(×CG) linear and nonlinear; linear and nonlinear kite models
  (`velocity = :rt | :bdm | :dual_radau`, conservative variant default, poster form via
  `flux=:pointwise, vorticity=:absolute`).
- Parallel: KernelAbstractions kernels with atomics, MPI distribution with equal-area partition,
  one vertex ring of ghost cells, local numbering, halo exchange of u, h, ζ, F.
- Documentation: `README.md`, `docs/notes/kite_shallow_water.tex` (+ PDF), wiki pages (see below).

## Open items

1. **Wiki**: pages are written but not yet pushed (the GitHub wiki must be enabled and a first page
   created in the browser; then push the pages to `https://github.com/OsKnoth/dmplex.wiki.git`).
   If the local folder `dmplex.wiki` is not available, the pages are reproduced from the notes.
2. **Scaling**: measure 1–8 ranks on an idle machine (`mpiexec -n N julia --project examples/galewsky.jl hexagon 4 0.25`);
   the last measurement ran under load and is not meaningful.
3. **Galewsky at poster resolution** (`hexagon 5` ≈ 61k kites, or `x1.40962.grid.nc` ≈ 245k kites) on 6 ranks.
4. **Damping**: APVM in the potential-vorticity term (energy conserving), upwind height flux on primal
   edges, biharmonic hyperviscosity for comparison.
5. **Distributed mesh construction** (the global mesh is still built on every rank), GPU arrays.

## Environment notes

- Julia 1.12; `julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'` (the test suite runs an
  `mpiexec` job with 1 and 3 ranks, a few minutes).
- NCDatasets must be ≥ 0.14 (uses NetCDF_jll). An old 0.10 version tries to install NetCDF via Conda.
- MPI.jl uses `MPICH_jll`; start parallel runs with `mpiexec` from `MPI.mpiexec()` or the system `mpiexec`
  if it is MPICH-compatible.
- Grid files are expected in `grid/` (not in git): ICON `icon_grid_*.nc`, MPAS `x1.*.grid.nc`, `QU240.nc`.
- Notes: `cd docs/notes && latexmk -pdf kite_shallow_water.tex`.
- The personal website is the separate repository `OsKnoth/OsKnoth.github.io`.
