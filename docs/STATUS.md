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
- Finite volumes: `TRiSKShallowWater(dm; comm, backend)` (C grid on Voronoi meshes: hexagon mesh, MPAS,
  dual of ICON; TRiSK weights of Thuburn et al. 2009 as `i_cori_method = 4` in ICON-IAP,
  energy-conserving PV flux, ICON kinetic energy). KernelAbstractions kernels without atomics (gathers
  over padded neighbour tables); MPI with `TRiSKDistribution` (equal-area cells, two vertex rings of
  ghost cells, one halo exchange of `[u; h]` per tendency): 1, 3, 4 ranks bitwise identical, on CUDA
  (Levante, 1 and 4 A100) too. Time per RK4 step (TC2, hexagon 7/8/9 = 164k/655k/2.6M cells):
  1 A100 2.3/4.6/16.7 ms, 4 A100 2.4/2.8/5.8 ms (one merged halo message per neighbour; kernels are
  not synchronized individually on GPUs). TC2: mass to round-off, semi-discrete dE/dt = 0, L2 height error
  ≈ 1.5th order; the max error saturates (TRiSK is inconsistent on unoptimized icosahedral grids).
  `pv_flux = :gassmann`: reconstruction of Gassmann (2013) with the PV usage of Gassmann (2018), ICON-IAP
  `i_cori_method = 1` (12 edges, rhombus PV; the metric `n_k ⋅ (r̂ × n_e)` with `r̂` at the midpoint of both
  edges makes it exactly antisymmetric, so energy is conserved); consistent where TRiSK is not (tangential
  max error 0.015 → 0.0016 for hexagon 3 → 6, TRiSK 0.12), TC2 L2 height errors 2–4× smaller.
  Without dissipation its PV usage lets grid-scale vorticity grow in the Galewsky jet from day 3 (also
  for the unperturbed jet; mixing the pairings does not help). `pv_flux = :gassmann_mean` uses the mean
  `(q_e + q_e')/2` (needs three ghost rings) and stays smooth. `examples/trisk_galewsky.jl` (hexagon 7,
  6 days, 4 ranks): TRiSK and `:gassmann_mean` give the expected vortex roll-up (|ζ| ≤ 1.6e-4 1/s,
  potential enstrophy +0.4 %), `:gassmann` |ζ| 5e-4 and +51 % enstrophy.
  Levante A100, hexagon 8 (655k cells): TC2 per RK4 step `:gassmann` 11.6 ms (1 GPU) / 4.4 ms (4 GPUs),
  `:gassmann_mean` 8.3 / 3.5 ms, 1 and 4 GPUs identical errors; Galewsky 6 days on 4 GPUs 28 s (TRiSK),
  58 s (`:gassmann`), 43 s (`:gassmann_mean`).
  Next: APVM/upwind PV or hyperdiffusion, distributed mesh construction (the global grid and TRiSK tables
  are still built on every rank); Voronoi generation itself is not a priority (MPAS grids are used).
- Parallel: KernelAbstractions kernels with atomics, MPI distribution with equal-area partition,
  one vertex ring of ghost cells, local numbering, halo exchange of u, h, ζ, F.
- Classical models over MPI and KernelAbstractions: `LinearShallowWater`/`NonlinearShallowWater(...; comm, backend)` on a
  `CellDistribution` (owned cells + vertex ring of ghosts), owned rows of the global operators, nonlinear
  terms over the local cells, `distributed_mass_solver`: local blocks + halo for DG, `GatheredMassSolver`
  (Allgatherv + replicated Cholesky, exact) for RT/CG, `solver=:cudss` for GPUs; 1 and 3 ranks agree with
  serial to round-off (`test/mpi_fem.jl`). Run-time data on the backend: operator rows as `RowMatrix`,
  nonlinear terms, energy and enstrophy as kernels over dense per-cell bases (`CellTables`, one cell shape
  per mesh), DG solves as `BlockSolver`; coupled mass matrices with cuDSS or through the host
  (`HostSolver`, `GatheredMassSolver`).
- GPU: `NonlinearKiteShallowWater(...; backend)` keeps all run-time arrays on the backend
  (block solves `BlockSolver`, `RowMatrix` products, halo pack/unpack as kernels; plain loops on the CPU).
- Output: ADIOS2 in the VTX data model (`vtx_writer`, `write_vtx!`, extension `dmplexADIOS2Ext`),
  one block per rank, device arrays by pointer; checked in ParaView 6.1 (serial and 2 ranks, CPU).
- cuDSS (extension `dmplexCUDSSExt`): `cudss_solver` (one GPU) and `distributed_cudss_solver` (MGMN,
  rows in the contiguous numbering of `DofDistribution(V, cells)`) for the classical element pairs.
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
5. **Distributed mesh construction** (the global mesh is still built on every rank).
6. **Levante GPU runs** (2026-10-28, 1 node, 4 × A100, `docs/levante.md`): working on CUDA —
   cuDSS single GPU and MGMN for RT1/DG1/CG2 (errors ≤ 4e-15), classical TC2 with `--gpu` on 1 and 4
   ranks (identical diagnostics, mass 1e-15, energy 3e-14), kite Galewsky `hexagon 5` on 4 GPUs
   (7.7 ms per step), CUDA-aware OpenMPI. Open: ADIOS2 from device pointers (needs an ADIOS2 built with
   CUDA, none on Levante yet); scaling — at `cubed 32` 4 GPUs are slower than 1 (TC2 52 s vs 18 s; the
   distributed cuDSS solve takes 4 ms against 0.5 ms on one GPU), measure at larger meshes.
7. **Classical models on GPUs**: all run-time parts are KernelAbstractions kernels and run on CUDA.
   `GatheredMassSolver` replicates the factorization (not scalable) and is the CPU reference; for
   many GPUs compare `solver=:cudss` with replicated single-GPU cuDSS factorizations.

## Environment notes

- Julia 1.12; `julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'` (the test suite runs an
  `mpiexec` job with 1 and 3 ranks, a few minutes).
- GPU, ADIOS2 and cuDSS runs use the environment `examples/gpu` (weak dependencies); see `docs/levante.md`.
- NCDatasets must be ≥ 0.14 (uses NetCDF_jll). An old 0.10 version tries to install NetCDF via Conda.
- MPI.jl uses `MPICH_jll`; start parallel runs with `mpiexec` from `MPI.mpiexec()` or the system `mpiexec`
  if it is MPICH-compatible.
- Grid files are expected in `grid/` (not in git): ICON `icon_grid_*.nc`, MPAS `x1.*.grid.nc`, `QU240.nc`.
- Notes: `cd docs/notes && latexmk -pdf kite_shallow_water.tex`.
- The personal website is the separate repository `OsKnoth/OsKnoth.github.io`.
