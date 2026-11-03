# Examples

The scripts in `examples/` run the models of dmplex.jl on standard test cases. Each one prints its
diagnostics while it runs (mass, energy, errors or extrema), reports the wall time per step, and
writes a VTKHDF file for ParaView (or ADIOS2/VTX with `--adios`). The arguments are listed at the
top of each script.

| Script | Model | Test case | Discretization | Runs on |
|---|---|---|---|---|
| `advection.jl` | finite volumes | cosine bell, solid-body rotation | first-order upwind | CPU |
| `linear_shallow_water.jl` | `LinearShallowWater` | inertia-gravity waves from a bump | RT_k × DG_k | CPU |
| `nonlinear_shallow_water.jl` | `NonlinearShallowWater` | Williamson TC2 | RT_k × DG_k × CG_{k+1} | CPU, MPI, GPU (cuDSS) |
| `trisk_shallow_water.jl` | `TRiSKShallowWater`, `ICONShallowWater` | Williamson TC2 | C-grid finite volumes | CPU, MPI, GPU |
| `trisk_galewsky.jl` | `TRiSKShallowWater`, `ICONShallowWater` | Galewsky jet | C-grid finite volumes | CPU, MPI, GPU |
| `galewsky.jl` | `NonlinearKiteShallowWater` | Galewsky jet | kite elements RT1 × Q2 × CG2 | CPU, MPI, GPU |
| `bickley.jl` | `NonlinearKiteShallowWater` | Bickley jet on a periodic plane | kite elements | CPU |
| `baroclinic_wave.jl` | `TRiSKNonhydrostatic` | baroclinic wave (Ullrich et al. 2014) | 3D TRiSK, HEVI | CPU, GPU |
| `gpu/cudss_mass.jl` | mass matrices | solves against a known solution | RT_k, DG_k, CG_{k+1} | GPU, multi-GPU |

Meshes are given by name: `icosahedron`/`triangle`, `cubed`, `hexagon` (the Voronoi dual of the
icosahedral mesh), `healpix`. Or give the path of an ICON or MPAS grid file, which are expected in
`grid/` and are not in git. The resolution is the number of bisections of the icosahedron (hexagon
`n` has 10·4ⁿ + 2 cells, so hexagon 6 ≈ 120 km and hexagon 7 ≈ 60 km), or the number of cells per
panel edge on the cubed sphere. Shallow-water examples use nondimensional units (a = Ω = g = 1, or
heights in units of 10 km for Galewsky). The baroclinic wave uses SI units.

GPU runs need the environment `examples/gpu` (CUDA, cuDSS, ADIOS2 as weak dependencies); see
[levante.md](levante.md) for the Levante setup and the batch scripts.

## Finite-volume advection: `advection.jl`

```sh
julia --project examples/advection.jl 4 cubed
```

A cosine bell carried by solid-body rotation with first-order upwind fluxes. This is the smallest
example of a `Plex` with its metrics (`compute_mesh_metrics`), the same code on triangles,
quadrilaterals and hexagons. It writes the tracer as VTKHDF.

## Linear shallow water: `linear_shallow_water.jl`

```sh
julia --project examples/linear_shallow_water.jl 2 cubed 8      # RT2 × DG2
```

A geostrophically balanced solid-body rotation with a Gaussian bump on top, which radiates
inertia-gravity waves (the figure in the README). The mixed RT_k × DG_k semi-discretization
conserves mass and energy, and the example prints both. Mass matrices are factorized once.

## Nonlinear shallow water, compatible elements: `nonlinear_shallow_water.jl`

```sh
julia --project examples/nonlinear_shallow_water.jl 1 healpix 8
mpiexec -n 4 julia --project examples/nonlinear_shallow_water.jl 1 cubed 16
mpiexec -n 4 julia --project=examples/gpu examples/nonlinear_shallow_water.jl 1 cubed 64 --gpu
```

Williamson test case 2 (steady geostrophic flow) in vector-invariant form, with velocity in RT_k,
height in DG_k and potential vorticity in CG_{k+1}. It prints the height error against the steady
state. With an ICON grid file the triangles are used; with an MPAS file the Delaunay dual, since
the elements need triangles or quadrilaterals. Over MPI, 1 and 3 ranks agree with the serial run
to round-off. On GPUs the coupled RT and CG mass matrices are solved by cuDSS.

## C-grid finite volumes, TC2: `trisk_shallow_water.jl`

```sh
julia --project examples/trisk_shallow_water.jl hexagon 6
mpiexec -n 4 julia --project examples/trisk_shallow_water.jl hexagon 7
julia --project examples/trisk_shallow_water.jl triangle 6 --icon
```

TC2 with TRiSK on Voronoi meshes (the hexagon mesh, MPAS grids, the dual of ICON grids), or with
`--icon` the triangular C grid of ICON's nonhydrostatic core. TRiSK conserves mass to round-off,
and its semi-discrete energy tendency is zero. Options:

- `--gassmann`, `--gassmann-mean`: the PV flux of Gassmann (2013, 2018).
- `--kinetic=cell|mixed|vertex|α`: the TRiSK kinetic energy (the vertex form is consistent and
  converges to first order).
- `--spring[=β]`, `--scvt`: mesh optimization (spring dynamics, centroidal Voronoi).
- `--centered-k`: the centered kinetic-energy gradient for the ICON model.

**Results.**
- ICON grids, one A100: L2 height error 2.5e-7 / 6.3e-8 / 1.6e-8 for R02B06/07/08 (second order),
  at 1.5 / 2.4 / 7.8 ms per RK4 step.
- TRiSK on one A100: 2.3 / 4.6 / 16.7 ms per RK4 step for hexagon 7/8/9.

## C-grid finite volumes, Galewsky jet: `trisk_galewsky.jl`

```sh
mpiexec -n 4 julia --project examples/trisk_galewsky.jl hexagon 7 6 --gassmann-mean
mpiexec -n 4 julia --project examples/trisk_galewsky.jl icon_grid_0021_R02B06_G.nc 0 6 --icon
julia --project examples/trisk_galewsky.jl x4.163842.grid.nc 0 6 --icon --spring --centered-k
```

The barotropically unstable jet of Galewsky et al. (2004) for 6 days, with the same options as
TC2. It runs inviscid except for ICON's fourth-order divergence damping on triangles, which uses
the local cell area. It writes the relative vorticity.

**Results.**
- TRiSK and `--gassmann-mean` give the expected roll-up (|ζ| ≤ 1.6e-4 1/s).
- `--gassmann` develops grid-scale vorticity from day 3.
- The ICON model is stable on the ICON grids and on the cleaned-up, locally refined MPAS mesh
  x4.163842. On ICON R02B07 with 4 A100 it runs 6 days in 60 s.

## Kite elements, Galewsky jet: `galewsky.jl`

```sh
mpiexec -n 6 julia --project examples/galewsky.jl hexagon 5 6
mpiexec -n 4 julia --project=examples/gpu examples/galewsky.jl hexagon 6 6 --gpu --adios
julia --project docs/galewsky_figure.jl hexagon 4 6      # draws docs/images/galewsky.svg
```

The staggered kite discretization: every primal cell is split into kites, one per corner. The
fields are:
- height `KiteRadau`, broken across primal edges, with a diagonal mass matrix;
- velocity `KiteBrokenRT`, broken H(div) on the dual cells, with a block-diagonal mass matrix;
- vorticity continuous on the kites.

With the projected mass flux and the potential vorticity (the defaults), mass and energy are
conserved exactly, and the example prints both. The figure in the README is hexagon 4 after 6 days.
`--adios` writes the vorticity every output interval with ADIOS2 (VTX).

## Kite elements on a plane: `bickley.jl`

```sh
julia --project examples/bickley.jl quadrilateral 32 40
julia --project docs/bickley_figure.jl quadrilateral 32 40   # draws docs/images/bickley.svg
```

A Bickley jet u = sech²(y − 2π) on the doubly periodic plane [0, 4π)² at Froude number 0.2, without
rotation. A perturbation in x-wavenumber 1 rolls the jet up into a vortex street. The primal mesh
is made of quadrilaterals or triangles (`create_periodic_plane`).

## Nonhydrostatic baroclinic wave: `baroclinic_wave.jl`

```sh
julia --project examples/baroclinic_wave.jl 5 10 300 30
julia --project=examples/gpu examples/baroclinic_wave.jl 7 10 150 30 --gpu
sbatch --account=bb1143 examples/gpu/baroclinic_wave.sbatch     # hexagon 6 and 7 on one A100
```

The baroclinic wave of Ullrich et al. (2014), shallow atmosphere over flat ground in height
coordinates, with `TRiSKNonhydrostatic` on the extruded hexagon mesh (Lorenz staggering: θ with ρ
at the layer centres, w at the interfaces). The vertical grid has 30 equidistant layers up
to 30 km, with Rayleigh damping of w above 22 km. The model follows the ICON-IAP standard:
- PV usage as in ICON-IAP;
- vertical momentum advection by mass fluxes;
- ICON-IAP TRiSK weights;
- Smagorinsky diffusion;
- third-order upwind θ at the interfaces in the ρθ flux (`vertical_theta = :upwind`).

The time step is a Wicker–Skamarock RK3 with a HEVI split. Each stage has an implicit tridiagonal
solve per column for w, ρ, ρθ, linearized around the Exner pressure of the base state.
`hilbert_order` renumbers the mesh along a Hilbert curve first, so that neighbours are close in
memory. The example prints the minimum surface pressure once a day and the time per step, and
writes p_s and the temperature of the lowest layer.

**Minimum surface pressure (hPa)**, `vertical_theta = :upwind`:

| Day | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|
| hexagon 5, dt 300 s | 992.6 | 983.5 | 968.4 | 946.9 | 921.8 |
| hexagon 6, dt 300 s | 991.66 | 982.25 | 963.13 | 940.43 | 925.06 |
| hexagon 7, dt 150 s | 990.2 | 975.8 | 957.5 | 949.5 | 931.4 |

The GPU and CPU runs of hexagon 6 agree, and mass is conserved to 4e-16.

**Time per step on one A100** (30 layers):

| Version | hexagon 6 | hexagon 7 |
|---|---|---|
| original | 19.1 ms | 99.5 ms |
| ρ_e and ρ_e·vn once per stage | 13.6 ms | 59.8 ms |
| plus Hilbert ordering | 12.5 ms | 49.2 ms |
| plus Exner pressure of the base state once per step, edge-first Int32 TRiSK tables, blocks of 32 points × 8 levels | 9.1 ms | 27.3 ms |

In the last version, the two column kernels (`theta_interface`, `vertical_solve`) take 10 of the
27 ms on hexagon 7, and `vn_tendency` 4.4 ms.

## Mass solves with cuDSS: `gpu/cudss_mass.jl`

```sh
srun -n 8 julia --project=examples/gpu examples/gpu/cudss_mass.jl cubed 64 1
```

Solves the RT_k, DG_k and CG_{k+1} mass matrices with cuDSS: one GPU per rank, and distributed over
all ranks in multi-GPU, multi-node mode. Each solve is checked against a known solution (errors
≤ 4e-15 on Levante). The distributed mode needs the cuDSS OpenMPI layer (`CUDSS_COMM_LIB`).
