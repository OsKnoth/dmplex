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
| `baroclinic_wave.jl` | `MPASGassmannNonhydrostatic`, `ICONNonhydrostatic` (`--icon`) | baroclinic wave (Ullrich et al. 2014) | 3D MPAS-Gassmann (TRiSK), HEVI; ICON triangles, predictor–corrector | CPU, GPU |
| `thermal_wind.jl` | `MPASGassmannNonhydrostatic`, `ICONNonhydrostatic`, `KiteNonhydrostatic` | steady thermal wind (`thermal_wind`) | 3D MPAS-Gassmann, 3D kite elements, HEVI | CPU |
| `baroclinic_channel.jl` | `KiteNonhydrostatic` | baroclinic wave in an f-plane channel (Ullrich, Reed and Jablonowski 2015) | 3D kite elements on a periodic plane, HEVI | CPU |
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
coordinates, with `MPASGassmannNonhydrostatic` on the extruded hexagon mesh (Lorenz staggering: θ with ρ
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

With `--icon` the same script runs `ICONNonhydrostatic` on the triangle mesh (ICON's
predictor–corrector, Miura fluxes, divergence damping, fourth-order diffusion scaled with Δx³), see
[dycores/ICON.md](dycores/ICON.md):

| Day | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|
| ICON R2B5 (triangles 6), dt 75 s | 992.63 | 981.41 | 961.49 | 939.03 | 918.16 |
| ICON R2B6 (triangles 7), dt 37.5 s | 991.88 | 979.15 | 958.49 | 937.32 | 917.30 |

`examples/gpu/dycores.sbatch` runs ICON and TRiSK in one job; on one A100 ICON takes 5.2 ms per step
on R2B5 and 18.8 ms on R2B6. The three dycores are compared in [dycores/](dycores/README.md).

**Time per step on one A100** (30 layers):

| Version | hexagon 6 | hexagon 7 |
|---|---|---|
| original | 19.1 ms | 99.5 ms |
| ρ_e and ρ_e·vn once per stage | 13.6 ms | 59.8 ms |
| plus Hilbert ordering | 12.5 ms | 49.2 ms |
| plus Exner pressure of the base state once per step, edge-first Int32 TRiSK tables, blocks of 32 points × 8 levels | 9.1 ms | 27.3 ms |
| plus the pentadiagonal θ solve with rows built on the fly | 7.7 ms | 25.1 ms |

In the last version, on hexagon 7, `vertical_solve` takes 5.1 ms, `vn_tendency` 4.5 ms and
`theta_interface` 2.3 ms.

## Nonhydrostatic thermal wind: `thermal_wind.jl`

```sh
julia --project examples/thermal_wind.jl trisk 4 5 300
julia --project examples/thermal_wind.jl kite 3 5 150 rt dg_upwind hexagon dg
```

For the kite model the last two arguments are the primal mesh, `hexagon`, `cubed` (resolution =
cells per panel edge) or `triangles`, and the `horizontal_advection`: `dg` (default) or
`vector_invariant`.

A steady rotating wind in exact thermal-wind (gradient-wind) balance (`thermal_wind`). The
temperature falls towards the poles near the ground, `1/T = (1 + ε sin²φ sech²(z/z_s))/T0` with
T0 = 300 K, ε = 0.1, z_s = 12 km. The surface pressure is the same everywhere, so there is no
wind at the ground. The westerly grows up to about z_s and is nearly constant above (24 m/s at the
equator), so the damping layer sees no shear. The flow should stay at rest in the rotating
frame; the script prints max |w|, max |θ − θ(0)| and max |Δp_s| twice a day. 30 layers up to
30 km, Rayleigh damping of w above 22 km.

| model | max \|w\| day 1 | day 3 | day 5 |
|---|---|---|---|
| TRiSK, hexagon 4, dt 300 s | 2.6e-3 m/s | 8.4e-4 | 6.7e-4 (θ′ 0.35 K, Δp_s 0.35 hPa) |
| kite RT1 × Q2, upwind θ, hexagon 3 | 7.3e-4 | 0.30 | unstable |
| kite RT1 × Q2, DG θ (`dg_upwind`), vector-invariant momentum, hexagon 3 | 5.2e-5 | 2.9e-2 | unstable |
| kite RT1 × Q2, `dg_upwind`, `dg` advection, hexagon 3, dt 150 s | 6.6e-5 | 1.3e-4 | |
| the same, cubed sphere 12 | 4.0e-5 | 1.1e-4 | |
| the same, triangles 3 | 7.1e-5 | 1.5e-4 | |

The vector-invariant momentum advection is unstable in supercritical flow; see
[KITE3D_PLAN.md](KITE3D_PLAN.md).

## Baroclinic channel: `baroclinic_channel.jl`

```sh
julia --project examples/baroclinic_channel.jl 60 14 200 quadrilateral dg_upwind true dg
```

The baroclinic wave in a Cartesian f-plane channel of Ullrich, Reed and Jablonowski (2015)
(`baroclinic_channel`). The channel is 6000 km wide with walls at y = 0 and 6000 km; the jet
reaches u0 = 35 m/s, with f0 = 1.0313e-4 1/s, a flat ground and the lid at 30 km. A Gaussian
perturbation of u (1 m/s, 600 km) starts the wave. The kite model has no walls, so the box is
doubly periodic, 40000 km × 12000 km: the channel and its mirror image with the easterly jet. Then
v = 0 at the walls holds by symmetry, and the unperturbed state is an exact steady state.

- Arguments: `[nx=80] [days=12] [dt=150·80/nx] [cells=quadrilateral|triangle] [theta=dg_upwind]
  [perturbation=true] [advection=dg|vector_invariant]`, with ny = 3nx/10.
- 30 layers, Rayleigh damping of w above 22 km.
- Twice a day the script prints:
  - the minimum and maximum surface pressure and max |w|;
  - the amplitude of zonal wavenumber 10 of p_s in the channel, with its growth rate over the
    last half day (linear QG theory: 6.59e-6 1/s for this most unstable 4000 km wave);
  - the relative mass change and the time per step.

Only the kite model runs on planes; TRiSK and ICON need the spherical grids.

| advection `dg` | min p_s day 4 / 8 / 12 | k = 10 growth, days 3–12 | s/step (CPU) |
|---|---|---|---|
| 40 × 12 quadrilaterals (1000 km), dt 300 s | 999.6 / 997.7 / 979.8 hPa | | 2.1 |
| 40 × 12 triangles, dt 300 s | 999.6 / 997.8 / 980.7 hPa | | 3.3 |
| 60 × 18 quadrilaterals (667 km), dt 200 s | 999.7 / 997.7 / 978.7 hPa | 5.9–6.6e-6 1/s | 4.5 |
| 60 × 18 triangles, dt 200 s | 999.7 / 997.8 / … | 5.8–6.0e-6 1/s to day 8 | 7.1 |

With `vector_invariant` the triangles develop a grid-scale mode from day 1.

## Mass solves with cuDSS: `gpu/cudss_mass.jl`

```sh
srun -n 8 julia --project=examples/gpu examples/gpu/cudss_mass.jl cubed 64 1
```

Solves the RT_k, DG_k and CG_{k+1} mass matrices with cuDSS: one GPU per rank, and distributed over
all ranks in multi-GPU, multi-node mode. Each solve is checked against a known solution (errors
≤ 4e-15 on Levante). The distributed mode needs the cuDSS OpenMPI layer (`CUDSS_COMM_LIB`).
