# dmplex.jl

`dmplex.jl` is a Julia-native mesh and finite-element library for the sphere,
modelled on PETSc's `DMPlex`, `PetscSection` and `PetscFE` but implemented
directly in Julia, without calling PETSc.

## Layout

| Directory | Module | Contents |
|---|---|---|
| `src/Topology` | `Topology` | `Plex` (mesh DAG in CSR form with PETSc orientations), `Section` (dof layout), `closure_dofs` |
| `src/Grids` | `Grids` | icosahedral, cubed-sphere, hexagonal and HEALPix meshes; ICON and MPAS grid-file readers (primal or dual) |
| `src/Geometry` | `Geometry` | reference triangle/quadrilateral, reference-to-sphere map, Piola maps, FV metrics |
| `src/FEM` | `FEM` | quadrature, element families of arbitrary order, `ReferenceElement`, `FESpace`, assembly (mass, divergence, Coriolis, load, projection) |
| `src/FiniteVolumes` | `FiniteVolumes` | first-order upwind advection |
| `src/TimeSteppers` | `TimeSteppers` | explicit Euler, classical RK4 |
| `src/ShallowWater` | `ShallowWater` | linear (RT_k × DG_k) and nonlinear vector-invariant (RT_k × DG_k × CG_{k+1}, PV form) rotating shallow water, mass and energy conserving |
| `src/Outputs` | `Outputs` | VTKHDF output for ParaView |

## Grids

```julia
dm = create_healpix_mesh(16)                              # 12·16² quads
dm = read_grid("grid/icon_grid_0010_R02B04_G.nc")          # ICON triangles
dm = read_grid("grid/x1.40962.grid.nc"; dual=true)         # MPAS Delaunay triangles
```

ICON and MPAS files are read with NCDatasets; cells and vertices keep the
file numbering. Open (regional or ocean) meshes such as MPAS-Ocean QU240 are
supported. Finite elements need triangles or quadrilaterals, so Voronoi grids
(MPAS primal, ICON dual) are for finite volumes until the kite grid exists.
Local grid files live in `grid/`, which is not tracked.

## Mesh topology

Points are numbered as in an interpolated PETSc mesh: cells, edges, vertices.
Edges run from their lower to their higher vertex, cells are counter-clockwise
seen from outside. Each cone entry stores a PETSc orientation (`0` along the
edge, `-1` reversed), so the cell with orientation `0` is the *left* cell of
an edge, and the edge normal points from left to right.

## Finite elements

`Lagrange(k)`, `Discontinuous(k)`, `RaviartThomas(k)` and `Nedelec(k)` are
available for any order, on triangles (P_k, RT_k, N_k) and quadrilaterals
(Q_k, RTCF_k, NCE_k). A `ReferenceElement` is built numerically. Its prime
basis is a tensor-Legendre basis, its dof functionals are GLL-node values,
normal/tangential components at edge Gauss points and interior moments, and
the basis is the dual of the functionals. Dofs come in PETSc closure order.
Global numbering is done only by the `Section`. `closure_dofs` reverses edge
nodes and flips signs on reversed edges, which makes the spaces H1-, H(div)-
or H(curl)-conforming on the sphere. The reference cells are mapped to the
sphere by radial projection, which reproduces the great-circle edges exactly.

```julia
using dmplex

dm = create_cubed_sphere_mesh(8)        # or create_spherical_mesh(3) for triangles
V = FESpace(dm, RaviartThomas(2))       # element chosen per cell shape
M = assemble_mass(V)

# mixed RT_k × DG_k on one section, e.g. for shallow water
s = Section(dm, field(RaviartThomas(1); name="u"), field(Discontinuous(1); name="h"); ordering=:field)
U, H = FESpace(dm, s, "u"), FESpace(dm, s, "h")
```

## Direction

The goal is finite-element modelling of the linear and nonlinear shallow-water
equations on the sphere, eventually with the kite-grid approach of
[CGDycore.jl](https://github.com/CliMA/CGDycore.jl/tree/main/src/FEM). A kite
grid splits every primal polygon into quadrilaterals; it will be built as an
ordinary quad `Plex` (with labels for parent cells and edge types), so all
quadrilateral elements above apply to it directly. This is an independent
implementation, not a wrapper around CGDycore or PETSc.

## Run

```sh
julia --project -e 'using Pkg; Pkg.test()'
julia --project examples/advection.jl 4 cubed            # FV upwind advection, writes VTKHDF
julia --project examples/linear_shallow_water.jl 2 cubed 8   # RT2 × DG2 linear shallow water
julia --project examples/nonlinear_shallow_water.jl 1 healpix 8   # Williamson TC2, also with an ICON/MPAS file
```
