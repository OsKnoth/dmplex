# dmplex.jl

`dmplex.jl` is a Julia-native mesh and finite-element library for the sphere,
modelled on PETSc's `DMPlex`, `PetscSection` and `PetscFE` but implemented
directly in Julia, without calling PETSc.

## Layout

| Directory | Module | Contents |
|---|---|---|
| `src/Topology` | `Topology` | `Plex` (mesh DAG in CSR form with PETSc orientations), `Section` (dof layout), `closure_dofs` |
| `src/Grids` | `Grids` | icosahedral, cubed-sphere and hexagonal meshes of the unit sphere |
| `src/Geometry` | `Geometry` | reference triangle/quadrilateral, reference-to-sphere map, Piola maps, FV metrics |
| `src/FEM` | `FEM` | quadrature, element families of arbitrary order, `ReferenceElement`, `FESpace`, mass assembly |
| `src/FiniteVolumes` | `FiniteVolumes` | first-order upwind advection |
| `src/TimeSteppers` | `TimeSteppers` | explicit Euler |
| `src/Outputs` | `Outputs` | VTKHDF output for ParaView |

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
julia --project examples/advection.jl 4 cubed   # FV upwind advection, writes VTKHDF
```
