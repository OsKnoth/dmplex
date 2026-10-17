# dmplex.jl

`dmplex.jl` is a Julia-native mesh and discretization library for the sphere,
modelled on PETSc's `DMPlex` and `PetscSection` but implemented directly in
Julia, without calling PETSc.

## Mesh topology (`Plex`)

- Icosahedral, cubed-sphere (equiangular) and hexagonal (dual) meshes of the
  unit sphere, all built through `plex_from_polygons`.
- Points are numbered as in an interpolated PETSc mesh: cells, edges, vertices.
  Cones and supports live in compressed (CSR) arrays; each cone entry carries a
  PETSc orientation (`0` along the edge, `-1` reversed).
- Conventions: edges run from their lower to their higher vertex, cells are
  counter-clockwise seen from outside. The cell with orientation `0` is the
  *left* cell of an edge, and the edge normal points from left to right.
- Queries: `get_cone`, `get_support`, `get_closure` (PETSc order), `get_star`,
  `get_adjacency` (FEM or FVM), `get_edge_cells`, `boundary_matrix`, labels,
  and `check_plex` for the topological invariants.

## Dof layout (`Section`)

A `Section` assigns degrees of freedom to mesh points for one or more fields:
`lagrange(k)`, `discontinuous(k)`, `raviart_thomas(k)` and `nedelec(k)` on
triangles and quadrilaterals, with point-major (PETSc) or field-block ordering.
`closure_dofs` returns the global indices of a cell's dofs in closure order. It
reverses edge nodes and flips H(div)/H(curl) signs for reversed edges, so
neighbouring cells assemble consistently.

```julia
using dmplex

dm = create_cubed_sphere_mesh(8)
check_plex(dm)
metrics = compute_mesh_metrics(dm)

# mixed RT0 × DG0 layout, e.g. for linear shallow water
s = Section(dm, raviart_thomas(0; name="u"), discontinuous(0; name="h"))
indices, signs = closure_dofs(dm, s, first(cell_range(dm)), "u")
```

## Direction

The goal is finite-element modelling of the linear and nonlinear shallow-water
equations on the sphere. Next steps: reference elements and quadrature, the
reference-to-sphere mapping, sparse operator assembly on top of `Section`,
and time integration. The [CGDycore.jl FEM sources](https://github.com/CliMA/CGDycore.jl/tree/main/src/FEM)
are a reference for organization and formulations; this is an independent
implementation, not a wrapper around that code or PETSc.

## Run

```sh
julia --project -e 'using Pkg; Pkg.test()'
julia --project examples/advection.jl 4 cubed   # FV upwind advection, writes VTKHDF
```
