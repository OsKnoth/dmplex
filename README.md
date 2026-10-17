# dmplex.jl

`dmplex.jl` is an early, Julia-native mesh and discretization project inspired by
PETSc DMPlex. Its goal is to develop mesh topology and numerical methods
directly in Julia, without calling the PETSc interface.

The current prototype builds icosahedral, cubed-sphere, and dual hexagonal
spherical meshes. It provides oriented cell-edge topology, labels, transitive
closures, adjacency queries, point-wise field storage, spherical metrics, and a
conservative first-order upwind advection operator.

```julia
using dmplex

dm, vertices = create_spherical_mesh(1)
metrics = compute_mesh_metrics(dm, vertices)
tracer = init_tracer(dm, vertices)

set_label!(dm, "surface-region", dm.cell_start, 1)
closure = get_transitive_closure(dm, dm.cell_start)
neighbors = get_cell_neighbors(dm, dm.cell_start)
set_field!(dm, "tracer", dm.cell_start, tracer[1])
```

## Direction

The longer-term goal is finite-element modeling of the linear and nonlinear
shallow-water equations on the sphere. The intended development path is to
build on the topology and field APIs with finite-element spaces, basis
evaluation and quadrature, sparse operator assembly, and time integration.
Mixed and discontinuous spaces for velocity and height, and the discrete
divergence/gradient operators, will be important parts of that work.

The existing [CGDycore.jl FEM sources](https://github.com/CliMA/CGDycore.jl/tree/main/src/FEM)
are a useful reference for finite-element organization and shallow-water
formulations; this project is an independent Julia implementation, not a
wrapper around that code or PETSc. Finite-element shallow-water solvers are
future work and are not yet implemented here.

## Test

```sh
julia --project -e 'using Pkg; Pkg.test()'
```
