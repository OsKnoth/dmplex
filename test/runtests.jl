using LinearAlgebra
using MPI
using Random
using SparseArrays
using StaticArrays
using Test
using dmplex

const MESHES = [
    ("icosahedral", create_spherical_mesh, 0:2),
    ("hexagonal", create_hexagon_mesh, 0:2),
    ("cubed sphere", create_cubed_sphere_mesh, 1:3),
]

include("topology.jl")
include("grids.jl")
include("finite_volumes.jl")
include("section.jl")
include("fem.jl")
include("shallow_water.jl")
include("kite.jl")
include("parallel.jl")
