"Shallow-water models on the sphere built on the FEM spaces."
module ShallowWater

using LinearAlgebra
using StaticArrays
using SparseArrays
using ..Topology
using ..Grids: create_kite_mesh
using ..FEM

export LinearShallowWater, NonlinearShallowWater, LinearKiteShallowWater, velocity_dofs, height_dofs, initial_state, tendency!,
    total_mass, total_energy, potential_enstrophy, diagnostics

include("LinearShallowWater.jl")
include("NonlinearShallowWater.jl")
include("KiteShallowWater.jl")

end # module
