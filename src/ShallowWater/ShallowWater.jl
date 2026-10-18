"Shallow-water models on the sphere built on the FEM spaces."
module ShallowWater

using LinearAlgebra
using StaticArrays
using SparseArrays
using ..Topology
using ..Grids: create_kite_mesh
using ..Geometry
using ..FEM
using ..Parallel
using Atomix: @atomic
using KernelAbstractions

export LinearShallowWater, NonlinearShallowWater, LinearKiteShallowWater, NonlinearKiteShallowWater, velocity_dofs, height_dofs, initial_state, tendency!,
    total_mass, total_energy, potential_enstrophy, diagnostics, gather_state,
    global_velocity_dofs, global_height_dofs

include("LinearShallowWater.jl")
include("NonlinearShallowWater.jl")
include("KiteShallowWater.jl")
include("NonlinearKiteShallowWater.jl")

end # module
