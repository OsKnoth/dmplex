"Shallow-water models on the sphere built on the FEM spaces."
module ShallowWater

using LinearAlgebra
using SparseArrays
using ..Topology
using ..FEM

export LinearShallowWater, velocity_dofs, height_dofs, initial_state, tendency!, total_mass, total_energy

include("LinearShallowWater.jl")

end # module
