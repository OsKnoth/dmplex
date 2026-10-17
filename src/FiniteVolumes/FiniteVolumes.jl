"Cell-centered finite-volume operators."
module FiniteVolumes

using LinearAlgebra
using StaticArrays
using ..Topology
using ..Geometry

export init_edge_velocity, init_tracer, compute_advection_rhs!

include("DataInit.jl")
include("Advection.jl")

end # module
