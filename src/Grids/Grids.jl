"Spherical mesh generators; all return an interpolated `Plex` on the unit sphere."
module Grids

using LinearAlgebra
using StaticArrays
using ..Topology

export create_spherical_mesh, create_cubed_sphere_mesh, create_hexagon_mesh

include("Icosahedral.jl")
include("CubedSphere.jl")
include("Hexagonal.jl")

end # module
