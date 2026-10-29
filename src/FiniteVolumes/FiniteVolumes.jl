"""
Cell-centered finite-volume operators, and the C-grid shallow-water model with
TRiSK operators on Voronoi meshes (as in ICON and MPAS).
"""
module FiniteVolumes

using LinearAlgebra
using StaticArrays
using KernelAbstractions
using MPI
using ..Topology
import ..Topology: num_cells, num_edges, num_vertices
using ..Geometry
using ..Geometry: spherical_triangle_area
using ..Grids: cell_centers, orthogonality_defect
import ..FEM
using ..FEM: to_backend
using ..Parallel: DofDistribution, HaloExchange, exchange!, gather, owned, num_local, equal_area_partition
import ..Parallel: is_distributed, global_sum
import ..ShallowWater: velocity_dofs, height_dofs, global_velocity_dofs, global_height_dofs, initial_state,
    tendency!, total_mass, total_energy, potential_enstrophy, gather_state

export init_edge_velocity, init_tracer, compute_advection_rhs!
export TRiSKGrid, TRiSKDistribution, TRiSKShallowWater, ICONShallowWater, gassmann_coefficients, rbf_coefficients,
       vertex_edge_areas,
       relative_vorticity, tangential_velocity

include("DataInit.jl")
include("Advection.jl")
include("TRiSKGrid.jl")
include("TRiSKDistribution.jl")
"""
C-grid shallow-water models on a `TRiSKGrid` (normal velocities on the edges,
heights at the cells, distributed by a `TRiSKDistribution`): the state layout,
`initial_state`, `gather_state`, the halo exchange and the diagnostics are
shared. Subtypes have the fields `grid`, `distribution`, `backend`,
`cell_areas`, `vertex_areas`, `vertex_edges`, `curl`, `vertex_cells`,
`vertex_height`, `coriolis` (at the vertices) and `work` (with `K`, `ζv`, `hv`,
`qv`), and implement `kinetic_energy!` and `tendency!`.
"""
abstract type CGridShallowWater end

include("TRiSKShallowWater.jl")
include("ICONShallowWater.jl")

end # module
