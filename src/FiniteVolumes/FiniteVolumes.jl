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
export TRiSKGrid, TRiSKDistribution, TRiSKShallowWater, relative_vorticity, tangential_velocity

include("DataInit.jl")
include("Advection.jl")
include("TRiSKGrid.jl")
include("TRiSKDistribution.jl")
include("TRiSKShallowWater.jl")

end # module
