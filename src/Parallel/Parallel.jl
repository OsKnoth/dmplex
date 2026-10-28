"""
Distributed-memory parallelism: equal-area partitioning of the primal cells,
distribution of kite models over MPI ranks with one vertex ring of ghost
cells, and halo exchange of dof vectors.
"""
module Parallel

using LinearAlgebra
using MPI
using SparseArrays
using StaticArrays
using KernelAbstractions
using ..Topology
using ..Geometry
using ..FEM
using ..FEM: BlockInverse

export equal_area_zones, equal_area_partition, HaloExchange, exchange!, KiteDistribution, DofDistribution,
    is_distributed, global_sum, gather, local_solver, owned, num_local, owned_kites, num_local_kites,
    dof_owners, contiguous_numbering, distributed_rows, distributed_cudss_solver,
    CellDistribution, owned_cells, num_local_cells, GatheredMassSolver, distributed_mass_solver

include("Partition.jl")
include("Halo.jl")
include("Distribution.jl")
include("FEDistribution.jl")

end # module
