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
using ..Topology
using ..Geometry
using ..FEM
using ..FEM: BlockInverse

export equal_area_zones, equal_area_partition, HaloExchange, exchange!, KiteDistribution, DofDistribution,
    is_distributed, global_sum, gather_owned, restrict

include("Partition.jl")
include("Halo.jl")
include("Distribution.jl")

end # module
