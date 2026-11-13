"""
    Polyhedral

Finite volumes on polyhedral cells after POLYEDER: cell complexes with vertex coordinates
(`PolyComplex`, from an `ExtrudedPlex` and later from general 3D meshes), the exact integration of
monomials over cells and faces (`cell_moments`, `face_moments`) and the geometry derived from it
(`PolyGeometry`), the polynomial reconstruction of cell averages (`PolyReconstruction`), the
transport with its upwind fluxes (`PolyAdvection`) and the shallow-water equations on one layer
(`PolyShallowWater`, KernelAbstractions kernels, MPI), and cut cells over orography (`cut_complex`,
`merge_small_cells`). Description: `docs/polyhedral/polyhedral.tex`.
"""
module Polyhedral

using LinearAlgebra
using SparseArrays
using StaticArrays
using KernelAbstractions
using MPI
using ..Topology
import ..FEM
using ..FEM: to_backend
using ..Parallel: DofDistribution, exchange!, gather, num_local, equal_area_partition
import ..Parallel: is_distributed, global_sum
import ..ShallowWater: initial_state, gather_state, total_mass, tendency!, height_dofs, velocity_dofs
using ..FiniteVolumes: merged_halo

export PolyComplex, extruded_complex, num_poly_cells, num_poly_faces, num_poly_edges, num_poly_vertices
export MonomialBasis, num_monomials, cell_moments, face_moments, polygon_moments, segment_moments
export PolyGeometry, face_cells, vertex_neighbours
export PolyReconstruction, reconstruct, face_value_matrices, point_value_matrices, face_quadrature
export PolyAdvection, stream_function_flux, upwind_time_step
export interaction_operators, cell_vectors, cell_vector_matrices, face_normal_components, linear_vector_reconstruction
export PolyShallowWater, shallow_water_time_step, cell_momentum!
export cut_complex, merge_cells, merge_small_cells, vector_potential_flux

include("Complex.jl")
include("Integration.jl")
include("Geometry.jl")
include("Reconstruction.jl")
include("Advection.jl")
include("InteractionRegion.jl")
include("ShallowWater.jl")
include("CutCell.jl")

end # module Polyhedral
