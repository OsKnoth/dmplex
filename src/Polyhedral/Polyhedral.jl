"""
    Polyhedral

Finite volumes on polyhedral cells after POLYEDER: cell complexes with vertex coordinates
(`PolyComplex`, from an `ExtrudedPlex` and later from general 3D meshes), the exact integration of
monomials over cells and faces (`cell_moments`, `face_moments`) and the geometry derived from it
(`PolyGeometry`), the polynomial reconstruction of cell averages (`PolyReconstruction`) and the
transport with its upwind fluxes (`PolyAdvection`). Description: `docs/polyhedral/polyhedral.tex`.
"""
module Polyhedral

using LinearAlgebra
using SparseArrays
using StaticArrays
using ..Topology

export PolyComplex, extruded_complex, num_poly_cells, num_poly_faces, num_poly_edges, num_poly_vertices
export MonomialBasis, num_monomials, cell_moments, face_moments, polygon_moments, segment_moments
export PolyGeometry, face_cells, vertex_neighbours
export PolyReconstruction, reconstruct, face_value_matrices, point_value_matrices, face_quadrature
export PolyAdvection, stream_function_flux, upwind_time_step

include("Complex.jl")
include("Integration.jl")
include("Geometry.jl")
include("Reconstruction.jl")
include("Advection.jl")

end # module Polyhedral
