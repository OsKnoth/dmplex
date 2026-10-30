"""
Mesh topology (`Plex`, analogue of PETSc `DMPlex`) and dof layout (`Section`,
analogue of `PetscSection`).
"""
module Topology

using LinearAlgebra
using SparseArrays
using StaticArrays

export Plex, plex_from_polygons,
    get_chart, get_depth_stratum, get_height_stratum, get_depth,
    cell_range, edge_range, vertex_range, num_cells, num_edges, num_vertices,
    cell_index, edge_index, vertex_index, get_coordinates,
    get_cone, get_cone_size, get_cone_orientation, get_support, get_support_size,
    get_closure, get_star, get_cell_vertices, get_edge_cells, edge_sign,
    get_adjacency, get_cell_neighbors, boundary_matrix, check_plex,
    set_label!, get_label_value, get_label_stratum, get_label_names, clear_label!,
    cell_coordinates, edge_coordinates, surface_point, reference_normal, unwrap
export FieldLayout, Section, num_nodes, num_fields, field_names, field_index, get_dof, get_offset,
    get_field_dof, get_field_offset, storage_size, dof_range, field_range,
    closure_dofs, create_vector
export ExtrudedPlex, num_layers, num_levels, extruded_point, base_point, get_cone_signs

include("Plex.jl")
include("Section.jl")
include("Extruded.jl")

end # module
