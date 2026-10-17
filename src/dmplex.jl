module dmplex

include("DMPlexModule.jl")
include("SphereMesh.jl")
include("CubedSphereMesh.jl")
include("HexagonMesh.jl")
include("MetricModule.jl")
include("DataInitModule.jl")
include("AdvectionModule.jl")
include("TimeStepperModule.jl")

using .DMPlexModule: MyDMPlex, add_cell_cone!, add_edge_cone!, get_cone,
    get_support_cells, get_support_points, get_transitive_closure, get_adjacency,
    get_cell_neighbors, set_cell_vertices, get_cell_vertices, set_label!,
    get_label_value, get_label_stratum, get_label_names, clear_label!,
    set_field!, get_field, get_field_names, remove_field!
using .SphereMesh: create_spherical_mesh
using .CubedSphereMesh: create_cubed_sphere_mesh
using .HexagonMesh: create_hexagon_mesh
using .MetricModule: MeshMetrics, compute_mesh_metrics
using .DataInitModule: init_edge_velocity, init_tracer
using .AdvectionModule: compute_advection_rhs!
using .TimeStepperModule: euler_forward_step!

export MyDMPlex, add_cell_cone!, add_edge_cone!, get_cone, get_support_cells,
    get_support_points, get_transitive_closure, get_adjacency, get_cell_neighbors,
    set_cell_vertices, get_cell_vertices, set_label!, get_label_value,
    get_label_stratum, get_label_names, clear_label!, set_field!, get_field,
    get_field_names, remove_field!
export create_spherical_mesh, create_cubed_sphere_mesh, create_hexagon_mesh
export MeshMetrics, compute_mesh_metrics, init_edge_velocity, init_tracer
export compute_advection_rhs!, euler_forward_step!

end # module dmplex
