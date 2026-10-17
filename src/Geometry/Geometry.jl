"Reference cells, the reference-to-sphere map and finite-volume metrics."
module Geometry

using LinearAlgebra
using StaticArrays
using ..Topology

export ReferenceShape, Triangle, Quadrilateral, reference_shape, reference_vertices,
    reference_area, edge_point, edge_tangent, edge_normal
export CellMap, map_point, surface_normal, piola_hdiv, piola_hcurl
export MeshMetrics, compute_mesh_metrics

include("ReferenceShapes.jl")
include("SphereMapping.jl")
include("Metrics.jl")

end # module
