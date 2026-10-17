"""
    MeshMetrics

Geometric quantities of the mesh surface, stored per stratum and indexed with
`cell_index`/`edge_index`. On the sphere edges are great-circle arcs, on a
plane straight segments; `edge_normals` are unit tangent vectors at the edge
midpoint pointing from the left to the right cell.
"""
struct MeshMetrics
    cell_areas::Vector{Float64}
    cell_centers::Vector{SVector{3, Float64}}
    edge_lengths::Vector{Float64}
    edge_midpoints::Vector{SVector{3, Float64}}
    edge_normals::Vector{SVector{3, Float64}}
end

"Area of the spherical triangle `ABC` on the unit sphere (Van Oosterom–Strackee)."
function spherical_triangle_area(A, B, C)
    determinant = dot(A, cross(B, C))
    denominator = 1.0 + dot(A, B) + dot(B, C) + dot(C, A)
    return 2.0 * atan(abs(determinant), denominator)
end

function compute_mesh_metrics(dm::Plex)::MeshMetrics
    sphere = dm.surface == :sphere
    edge_lengths = Vector{Float64}(undef, num_edges(dm))
    edge_midpoints = Vector{SVector{3, Float64}}(undef, num_edges(dm))
    edge_normals = Vector{SVector{3, Float64}}(undef, num_edges(dm))
    for e in edge_range(dm)
        i = edge_index(dm, e)
        p1, p2 = edge_coordinates(dm, e)
        edge_midpoints[i] = surface_point(dm, p1 + p2)
        if sphere
            edge_lengths[i] = atan(norm(cross(p1, p2)), dot(p1, p2))
            # (p2 - p1) × (p1 + p2) = 2 p2 × p1: right of the edge, seen from outside
            edge_normals[i] = normalize(cross(p2, p1))
        else
            edge_lengths[i] = norm(p2 - p1)
            edge_normals[i] = normalize(cross(p2 - p1, SVector(0.0, 0.0, 1.0)))
        end
    end

    cell_areas = Vector{Float64}(undef, num_cells(dm))
    cell_centers = Vector{SVector{3, Float64}}(undef, num_cells(dm))
    for c in cell_range(dm)
        i = cell_index(dm, c)
        x = cell_coordinates(dm, c)
        cell_areas[i] = sphere ?
            sum(spherical_triangle_area(x[1], x[k], x[k + 1]) for k in 2:length(x) - 1) :
            sum(cross(x[k], x[mod1(k + 1, length(x))])[3] for k in eachindex(x)) / 2
        cell_centers[i] = sphere ? normalize(sum(x)) : surface_point(dm, sum(x) / length(x))
    end

    return MeshMetrics(cell_areas, cell_centers, edge_lengths, edge_midpoints, edge_normals)
end
