"""
    MeshMetrics

Geometric quantities on the unit sphere, stored per stratum and indexed with
`cell_index`/`edge_index`. Edges are great-circle arcs; `edge_normals` are unit
tangent vectors at the edge midpoint pointing from the left to the right cell.
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
    edge_lengths = Vector{Float64}(undef, num_edges(dm))
    edge_midpoints = Vector{SVector{3, Float64}}(undef, num_edges(dm))
    edge_normals = Vector{SVector{3, Float64}}(undef, num_edges(dm))
    for e in edge_range(dm)
        i = edge_index(dm, e)
        p1, p2 = (get_coordinates(dm, v) for v in get_cone(dm, e))
        edge_lengths[i] = atan(norm(cross(p1, p2)), dot(p1, p2))
        edge_midpoints[i] = normalize(p1 + p2)
        # (p2 - p1) × (p1 + p2) = 2 p2 × p1: right of the edge, seen from outside
        edge_normals[i] = normalize(cross(p2, p1))
    end

    cell_areas = Vector{Float64}(undef, num_cells(dm))
    cell_centers = Vector{SVector{3, Float64}}(undef, num_cells(dm))
    for c in cell_range(dm)
        i = cell_index(dm, c)
        x = [get_coordinates(dm, v) for v in get_cell_vertices(dm, c)]
        cell_areas[i] = sum(spherical_triangle_area(x[1], x[k], x[k + 1]) for k in 2:length(x) - 1)
        cell_centers[i] = normalize(sum(x))
    end

    return MeshMetrics(cell_areas, cell_centers, edge_lengths, edge_midpoints, edge_normals)
end
