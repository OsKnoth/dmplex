module MetricModule

using StaticArrays
using LinearAlgebra
using ..DMPlexModule

export MeshMetrics, compute_mesh_metrics

struct MeshMetrics
    edge_lengths::Dict{Int, Float64}
    edge_normals::Dict{Int, SVector{3, Float64}}
    cell_volumes::Dict{Int, Float64}
end

function compute_spherical_triangle_volume(A, B, C)
    determinant = dot(A, cross(B, C))
    denominator = 1.0 + dot(A, B) + dot(B, C) + dot(C, A)
    return 2.0 * atan(abs(determinant), denominator)
end

function compute_mesh_metrics(dm::MyDMPlex, vertices::Vector{SVector{3, Float64}})::MeshMetrics
    edge_lengths = Dict{Int, Float64}()
    edge_normals = Dict{Int, SVector{3, Float64}}()
    cell_volumes = Dict{Int, Float64}()

    # 1. Kantenmetriken
    for e_point in dm.edge_start:dm.edge_end
        v_cone = get_cone(dm, e_point)
        p1 = vertices[v_cone[1] - dm.vertex_start + 1]
        p2 = vertices[v_cone[2] - dm.vertex_start + 1]
        
        edge_lengths[e_point] = acos(clamp(dot(p1, p2), -1.0, 1.0))
        mid = (p1 + p2) / 2.0; mid /= norm(mid)
        n = cross(mid, p2 - p1)
        edge_normals[e_point] = n / norm(n)
    end

    # 2. Polygon-unabhängige Volumenberechnung
    for cell_point in dm.cell_start:dm.cell_end
        edges = get_cone(dm, cell_point)
        cell_vertex_points = get_cell_vertices(dm, cell_point)
        if isempty(cell_vertex_points)
            v_indices = Int[]
            for oriented_edge in edges
                append!(v_indices, get_cone(dm, abs(oriented_edge)))
            end
            cell_vertex_points = unique(v_indices)
        end

        cell_nodes = [vertices[v - dm.vertex_start + 1] for v in cell_vertex_points]
        area = 0.0
        for i in 2:length(cell_nodes)-1
            area += compute_spherical_triangle_volume(cell_nodes[1], cell_nodes[i], cell_nodes[i + 1])
        end
        cell_volumes[cell_point] = area
    end

    return MeshMetrics(edge_lengths, edge_normals, cell_volumes)
end

norm(v) = sqrt(sum(v .^ 2))

end # module
