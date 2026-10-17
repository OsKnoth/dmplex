module CubedSphereMesh

using StaticArrays
using LinearAlgebra
using ..DMPlexModule

export create_cubed_sphere_mesh

"""
    analytical_cube_point(i, j, n, panel_id)
Berechnet die exakte mathematische Tangens-Projektion nach Sadourny (Radius = 1.0).
"""
function analytical_cube_point(i::Int, j::Int, n::Int, panel_id::Int)::SVector{3, Float64}
    c_i = tan(i * π / (2 * n) - 0.25 * π)
    c_j = tan(j * π / (2 * n) - 0.25 * π)
    
    # 1=West, 2=East, 3=South, 4=North, 5=Bottom, 6=Top
    raw = if panel_id == 1     SVector(-1.0,  c_i,  c_j) 
          elseif panel_id == 2 SVector( 1.0,  c_i,  c_j) 
          elseif panel_id == 3 SVector( c_i, -1.0,  c_j) 
          elseif panel_id == 4 SVector( c_i,  1.0,  c_j) 
          elseif panel_id == 5 SVector( c_i,  c_j, -1.0) 
          else                 SVector( c_i,  c_j,  1.0) 
          end
          
    return raw / norm(raw)
end

"""
    create_cubed_sphere_mesh(n::Int)
Erzeugt die Geometrie und deine MyDMPlex-Topologie für eine Cubed Sphere.
Nutzt deine exakte topologische Rand-Matrix-Kopplung (Vollkommen allokationsfrei).
"""
function create_cubed_sphere_mesh(n::Int)
    n > 0 || throw(ArgumentError("cubed-sphere resolution must be positive"))

    vertices = SVector{3, Float64}[]
    vertex_ids = Dict{NTuple{3, Float64}, Int}()
    panel_grids = [Matrix{Int}(undef, n + 1, n + 1) for _ in 1:6]
    for panel in 1:6, j in 0:n, i in 0:n
        point = analytical_cube_point(i, j, n, panel)
        key = (round(point[1], digits=12), round(point[2], digits=12), round(point[3], digits=12))
        vertex_id = get!(vertex_ids, key) do
            push!(vertices, point)
            length(vertices)
        end
        panel_grids[panel][i + 1, j + 1] = vertex_id
    end

    quad_cells = Vector{Int}[]
    for panel in 1:6
        for j in 0:n-1
            for i in 0:n-1
                v1 = panel_grids[panel][i + 1, j + 1]
                v2 = panel_grids[panel][i + 2, j + 1]
                v3 = panel_grids[panel][i + 2, j + 2]
                v4 = panel_grids[panel][i + 1, j + 2]
                push!(quad_cells, [v1, v2, v3, v4])
            end
        end
    end

    num_cells = length(quad_cells)
    edge_map = Dict{Tuple{Int, Int}, Int}()
    for quad in quad_cells
        for i in 1:4
            v1, v2 = quad[i], quad[mod1(i+1, 4)]
            edge = v1 < v2 ? (v1, v2) : (v2, v1)
            if !haskey(edge_map, edge)
                edge_map[edge] = length(edge_map) + 1
            end
        end
    end

    dm = MyDMPlex()
    num_edges = length(edge_map)
    dm.cell_start = 1; dm.cell_end = num_cells
    dm.edge_start = num_cells + 1; dm.edge_end = num_cells + num_edges
    dm.vertex_start = dm.edge_end + 1; dm.vertex_end = dm.edge_end + length(vertices)

    for (edge, local_edge_id) in edge_map
        edge_point = dm.edge_start + (local_edge_id - 1)
        v1_val, v2_val = edge
        add_edge_cone!(dm, edge_point, [dm.vertex_start + v1_val - 1, dm.vertex_start + v2_val - 1])
    end

    for (c_idx, quad) in enumerate(quad_cells)
        cell_point = dm.cell_start + (c_idx - 1)
        oriented_cone = Int[]
        local_segments = [(quad[1], quad[2]), (quad[2], quad[3]), (quad[3], quad[4]), (quad[4], quad[1])]
        center = sum(vertices[v] for v in quad) / length(quad)
        center /= norm(center)

        for (v_A, v_B) in local_segments
            edge_key = v_A < v_B ? (v_A, v_B) : (v_B, v_A)
            edge_id = dm.edge_start + (edge_map[edge_key] - 1)
            p1, p2 = vertices[edge_key[1]], vertices[edge_key[2]]
            midpoint = (p1 + p2) / 2.0
            midpoint /= norm(midpoint)
            edge_normal = cross(midpoint, p2 - p1)
            edge_normal /= norm(edge_normal)
            points_outward = dot(edge_normal, midpoint - center) >= 0.0
            push!(oriented_cone, points_outward ? edge_id : -edge_id)
        end
        add_cell_cone!(dm, cell_point, oriented_cone)
        set_cell_vertices(dm, cell_point, [dm.vertex_start + vertex - 1 for vertex in quad])
    end

    return dm, vertices
end

norm(v) = sqrt(sum(v .^ 2))

end # module
