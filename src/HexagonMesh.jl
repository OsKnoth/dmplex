module HexagonMesh

using StaticArrays
using LinearAlgebra
using ..DMPlexModule
using ..SphereMesh

export create_hexagon_mesh

"""
    create_hexagon_mesh(subdivisions::Int)

Erzeugt die Geometrie und die PETSc-orientierte MyDMPlex-Topologie für ein Hexagon-Gitter
durch topologische Dualisierung des Ikosaeder-Kugelgitters.
"""
function create_hexagon_mesh(subdivisions::Int)
    # 1. Generiere zuerst das fehlerfreie Ikosaeder-Basisgitter
    ico_dm, ico_vertices = create_spherical_mesh(subdivisions)
    
    # 2. Geometrie des neuen Gitters:
    # Die neuen Vertices sind die Schwerpunkte (Zentren) des alten Dreiecks-Gitters
    num_new_vertices = length(ico_dm.cell_start:ico_dm.cell_end)
    vertices = Vector{SVector{3, Float64}}(undef, num_new_vertices)
    
    for ico_cell in ico_dm.cell_start:ico_dm.cell_end
        # Berechne das exakte Zentrum der alten Dreiecks-Zelle
        edges = get_cone(ico_dm, ico_cell)
        v_indices = Int[]
        for e in edges
            append!(v_indices, get_cone(ico_dm, abs(e)))
        end
        unique_v = unique(v_indices)
        
        center = zeros(SVector{3, Float64})
        for v in unique_v
            center += ico_vertices[v - ico_dm.vertex_start + 1]
        end
        center /= length(unique_v)
        
        # Neue Vertex-ID (1-basiert)
        new_v_id = ico_cell - ico_dm.cell_start + 1
        vertices[new_v_id] = center / norm(center) # Auf Einheitssphäre projizieren
    end

    # 3. Topologie des neuen Gitters:
    # Jedes alte Vertex-Point wird zu einer neuen Zelle.
    # Wir sammeln für jeden alten Knoten alle umgebenden alten Zellen (das entspricht ico_dm.supports).
    # Diese alten Zellen-IDs sind genau die neuen Vertex-IDs, die im Kreis angeordnet das neue Polygon bilden!
    poly_cells = Vector{Int}[]
    
    for ico_v in ico_dm.vertex_start:ico_dm.vertex_end
        # Finde alle alten Zellen, die an diesem alten Knoten hängen
        surrounding_ico_cells = Int[]
        for ico_cell in ico_dm.cell_start:ico_dm.cell_end
            edges = get_cone(ico_dm, ico_cell)
            v_indices = Int[]
            for e in edges; append!(v_indices, get_cone(ico_dm, abs(e))); end
            if ico_v in v_indices
                push!(surrounding_ico_cells, ico_cell - ico_dm.cell_start + 1)
            end
        end
        
        # Die neuen Vertex-IDs für dieses Polygon geometrisch im Kreis sortieren,
        # damit die Umlaufrichtung für ParaView und das Upwind-Schema stimmt (Gegenuhrzeigersinn)
        v_coords = [vertices[id] for id in surrounding_ico_cells]
        poly_center = sum(v_coords) / length(v_coords)
        poly_center /= norm(poly_center)
        
        e1 = v_coords[1] - poly_center
        angles = map(p -> begin
            v = p - poly_center
            atan(dot(cross(e1, v), poly_center), dot(e1, v))
        end, v_coords)
        
        p_perm = sortperm(angles)
        push!(poly_cells, surrounding_ico_cells[p_perm])
    end

    # --- SCHRITT 4: UNSTRUKTURIERTE DMPLEX TOPOLOGIE FÜR HEXAGONE AUFBAUEN ---
    dm = MyDMPlex()
    num_cells = length(poly_cells)

    # Kanten-Map extrahieren (Jedes n-Eck hat n Kanten)
    edge_map = Dict{Tuple{Int, Int}, Int}()
    edge_counter = 0
    for poly in poly_cells
        n_edges = length(poly)
        for i in 1:n_edges
            v1, v2 = poly[i], poly[mod1(i+1, n_edges)]
            edge = v1 < v2 ? (v1, v2) : (v2, v1)
            if !haskey(edge_map, edge)
                edge_counter += 1
                edge_map[edge] = edge_counter
            end
        end
    end
    num_edges = edge_counter

    dm.cell_start = 1; dm.cell_end = num_cells
    dm.edge_start = num_cells + 1; dm.edge_end = num_cells + num_edges
    dm.vertex_start = dm.edge_end + 1; dm.vertex_end = dm.edge_end + num_new_vertices

    # 1. Kanten-Cones befüllen
    for (edge, local_edge_id) in edge_map
        edge_point = dm.edge_start + (local_edge_id - 1)
        v1_val, v2_val = edge
        v1_point = dm.vertex_start + v1_val - 1
        v2_point = dm.vertex_start + v2_val - 1
        add_edge_cone!(dm, edge_point, [v1_point, v2_point])
    end

    # 2. Zellen-Cones befüllen mit signierten PETSc-IDs
    for (c_idx, poly) in enumerate(poly_cells)
        cell_point = dm.cell_start + (c_idx - 1)
        oriented_cones = Int[]
        n_edges = length(poly)
        
        center = zeros(SVector{3, Float64})
        for v in poly; center += vertices[v]; end
        center /= n_edges; center /= norm(center)

        for i in 1:n_edges
            v1, v2 = poly[i], poly[mod1(i+1, n_edges)]
            sorted_e = v1 < v2 ? (v1, v2) : (v2, v1)
            edge_id = dm.edge_start + (edge_map[sorted_e] - 1)
            
            p1, p2 = vertices[sorted_e[1]], vertices[sorted_e[2]]
            mid_edge = (p1 + p2) / 2.0; mid_edge /= norm(mid_edge)
            n_math = cross(mid_edge, p2 - p1); n_math /= norm(n_math)
            
            push!(oriented_cones, dot(n_math, mid_edge - center) >= 0.0 ? edge_id : -edge_id)
        end
        add_cell_cone!(dm, cell_point, oriented_cones)
        set_cell_vertices(dm, cell_point, [dm.vertex_start + vertex - 1 for vertex in poly])
    end

    return dm, vertices
end

norm(v) = sqrt(sum(v .^ 2))

end # module
