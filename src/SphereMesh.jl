module SphereMesh

using StaticArrays
using LinearAlgebra
using ..DMPlexModule

export create_spherical_mesh

const PHI = (1.0 + sqrt(5.0)) / 2.0

function create_spherical_mesh(subdivisions::Int)
    # 1. Geometrie: Die 12 Ur-Knoten (Jetzt sauber 1-basiert von 1 bis 12)
    raw_coords = [
        SVector(-1.0,  PHI,  0.0), SVector( 1.0,  PHI,  0.0), SVector(-1.0, -PHI,  0.0), SVector( 1.0, -PHI,  0.0),
        SVector( 0.0, -1.0,  PHI), SVector( 0.0,  1.0,  PHI), SVector( 0.0, -1.0, -PHI), SVector( 0.0,  1.0, -PHI),
        SVector( PHI,  0.0, -1.0), SVector( PHI,  0.0,  1.0), SVector(-PHI,  0.0, -1.0), SVector(-PHI,  0.0,  1.0)
    ]
    vertices = [v / norm(v) for v in raw_coords]
    
    # Deine exakten 20 Original-Dreiecke um +1 verschoben (vollständig geschlossen)
    base_tris = [
        (1, 12, 6), (1, 6, 2), (1, 2, 8), (1, 8, 11), (1, 11, 12),
        (2, 6, 10), (6, 12, 5), (12, 11, 3), (11, 8, 7), (8, 2, 9),
        (4, 10, 5), (4, 5, 3), (4, 3, 7), (4, 7, 9), (4, 9, 10),
        (5, 10, 6), (3, 5, 12), (7, 3, 11), (9, 7, 8), (10, 9, 2)
    ]

    # Geometrische Subdivision (Rein 1-basiert)
    for lvl in 1:subdivisions
        new_tris = Tuple{Int, Int, Int}[]
        midpoints = Dict{Tuple{Int, Int}, Int}()
        
        function get_midpoint(v1::Int, v2::Int)
            edge = v1 < v2 ? (v1, v2) : (v2, v1)
            if haskey(midpoints, edge)
                return midpoints[edge]
            end
            mid_pos = (vertices[v1] + vertices[v2]) / 2.0
            push!(vertices, mid_pos / norm(mid_pos))
            new_idx = length(vertices)
            midpoints[edge] = new_idx
            return new_idx
        end
        
        for tri in base_tris
            a, b, c = tri
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            
            push!(new_tris, (a, ab, ca))
            push!(new_tris, (b, bc, ab))
            push!(new_tris, (c, ca, bc))
            push!(new_tris, (ab, bc, ca))
        end
        base_tris = new_tris
    end

        # --- AUFBAU DER PETSC-KONFORMEN DMPLEX TOPOLOGIE ---
    dm = MyDMPlex()
    num_cells = length(base_tris)

    # Eindeutige Kanten sammeln
    edge_map = Dict{Tuple{Int, Int}, Int}()
    edge_counter = 0
    for tri in base_tris
        edges_of_tri = [(tri[1], tri[2]), (tri[2], tri[3]), (tri[3], tri[1])]
        for e in edges_of_tri
            sorted_e = e[1] < e[2] ? (e[1], e[2]) : (e[2], e[1])
            if !haskey(edge_map, sorted_e)
                edge_counter += 1
                edge_map[sorted_e] = edge_counter
            end
        end
    end
    num_edges = edge_counter
    num_vertices = length(vertices)

    dm.cell_start = 1; dm.cell_end = num_cells
    dm.edge_start = num_cells + 1; dm.edge_end = num_cells + num_edges
    dm.vertex_start = dm.edge_end + 1; dm.vertex_end = dm.edge_end + num_vertices

    # 1. Kanten-Cones befüllen (unorientiert auf Vertices)
    for (edge, local_edge_id) in edge_map
        edge_point = dm.edge_start + (local_edge_id - 1)
        v1_point = dm.vertex_start + (edge[1] - 1)
        v2_point = dm.vertex_start + (edge[2] - 1)
        add_edge_cone!(dm, edge_point, [v1_point, v2_point])
    end

    # 2. Zellen-Cones mit signierten IDs befüllen (Strikte PETSc Orientierung)
    for (c_idx, tri) in enumerate(base_tris)
        cell_point = dm.cell_start + (c_idx - 1)

        tri_edges = [(tri[1], tri[2]), (tri[2], tri[3]), (tri[3], tri[1])]
        oriented_cones = Int[]

        # Ungefähres Zentrum der aktuellen Zelle für Orientierungstest
        center = (vertices[tri[1]] + vertices[tri[2]] + vertices[tri[3]]) / 3.0
        center /= norm(center)

        for e in tri_edges
            sorted_e = e[1] < e[2] ? (e[1], e[2]) : (e[2], e[1])
            edge_id = dm.edge_start + (edge_map[sorted_e] - 1)

            # Geometrischer Orientierungstest bezüglich der vordefinierten mathematischen Kanten-Normale
            p1, p2 = vertices[sorted_e[1]], vertices[sorted_e[2]]
            mid_edge = (p1 + p2) / 2.0; mid_edge /= norm(mid_edge)
            n_math = cross(mid_edge, p2 - p1); n_math /= norm(n_math)

            # Zeigt die mathematische Normale von der Zelle WEG?
            outward_vector = mid_edge - center
            is_outward = dot(n_math, outward_vector) >= 0.0

            # PETSc-Konvention: Wenn n_math nach innen zeigt, wird die ID negativ markiert
            push!(oriented_cones, is_outward ? edge_id : -edge_id)
        end

        # Übergabe an den DMPlex, der im Hintergrund die edge_to_cells Tabelle baut
        add_cell_cone!(dm, cell_point, oriented_cones)
        set_cell_vertices(dm, cell_point, [dm.vertex_start + vertex - 1 for vertex in tri])
    end

    return dm, vertices
end

norm(v) = sqrt(sum(v .^ 2))

end # module
