module DataInitModule

using LinearAlgebra
using StaticArrays
using ..DMPlexModule
using ..MetricModule

export init_edge_velocity, init_tracer

"""
Definiert ein analytisch vorgegebenes, starr rotierendes Windfeld um eine geneigte Achse.
"""
function analytical_wind_field(pos::SVector{3, Float64})::SVector{3, Float64}
    omega = SVector(0.1, 0.0, 1.0) # Rotationsachse (leicht gekippt)
    return cross(omega, pos)
end

"""
    init_edge_velocity(dm, metrics, vertices)

Berechnet den Fluss (Normalengeschwindigkeit) für jede Kante und speichert ihn in einem 
separaten, flachen Array der Länge `num_edges`. Dies entspricht dem diskreten Fluss-Layout in PETSc.
"""
function init_edge_velocity(dm::MyDMPlex, metrics::MeshMetrics, vertices::Vector{SVector{3, Float64}})::Vector{Float64}
    num_edges = length(dm.edge_start:dm.edge_end)
    U_normal = Vector{Float64}(undef, num_edges)

    for e_point in dm.edge_start:dm.edge_end
        e_idx = e_point - dm.edge_start + 1

        # Geometrischen Kantenmittelpunkt bestimmen
        v_cone = get_cone(dm, e_point)
        v1_idx = v_cone[1] - dm.vertex_start + 1
        v2_idx = v_cone[2] - dm.vertex_start + 1
        midpoint = (vertices[v1_idx] + vertices[v2_idx]) / 2.0
        midpoint = midpoint / norm(midpoint)

        # Analytischen Wind-Vektor abfragen
        u = analytical_wind_field(midpoint)

        # Geschwindigkeit auf die Kanten-Normale projizieren und im globalen Fluss-Array speichern
        U_normal[e_idx] = dot(u, metrics.edge_normals[e_point])
    end
    return U_normal
end

"""
    init_tracer(dm, vertices)

Initialisiert das stückweise konstante Tracerfeld als flachen Vektor auf den Zellen.
Platziert eine glatte Kosinus-Glocke als Anfangszustand auf der Kugeloberfläche.
"""
function init_tracer(dm::MyDMPlex, vertices::Vector{SVector{3, Float64}})::Vector{Float64}
    num_cells = length(dm.cell_start:dm.cell_end)
    C = Vector{Float64}(undef, num_cells)

    for cell_point in dm.cell_start:dm.cell_end
        c_idx = cell_point - dm.cell_start + 1

        # Zellzentrum näherungsweise bestimmen (Mittelwert der 3 Eckpunkte)
        edges = get_cone(dm, cell_point)
        v_indices = Int[]
        for e in edges
            append!(v_indices, get_cone(dm, e))
        end
        unique_v = unique(v_indices)
        
        center = SVector(0.0, 0.0, 0.0)
        for v in unique_v
            center += vertices[v - dm.vertex_start + 1]
        end
        center = center / length(unique_v)
        center = center / norm(center)

        # Kosinus-Glocke nahe dem Äquator (Zentrum bei x=1, y=0, z=0)
        dist = acos(clamp(dot(center, SVector(1.0, 0.0, 0.0)), -1.0, 1.0))
        C[c_idx] = dist < 0.5 ? 0.5 * (1.0 + cos(π * dist / 0.5)) : 0.0
    end
    return C
end

end # module

