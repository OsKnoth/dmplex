module SimulationModule

using LinearAlgebra
using StaticArrays
using ..DMPlexModule

export init_tracer, get_edge_fluxes, advection_step!

# Konstantes Windfeld: Solide Rotation um die polare/geneigte Achse
function wind_field(pos::SVector{3, Float64})
    omega = SVector(0.1, 0.0, 1.0) # Rotationsachse (leicht gekippt)
    return cross(omega, pos)
end

function init_tracer(dm::MyDMPlex)
    C = Dict{Int, Float64}()
    for cell in dm.cell_start:dm.cell_end
        # Ein "Schlauf-Tracer" (Kosinus-Glocke) auf der Kugeloberfläche initialisieren
        cone = get_cone(dm, cell)
        # Ungefähres Zentrum berechnen
        v_points = [get_cone(dm, e) for e in cone]
        all_v = unique(reduce(vcat, v_points))
        center = sum([dm.coordinates[v] for v in all_v]) / length(all_v)
        center /= norm(center)
        
        # Distanz zu einem Testpunkt auf der Kugel (z.B. nahe dem Äquator)
        dist = acos(clamp(dot(center, SVector(1.0, 0.0, 0.0)), -1.0, 1.0))
        C[cell] = dist < 0.5 ? 1.0 : 0.0
    end
    return C
end

function get_edge_fluxes(dm::MyDMPlex)
    U_n = Dict{Int, Float64}()
    for e in dm.edge_start:dm.edge_end
        # Mittelpunkt der Kante bestimmen
        v_cone = get_cone(dm, e)
        mid = (dm.coordinates[v_cone[1]] + dm.coordinates[v_cone[2]]) / 2.0
        mid /= norm(mid)
        
        u = wind_field(mid)
        # Normalenkomponente des Windfelds auf der Kante
        U_n[e] = dot(u, dm.edge_normals[e])
    end
    return U_n
end

function advection_step!(dm::MyDMPlex, C::Dict{Int, Float64}, U_n::Dict{Int, Float64}, dt::Float64)
    C_new = copy(C)
    
    for cell in dm.cell_start:dm.cell_end
        volume = dm.cell_volumes[cell]
        flux_sum = 0.0
        edges = get_cone(dm, cell)
        
        for e in edges
            length_e = dm.edge_lengths[e]
            u_n = U_n[e]
            
            # Wer sind die Nachbarn dieser Kante? (Support holen)
            neighbors = get_support(dm, e)
            if length(neighbors) < 2 continue end # Randbehandlung entfällt auf geschlossener Sphäre
            
            left_cell = neighbors[1]
            right_cell = neighbors[2]
            
            # Bestimmung der physikalischen Ausflussrichtung aus Sicht von 'cell'
            # Ist das Vorzeichen von u_n nach außen gerichtet?
            # Wir prüfen, ob die in der Kante gespeicherte Normale von left_cell zu right_cell zeigt
            is_left = (cell == left_cell)
            normal_sign = is_left ? 1.0 : -1.0
            
            # Effektive Geschwindigkeit nach außen
            velocity_out = u_n * normal_sign
            
            # --- Einfaches Upwind-Schema ---
            if velocity_out > 0.0
                # Fluss geht nach außen -> Nutze den Tracerwert der aktuellen Zelle
                flux_sum += length_e * velocity_out * C[cell]
            else
                # Fluss kommt von außen -> Nutze den Tracerwert des Nachbarn
                neighbor_cell = is_left ? right_cell : left_cell
                flux_sum += length_e * velocity_out * C[neighbor_cell]
            end
        end
        
        # Euler-Vorwärts Update
        C_new[cell] = C[cell] - (dt / volume) * flux_sum
    end
    
    return C_new
end

end # module

