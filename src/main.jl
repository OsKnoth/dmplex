# main.jl
using VTKHDF
using WriteVTK: VTKCellTypes, MeshCell, VTKCellData
using StaticArrays
using LinearAlgebra

include("DMPlexModule.jl")
include("SphereMesh.jl")
include("CubedSphereMesh.jl") # NEU
include("HexagonMesh.jl") # NEU
include("MetricModule.jl")
include("DataInitModule.jl")
include("AdvectionModule.jl")
include("TimeStepperModule.jl")
include("HexagonMesh.jl")

using .DMPlexModule
using .SphereMesh
using .CubedSphereMesh # NEU
using .HexagonMesh
using .MetricModule
using .DataInitModule
using .AdvectionModule
using .TimeStepperModule

"""
    export_general_simulation_to_vtkhdf(dm, vertices, C, filename_base)

Universeller Polygon-Export für ParaView.
Sortiert die unstrukturierten Knoten jeder Zelle mathematisch exakt über den Polarwinkel,
um Faltungen und Löcher (Backface-Culling) an allen Elementübergängen komplett auszuschließen.
"""

"""
    export_general_simulation_to_vtkhdf(dm, vertices, C, filename_base)

Universeller Export, der die Knoten der Zellen direkt aus dem DMPlex-Topologie-Cache liest.
Schließt jegliche Orientierungsfehler beim VTK-Export komplett aus.
"""
function export_general_simulation_to_vtkhdf(dm::MyDMPlex, vertices::Vector{SVector{3, Float64}}, C::Vector{Float64}, filename_base::String)
    num_vertices = length(vertices)

    # 1. Geometrie-Matrix vorbereiten
    points_matrix = Matrix{Float64}(undef, 3, num_vertices)
    for i in 1:num_vertices
        @show i,vertices[i]
        points_matrix[:, i] = vertices[i]
    end

    # 2. Topologie direkt aus dem cell_to_vertices Cache auslesen
    cells = dm.cell_start:dm.cell_end
    num_cells = length(cells)
    julia_cells = Vector{MeshCell}(undef, num_cells)

    @show dm.cell_to_vertices
    for (c_idx, cell_point) in enumerate(cells)
        # Direktes Abfragen der globalen Vertex-Points (kein Kanten-Cone-Suchen mehr!)
        global_v_indices = get_cell_vertices(dm, cell_point)
        num_nodes = length(global_v_indices)

        # Rückrechnung in sequentielle 1-basierte VTK-Indizes (1 bis N)
        vtk_indices = [v - dm.vertex_start + 1 for v in global_v_indices]

        # Zelltyp bestimmen
        cell_type = if num_nodes == 3
            VTKCellTypes.VTK_TRIANGLE
        elseif num_nodes == 4
            VTKCellTypes.VTK_QUAD
        else
            VTKCellTypes.VTK_POLYGON
        end
        @show cell_type,vtk_indices
        julia_cells[c_idx] = MeshCell(cell_type, vtk_indices)
    end

    # 3. Schreiben der VTKHDF-Datei
    println("Schreibe geschlossenes VTKHDF-Gitter via cached cell_to_vertices...")
    vtkhdf_grid(filename_base, points_matrix, julia_cells) do vtk
        vtk["Tracer", VTKCellData()] = C
    end
end

function main()
    # Erster Parameter: Auflösung (Für Ikosaeder = RefinementLevel, für Cubed Sphere = N_panel)
    res_param = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
    mesh_type = length(ARGS) >= 2 ? ARGS[2] : "icosahedron"

    println("=== Allokationsfreie Multi-Grid FV-Advektion ===")

    # 1. Gittertyp-Umschaltung mit angepasster Parameterbedeutung
    dm, vertices = if mesh_type == "cubed"
        println("Gittertyp: Cubed Sphere | Zellen pro Panelrichtung (N): $res_param")
        create_cubed_sphere_mesh(res_param)
    elseif mesh_type == "hexagon"
        println("Gittertyp: Hexagon-Kugel (Dual-Mesh) | Ausgangs-Refinements: $res_param")
        create_hexagon_mesh(res_param)
    else
        println("Gittertyp: Ikosaeder-Kugel | Refinement-Level: $res_param")
        create_spherical_mesh(res_param)
    end

    num_cells = length(dm.cell_start:dm.cell_end)
    println("Topologie erfolgreich aufgebaut: $num_cells Zellen im Gitter.")

    
    # 2. Metriken und Felder initialisieren
    metrics = compute_mesh_metrics(dm, vertices)
    @show dm.cell_start,dm.cell_end
    @show dm.edge_start,dm.edge_end
    C_curr = init_tracer(dm, vertices)
    C_next = copy(C_curr)
    @show sum(abs.(C_next))
    U_normal = init_edge_velocity(dm, metrics, vertices)
    rhs_cache = zeros(Float64, length(C_curr))
    
    # 3. CFL Zeitschritt
    min_vol = minimum(values(metrics.cell_volumes))
    @show min_vol,findmin(metrics.cell_volumes)
    eff_edge_length = sqrt(min_vol)
    max_u = maximum(abs.(U_normal))
    dt = 0.15 * (eff_edge_length / (max_u + 1e-8)) 
    
    t = 0.0; t_end = 3.0; step = 0
    initial_mass = sum([C_curr[i] * metrics.cell_volumes[p] for (i, p) in enumerate(dm.cell_start:dm.cell_end)])
    println("Start-Gesamtmasse: $(round(initial_mass, digits=6)) | dt = $(round(dt, digits=5))")
    
    # 4. Zeitschleife
    while t < t_end
        euler_forward_step!(C_next, C_curr, dt, rhs_cache, compute_advection_rhs!, dm, metrics, U_normal)
        @show sum(abs.(C_next))
        C_tmp = C_curr; C_curr = C_next; C_next = C_tmp
        t += dt; step += 1
        
        if step % 100 == 0
            current_mass = sum([C_curr[i] * metrics.cell_volumes[p] for (i, p) in enumerate(dm.cell_start:dm.cell_end)])
            mass_error = abs(current_mass - initial_mass) / (initial_mass + 1e-12)
            println("Schritt $step | Zeit: $(round(t, digits=3)) | rel. Massenfehler: $(round(mass_error * 100, digits=5)) %")
        end
    end
    
    # 5. Exportieren
    filename = "spherical_advection_$(mesh_type)_res$(res_param)"
    export_general_simulation_to_vtkhdf(dm, vertices, C_curr, filename)
    println("=== Simulation erfolgreich beendet! ===")
end

main()

