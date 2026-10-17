# First-order upwind advection of a cosine bell by solid-body rotation.
#
#   julia --project examples/advection.jl [resolution] [icosahedron|cubed|hexagon]
#
# resolution is the number of refinements (icosahedron, hexagon) or the number
# of cells per panel edge (cubed sphere). Writes a VTKHDF file for ParaView.
using dmplex

function main(args)
    resolution = length(args) >= 1 ? parse(Int, args[1]) : 4
    mesh_type = length(args) >= 2 ? args[2] : "icosahedron"

    dm = mesh_type == "cubed" ? create_cubed_sphere_mesh(resolution) :
         mesh_type == "hexagon" ? create_hexagon_mesh(resolution) :
         create_spherical_mesh(resolution)
    check_plex(dm)
    println("$mesh_type mesh: $(num_cells(dm)) cells, $(num_edges(dm)) edges, $(num_vertices(dm)) vertices")

    metrics = compute_mesh_metrics(dm)
    C_curr = init_tracer(dm, metrics)
    C_next = similar(C_curr)
    U_normal = init_edge_velocity(dm, metrics)
    rhs = similar(C_curr)

    dt = 0.15 * sqrt(minimum(metrics.cell_areas)) / (maximum(abs, U_normal) + 1e-8)
    mass(C) = sum(C .* metrics.cell_areas)
    initial_mass = mass(C_curr)
    println("initial mass $(round(initial_mass, digits=6)), dt = $(round(dt, digits=5))")

    t, t_end, step = 0.0, 3.0, 0
    while t < t_end
        euler_forward_step!(C_next, C_curr, dt, rhs, compute_advection_rhs!, dm, metrics, U_normal)
        C_curr, C_next = C_next, C_curr
        t += dt
        step += 1
        if step % 100 == 0
            error = abs(mass(C_curr) - initial_mass) / initial_mass
            println("step $step  t = $(round(t, digits=3))  relative mass error = $error")
        end
    end

    filename = write_vtkhdf("spherical_advection_$(mesh_type)_res$(resolution).vtkhdf", dm; tracer=C_curr)
    println("wrote $filename")
end

main(ARGS)
