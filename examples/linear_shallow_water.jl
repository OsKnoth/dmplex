# Linear rotating shallow water on the sphere, RT_k × DG_k.
#
#   julia --project examples/linear_shallow_water.jl [order] [icosahedron|cubed] [resolution]
#
# A geostrophically balanced solid-body rotation with a Gaussian height bump on
# top, which radiates inertia-gravity waves. Prints mass and energy, which the
# semi-discretization conserves, and writes cell-center heights as VTKHDF.
using LinearAlgebra
using StaticArrays
using dmplex

function main(args)
    k = length(args) >= 1 ? parse(Int, args[1]) : 1
    mesh_type = length(args) >= 2 ? args[2] : "cubed"
    resolution = length(args) >= 3 ? parse(Int, args[3]) : (mesh_type == "cubed" ? 8 : 3)

    dm = mesh_type == "cubed" ? create_cubed_sphere_mesh(resolution) : create_spherical_mesh(resolution)
    model = LinearShallowWater(dm, k; gravity=1.0, depth=1.0, coriolis=x -> 2x[3])
    println("$mesh_type mesh, $(num_cells(dm)) cells, RT$k × DG$k: $(num_dofs(model.U)) + $(num_dofs(model.H)) dofs")

    u0 = 0.5
    bump_center = normalize(SVector(1.0, 0.0, 0.5))
    velocity(x) = u0 * SVector(-x[2], x[1], 0.0)
    height(x) = -u0 * x[3]^2 + 0.1 * exp(-20 * (1 - dot(x, bump_center)))
    y = initial_state(model, velocity, height)

    cache = RK4Cache(y)
    dt = 0.1 * sqrt(4π / num_cells(dm)) / (k + 1)^2
    nsteps = ceil(Int, 2.0 / dt)
    mass0, energy0 = total_mass(model, y), total_energy(model, y)
    for step in 1:nsteps
        rk4_step!(y, dt, tendency!, cache, model)
        if step % max(1, nsteps ÷ 5) == 0 || step == nsteps
            println("t = $(round(step * dt, digits=3))  mass error = $(total_mass(model, y) - mass0)  ",
                    "relative energy error = $((total_energy(model, y) - energy0) / energy0)")
        end
    end

    h = y[height_dofs(model)]
    center(c) = get_cone_size(dm, c) == 3 ? SVector(-1 / 3, -1 / 3) : SVector(0.0, 0.0)
    cell_height = [evaluate(model.H, h, c, center(c))[2] for c in cell_range(dm)]
    filename = write_vtkhdf("linear_shallow_water_$(mesh_type)_k$(k).vtkhdf", dm; height=cell_height)
    println("wrote $filename")
end

main(ARGS)
