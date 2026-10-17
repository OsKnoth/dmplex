# Nonlinear rotating shallow water (vector-invariant, RT_k × DG_k × CG_{k+1}),
# Williamson test case 2: steady geostrophic solid-body rotation.
#
#   julia --project examples/nonlinear_shallow_water.jl [order] [mesh] [resolution]
#
# mesh: icosahedron | cubed | healpix, or the path of an ICON grid file or an
# MPAS grid file (the MPAS Delaunay dual is used, since the elements need
# triangles or quadrilaterals). Units: a = Ω = g = 1.
using LinearAlgebra
using StaticArrays
using dmplex

function load_mesh(mesh, resolution)
    mesh == "icosahedron" && return create_spherical_mesh(resolution)
    mesh == "cubed" && return create_cubed_sphere_mesh(resolution)
    mesh == "healpix" && return create_healpix_mesh(resolution)
    isfile(mesh) || error("unknown mesh $mesh")
    return read_grid(mesh; dual=!NCDataset(ds -> haskey(ds, "vertex_of_cell"), mesh))
end

function main(args)
    k = length(args) >= 1 ? parse(Int, args[1]) : 1
    mesh = length(args) >= 2 ? args[2] : "cubed"
    resolution = length(args) >= 3 ? parse(Int, args[3]) : 6

    dm = load_mesh(mesh, resolution)
    model = NonlinearShallowWater(dm, k)
    println("$(num_cells(dm)) cells, RT$k × DG$k × CG$(k + 1): ",
            "$(num_dofs(model.U)) + $(num_dofs(model.H)) dofs, $(num_dofs(model.Q)) PV dofs")

    u0 = 0.2
    velocity(x) = u0 * SVector(-x[2], x[1], 0.0)
    height(x) = 1 - (u0 + u0^2 / 2) * x[3]^2
    y = initial_state(model, velocity, height)
    y0 = copy(y)

    cache = RK4Cache(y)
    dt = 0.1 * sqrt(4π / num_cells(dm)) / (k + 1)^2
    nsteps = ceil(Int, 1.0 / dt)
    mass0, energy0, enstrophy0 = total_mass(model, y), total_energy(model, y), potential_enstrophy(model, y)
    for step in 1:nsteps
        rk4_step!(y, dt, tendency!, cache, model)
        if step % max(1, nsteps ÷ 4) == 0 || step == nsteps
            dh = (y - y0)[height_dofs(model)]
            println("t = $(round(step * dt, digits=3))  |h - h0| = $(sqrt(dot(dh, model.Mh * dh)))  ",
                    "mass: $(total_mass(model, y) - mass0)  ",
                    "energy: $((total_energy(model, y) - energy0) / energy0)  ",
                    "enstrophy: $((potential_enstrophy(model, y) - enstrophy0) / enstrophy0)")
        end
    end
end

using NCDatasets: NCDataset
main(ARGS)
