# Nonlinear rotating shallow water (vector-invariant, RT_k × DG_k × CG_{k+1}),
# Williamson test case 2: steady geostrophic solid-body rotation.
#
#   julia --project examples/nonlinear_shallow_water.jl [order] [mesh] [resolution]
#   mpiexec -n 4 julia --project examples/nonlinear_shallow_water.jl 1 cubed 16
#   mpiexec -n 4 julia --project=examples/gpu examples/nonlinear_shallow_water.jl 1 cubed 64 --gpu
#
# mesh: icosahedron | cubed | healpix, or the path of an ICON grid file or an
# MPAS grid file (the MPAS Delaunay dual is used, since the elements need
# triangles or quadrilaterals). Units: a = Ω = g = 1.
# --gpu runs the model on CUDA GPUs (one per rank) with the coupled RT and CG
# mass matrices solved by cuDSS; it needs the environment examples/gpu (see
# docs/levante.md).
using KernelAbstractions
using LinearAlgebra
using MPI
using StaticArrays
using dmplex

const USE_GPU = "--gpu" in ARGS
if USE_GPU
    using CUDA
    using CUDSS
end

function load_mesh(mesh, resolution)
    mesh == "icosahedron" && return create_spherical_mesh(resolution)
    mesh == "cubed" && return create_cubed_sphere_mesh(resolution)
    mesh == "healpix" && return create_healpix_mesh(resolution)
    isfile(mesh) || error("unknown mesh $mesh")
    return read_grid(mesh; dual=!NCDataset(ds -> haskey(ds, "vertex_of_cell"), mesh))
end

"Backend of the model: the node-local GPU of this rank, or the CPU."
function select_backend(comm)
    USE_GPU || return KernelAbstractions.CPU()
    local_rank = MPI.Comm_rank(MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm)))
    CUDA.device!(local_rank % length(CUDA.devices()))
    return CUDA.CUDABackend()
end

function main(args)
    args = filter(a -> !startswith(a, "--"), args)
    k = length(args) >= 1 ? parse(Int, args[1]) : 1
    mesh = length(args) >= 2 ? args[2] : "cubed"
    resolution = length(args) >= 3 ? parse(Int, args[3]) : 6

    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    nranks, root = MPI.Comm_size(comm), MPI.Comm_rank(comm) == 0
    dm = load_mesh(mesh, resolution)
    model = NonlinearShallowWater(dm, k; comm=nranks > 1 ? comm : nothing, backend=select_backend(comm),
                                  solver=USE_GPU ? :cudss : :gathered)
    root && println("$(num_cells(dm)) cells, RT$k × DG$k × CG$(k + 1): ",
                    "$(num_dofs(model.U)) + $(num_dofs(model.H)) dofs, $(num_dofs(model.Q)) PV dofs, $nranks ranks ",
                    "on $(nameof(typeof(model.backend)))")

    u0 = 0.2
    velocity(x) = u0 * SVector(-x[2], x[1], 0.0)
    height(x) = 1 - (u0 + u0^2 / 2) * x[3]^2
    y = initial_state(model, velocity, height)
    y0 = gather_state(model, y)

    cache = RK4Cache(y)
    dt = 0.1 * sqrt(4π / num_cells(dm)) / (k + 1)^2
    nsteps = ceil(Int, 1.0 / dt)
    mass0, energy0, enstrophy0 = total_mass(model, y), total_energy(model, y), potential_enstrophy(model, y)
    elapsed = @elapsed for step in 1:nsteps
        rk4_step!(y, dt, tendency!, cache, model)
        if step % max(1, nsteps ÷ 4) == 0 || step == nsteps
            mass, energy, enstrophy = total_mass(model, y), total_energy(model, y), potential_enstrophy(model, y)
            dh = (gather_state(model, y) - y0)[global_height_dofs(model)]
            root && println("t = $(round(step * dt, digits=3))  |h - h0| = $(sqrt(dot(dh, model.Mh * dh)))  ",
                    "mass: $(mass - mass0)  ",
                    "energy: $((energy - energy0) / energy0)  ",
                    "enstrophy: $((enstrophy - enstrophy0) / enstrophy0)")
        end
    end
    root && println("$(round(elapsed, digits=1)) s wall time")
end

using NCDatasets: NCDataset
main(ARGS)
