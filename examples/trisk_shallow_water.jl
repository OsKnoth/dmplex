# Nonlinear shallow water with C-grid finite volumes, Williamson test case 2:
# steady geostrophic solid-body rotation. TRiSK scheme on Voronoi meshes (as in
# MPAS and the hexagonal ICON), or with --icon the triangular C grid of ICON
# (RBF tangential velocity).
#
#   julia --project examples/trisk_shallow_water.jl [mesh] [resolution]
#   mpiexec -n 4 julia --project examples/trisk_shallow_water.jl hexagon 7
#   srun -n 4 julia --project=examples/gpu examples/trisk_shallow_water.jl hexagon 8 --gpu
#   julia --project examples/trisk_shallow_water.jl hexagon 6 --gassmann
#   julia --project examples/trisk_shallow_water.jl triangle 6 --icon
#
# mesh: hexagon (Voronoi dual of the icosahedral mesh), triangle (the
# icosahedral mesh, --icon), or an MPAS grid file (Voronoi cells) or an ICON grid
# file (its hexagonal dual, or its triangles with --icon). Units: a = Ω = g = 1.
# --gassmann uses the PV flux of Gassmann (2013, 2018) instead of TRiSK's (as in
# ICON-IAP), --gassmann-mean the same reconstruction with the mean edge PV.
using KernelAbstractions
using MPI
using StaticArrays
using dmplex

const USE_GPU = "--gpu" in ARGS
const USE_ICON = "--icon" in ARGS
const PV_FLUX = "--gassmann-mean" in ARGS ? :gassmann_mean : "--gassmann" in ARGS ? :gassmann : :trisk
if USE_GPU
    using CUDA
end

"""
Mesh and its mass points (`nothing`: computed by `TRiSKGrid`): Voronoi cells for
TRiSK, triangles with circumcenters for the ICON model (`--icon`).
"""
function load_mesh(mesh, resolution)
    mesh == "triangle" && return create_spherical_mesh(resolution), nothing
    if mesh == "hexagon"   # the generators are the vertices of the icosahedral mesh
        return create_hexagon_mesh(resolution), create_spherical_mesh(resolution).coordinates
    end
    isfile(mesh) || error("unknown mesh $mesh")
    icon_file = NCDataset(ds -> haskey(ds, "vertex_of_cell"), mesh)
    return read_grid(mesh; dual=icon_file && !USE_ICON), nothing   # TRiSK on ICON grids: the hexagonal dual
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
    mesh = length(args) >= 1 ? args[1] : USE_ICON ? "triangle" : "hexagon"
    resolution = length(args) >= 2 ? parse(Int, args[2]) : 5

    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    nranks, root = MPI.Comm_size(comm), MPI.Comm_rank(comm) == 0
    dm, centers = load_mesh(mesh, resolution)
    options = (; centers, comm=nranks > 1 ? comm : nothing, backend=select_backend(comm))
    model = USE_ICON ? ICONShallowWater(dm; options...) : TRiSKShallowWater(dm; pv_flux=PV_FLUX, options...)
    root && println("$(num_cells(dm)) cells, $(num_edges(dm)) edges, $(num_vertices(dm)) vertices, $nranks ranks on ",
                    nameof(typeof(model.backend)))

    u0 = 0.2
    velocity(x) = u0 * SVector(-x[2], x[1], 0.0)
    height(x) = 1 - (u0 + u0^2 / 2) * x[3]^2
    y = initial_state(model, velocity, height)
    h0 = gather_state(model, y)[global_height_dofs(model)]
    areas = model.grid.cell_areas

    cache = RK4Cache(y)
    dt = 0.2 * sqrt(4π / num_cells(dm))
    nsteps = ceil(Int, 1.0 / dt)
    mass0, energy0, enstrophy0 = total_mass(model, y), total_energy(model, y), potential_enstrophy(model, y)
    tendency!(similar(y), y, model)   # compile the kernels before timing
    elapsed = 0.0                     # time steps only, without the diagnostics
    for step in 1:nsteps
        MPI.Barrier(comm)
        elapsed += @elapsed rk4_step!(y, dt, tendency!, cache, model)
        if step % max(1, nsteps ÷ 4) == 0 || step == nsteps
            mass, energy, enstrophy = total_mass(model, y), total_energy(model, y), potential_enstrophy(model, y)
            dh = gather_state(model, y)[global_height_dofs(model)] - h0
            root && println("t = $(round(step * dt, digits=3))  |h - h0|₂ = $(sqrt(sum(areas .* dh .^ 2) / 4π))  ",
                    "max = $(maximum(abs, dh))  mass: $(mass - mass0)  energy: $((energy - energy0) / energy0)  ",
                    "enstrophy: $((enstrophy - enstrophy0) / enstrophy0)")
        end
    end
    elapsed = MPI.Allreduce(elapsed, max, comm)
    root && println("$(round(elapsed, digits=2)) s in $nsteps time steps, $(round(1000elapsed / nsteps, digits=2)) ms per step")
end

using NCDatasets: NCDataset
main(ARGS)
