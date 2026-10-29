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
# file (its hexagonal dual, or its triangles with --icon); with --icon an MPAS file gives its
# Delaunay triangles (e.g. the locally refined x4.163842). Units: a = Ω = g = 1.
# --gassmann uses the PV flux of Gassmann (2013, 2018) instead of TRiSK's (as in
# ICON-IAP), --gassmann-mean the same reconstruction with the mean edge PV.
# --spring[=β] optimizes the icosahedral mesh by spring dynamics (β = 1.2), --scvt
# makes it centroidal (Lloyd's algorithm: the hexagons are an SCVT, as MPAS grids); with --icon and a
# grid file --spring[=β] cleans up its triangles (`relax_triangles`, β = 1), --centered-k takes the
# centered kinetic-energy gradient instead of ICON's `coeff_gradekin`. --kinetic=cell|mixed|vertex|α sets
# TRiSK's kinetic energy (`kinetic_energy`: the cell one, Skamarock's mixture α = 3/8, the consistent
# vertex one).
using KernelAbstractions
using MPI
using StaticArrays
using dmplex

const USE_GPU = "--gpu" in ARGS
const USE_ICON = "--icon" in ARGS
const PV_FLUX = "--gassmann-mean" in ARGS ? :gassmann_mean : "--gassmann" in ARGS ? :gassmann : :trisk
const SPRING = filter(a -> startswith(a, "--spring"), ARGS)
const SPRING_BETA = isempty(SPRING) ? nothing : SPRING[1] == "--spring" ? 1.2 : parse(Float64, split(SPRING[1], "=")[2])
const CENTROIDAL = "--scvt" in ARGS
const KINETIC_GRADIENT = "--centered-k" in ARGS ? :centered : :icon
const KINETIC = filter(a -> startswith(a, "--kinetic="), ARGS)
const KINETIC_ENERGY = isempty(KINETIC) ? :cell :
                       (k = split(KINETIC[1], "=")[2]; k in ("cell", "mixed", "vertex") ? Symbol(k) : parse(Float64, k))
if USE_GPU
    using CUDA
end

"""
Mesh and its mass points (`nothing`: computed by `TRiSKGrid`): Voronoi cells for
TRiSK, triangles with circumcenters for the ICON model (`--icon`).
"""
function load_mesh(mesh, resolution)
    optimization = (; spring_beta=SPRING_BETA, centroidal=CENTROIDAL)
    mesh == "triangle" && return create_spherical_mesh(resolution; optimization...), nothing
    if mesh == "hexagon"   # the generators are the vertices of the icosahedral mesh
        return (create_hexagon_mesh(resolution; optimization...),
                create_spherical_mesh(resolution; optimization...).coordinates)
    end
    isfile(mesh) || error("unknown mesh $mesh")
    icon_file = NCDataset(ds -> haskey(ds, "vertex_of_cell"), mesh)
    # TRiSK on ICON grids: the hexagonal dual; the ICON model on MPAS grids: the Delaunay triangles
    dm = read_grid(mesh; dual=icon_file ? !USE_ICON : USE_ICON)
    # --spring on triangles from a file: clean up by spring dynamics with the local resolution (β = 1 by default)
    USE_ICON && !isempty(SPRING) && (dm = relax_triangles(dm; beta=SPRING[1] == "--spring" ? 1.0 : SPRING_BETA))
    return dm, nothing
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
    model = USE_ICON ? ICONShallowWater(dm; kinetic_gradient=KINETIC_GRADIENT, options...) : TRiSKShallowWater(dm; pv_flux=PV_FLUX, kinetic_energy=KINETIC_ENERGY, options...)
    root && println("$(num_cells(dm)) cells, $(num_edges(dm)) edges, $(num_vertices(dm)) vertices, $nranks ranks on ",
                    nameof(typeof(model.backend)))

    u0 = 0.2
    velocity(x) = u0 * SVector(-x[2], x[1], 0.0)
    height(x) = 1 - (u0 + u0^2 / 2) * x[3]^2
    y = initial_state(model, velocity, height)
    h0 = gather_state(model, y)[global_height_dofs(model)]
    areas = model.grid.cell_areas

    cache = RK4Cache(y)
    dt = min(0.2 * sqrt(4π / num_cells(dm)), 0.4 * minimum(model.grid.dual_lengths))   # locally refined meshes
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
