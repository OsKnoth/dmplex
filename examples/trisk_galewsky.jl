# Galewsky et al. (2004) barotropically unstable mid-latitude jet with the C-grid
# finite-volume shallow-water models: TRiSK on Voronoi meshes (as in MPAS and the
# hexagonal ICON), or with --icon the triangular C grid of ICON.
#
#   julia --project examples/trisk_galewsky.jl [mesh] [resolution] [days] [perturbed] [--gassmann[-mean]]
#   mpiexec -n 4 julia --project examples/trisk_galewsky.jl hexagon 7 6 --gassmann-mean
#   srun -n 4 julia --project=examples/gpu examples/trisk_galewsky.jl hexagon 8 6 --gpu
#   mpiexec -n 4 julia --project examples/trisk_galewsky.jl triangle 7 6 --icon
#
# mesh: hexagon (Voronoi dual of the icosahedral mesh), triangle (the
# icosahedral mesh, --icon), or an MPAS grid file (Voronoi cells) or an ICON
# grid file (its hexagonal dual, or its triangles with --icon). Nondimensional
# units: a = Ω = 1, heights in units of 10 km. The ICON model runs with ICON's
# fourth-order divergence damping (divdamp_fac = 0.0025). --gassmann uses the PV flux of
# Gassmann (2013, 2018) instead of TRiSK's (as in ICON-IAP), --gassmann-mean
# the same reconstruction with the mean edge PV; --spring[=β] optimizes the
# icosahedral mesh by spring dynamics (β = 1.2); --gpu runs on CUDA GPUs (one
# per rank, environment examples/gpu). No dissipation: the energy- and
# mass-conserving schemes run inviscid (--gassmann then develops grid-scale
# vorticity after day 3). Writes the relative vorticity (1/s) at the cells to
# trisk_galewsky_<mesh>_<scheme>.vtkhdf.
using KernelAbstractions
using LinearAlgebra
using MPI
using Printf
using StaticArrays
using dmplex

const USE_GPU = "--gpu" in ARGS
const USE_ICON = "--icon" in ARGS
const PV_FLUX = "--gassmann-mean" in ARGS ? :gassmann_mean : "--gassmann" in ARGS ? :gassmann : :trisk
const SCHEME = USE_ICON ? :icon : PV_FLUX
const SPRING = filter(a -> startswith(a, "--spring"), ARGS)
const SPRING_BETA = isempty(SPRING) ? nothing : SPRING[1] == "--spring" ? 1.2 : parse(Float64, split(SPRING[1], "=")[2])
if USE_GPU
    using CUDA
end

const EARTH_RADIUS = 6.37122e6
const OMEGA = 7.292e-5
const GRAVITY = 9.80616
const MEAN_DEPTH = 1.0e4
const G = GRAVITY * MEAN_DEPTH / (EARTH_RADIUS * OMEGA)^2   # nondimensional gravity
const DAY = 86400 * OMEGA                                   # one day in units of 1/Ω

const U_MAX = 80 / (EARTH_RADIUS * OMEGA)
const PHI0, PHI1 = π / 7, π / 2 - π / 7
const EN = exp(-4 / (PHI1 - PHI0)^2)

jet(φ) = PHI0 < φ < PHI1 ? U_MAX / EN * exp(1 / ((φ - PHI0) * (φ - PHI1))) : 0.0
latitude(x) = asin(clamp(x[3], -1, 1))
longitude(x) = atan(x[2], x[1])

"Zonal jet as a Cartesian vector: u(φ) times the eastward unit vector."
velocity(x) = (r = hypot(x[1], x[2]); r < 1e-14 ? zero(SVector{3, Float64}) :
               jet(latitude(x)) * SVector(-x[2] / r, x[1] / r, 0.0))

"Balanced height g h(φ) = g h₀ - ∫ u (f + tan φ' u) dφ', with mean depth 1."
function balanced_height()
    φ = range(-π / 2, π / 2; length=40001)
    integrand = [jet(p) * (2sin(p) + tan(p) * jet(p)) for p in φ]
    integrand[[1, end]] .= 0
    cumulative = [0.0; cumsum((integrand[1:end-1] .+ integrand[2:end]) ./ 2 .* step(φ))]
    profile = -cumulative ./ G
    mean = sum(((profile[1:end-1] .+ profile[2:end]) ./ 2) .* (cos.(φ[1:end-1]) .+ cos.(φ[2:end])) ./ 2 .* step(φ)) / 2
    h0 = 1 - mean
    return p -> begin
        s = (p + π / 2) / step(φ) + 1
        i = clamp(floor(Int, s), 1, length(φ) - 1)
        h0 + profile[i] + (s - i) * (profile[i + 1] - profile[i])
    end
end

"Height perturbation of 120 m at φ₂ = π/4, λ = 0 (α = 1/3, β = 1/15)."
function perturbation(x)
    λ, φ = longitude(x), latitude(x)
    return 120 / MEAN_DEPTH * cos(φ) * exp(-(3λ)^2) * exp(-(15 * (π / 4 - φ))^2)
end

"""
Mesh and its mass points (`nothing`: computed by `TRiSKGrid`): Voronoi cells for
TRiSK, triangles with circumcenters for the ICON model (`--icon`).
"""
function load_mesh(mesh, resolution)
    mesh == "triangle" && return create_spherical_mesh(resolution; spring_beta=SPRING_BETA), nothing
    if mesh == "hexagon"   # the generators are the vertices of the icosahedral mesh
        return (create_hexagon_mesh(resolution; spring_beta=SPRING_BETA),
                create_spherical_mesh(resolution; spring_beta=SPRING_BETA).coordinates)
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

"Relative vorticity at the cells from the global velocities: kite average of the vertex vorticities."
function cell_vorticity(g::TRiSKGrid, u)
    ζ = [sum(g.vertex_signs[k, v] * g.dual_lengths[g.vertex_edges[k, v]] * u[g.vertex_edges[k, v]]
             for k in axes(g.vertex_edges, 1)) / g.vertex_areas[v] for v in 1:num_vertices(g)]
    return [sum(g.cell_kites[k, i] * ζ[g.cell_vertices[k, i]] for k in axes(g.cell_vertices, 1)) / g.cell_areas[i]
            for i in 1:num_cells(g)]
end

function main(args)
    args = filter(a -> !startswith(a, "--"), args)
    mesh = length(args) >= 1 ? args[1] : USE_ICON ? "triangle" : "hexagon"
    resolution = length(args) >= 2 ? parse(Int, args[2]) : 6
    days = length(args) >= 3 ? parse(Float64, args[3]) : 6.0
    perturbed = length(args) >= 4 ? parse(Bool, args[4]) : true

    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    nranks, root = MPI.Comm_size(comm), MPI.Comm_rank(comm) == 0
    dm, centers = load_mesh(mesh, resolution)
    grid = TRiSKGrid(dm; centers)
    dt = 0.5 * minimum(grid.dual_lengths) / (sqrt(G * 1.1) + U_MAX)
    nsteps = ceil(Int, days * DAY / dt)
    dt = days * DAY / nsteps
    options = (; gravity=G, comm=nranks > 1 ? comm : nothing, backend=select_backend(comm))
    model = if USE_ICON   # ICON's fourth-order divergence damping, divdamp_fac = 0.0025 per time step
        ICONShallowWater(grid; divergence_damping=0.0025 * (4π / num_cells(dm))^2 / dt, options...)
    else
        TRiSKShallowWater(grid; pv_flux=PV_FLUX, options...)
    end
    root && @printf("%d cells (≈ %.0f km), %s scheme, %d ranks on %s\n", num_cells(dm),
                    sqrt(4π / num_cells(dm)) * EARTH_RADIUS / 1000, SCHEME, nranks, nameof(typeof(model.backend)))

    hb = balanced_height()
    height(x) = hb(latitude(x)) + (perturbed ? perturbation(x) : 0.0)
    y = initial_state(model, velocity, height)
    h0 = gather_state(model, y)[global_height_dofs(model)]
    areas = model.grid.cell_areas

    root && @printf("dt = %.1f s, %d RK4 steps\n", dt / OMEGA, nsteps)

    cache = RK4Cache(y)
    mass0, energy0, enstrophy0 = total_mass(model, y), total_energy(model, y), potential_enstrophy(model, y)
    tendency!(similar(y), y, model)   # compile the kernels before timing
    elapsed = 0.0
    reports = Set(round.(Int, range(0, nsteps; length=round(Int, 2days) + 1)[2:end]))   # every half day
    for step in 1:nsteps
        elapsed += @elapsed rk4_step!(y, dt, tendency!, cache, model)
        if step in reports
            mass, energy, enstrophy = total_mass(model, y), total_energy(model, y), potential_enstrophy(model, y)
            dh = gather_state(model, y)[global_height_dofs(model)] - h0
            root && @printf("day %5.2f  |h - h0|₂ = %.3e  mass %.1e  energy %.1e  enstrophy %.1e\n", step * dt / DAY,
                            sqrt(sum(areas .* dh .^ 2) / 4π), (mass - mass0) / mass0, (energy - energy0) / energy0,
                            (enstrophy - enstrophy0) / enstrophy0)
        end
    end
    elapsed = MPI.Allreduce(elapsed, max, comm)
    root && @printf("%.1f s in %d time steps, %.2f ms per step\n", elapsed, nsteps, 1000elapsed / nsteps)

    state = gather_state(model, y)
    if root
        vorticity = OMEGA .* cell_vorticity(model.grid, state[global_velocity_dofs(model)])
        @printf("relative vorticity: min %.2e, max %.2e 1/s\n", minimum(vorticity), maximum(vorticity))
        name = isfile(mesh) ? splitext(basename(mesh))[1] :
               "$(mesh)_$(resolution)" * (SPRING_BETA === nothing ? "" : "_spring$(SPRING_BETA)")
        filename = write_vtkhdf("trisk_galewsky_$(name)_$SCHEME.vtkhdf", dm; vorticity)
        println("wrote $filename")
    end
    return model, state
end

using NCDatasets: NCDataset
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
