# Galewsky et al. (2004) barotropically unstable mid-latitude jet with the
# nonlinear kite shallow-water model (RT1 × Q2 × CG2 by default).
#
#   julia --project examples/galewsky.jl [mesh] [resolution] [days] [perturbed]
#   mpiexec -n 6 julia --project examples/galewsky.jl hexagon 5 6
#
# mesh: hexagon | triangle | cubed, or an ICON/MPAS grid file (resolution is
# then ignored). Nondimensional units: a = Ω = 1, heights in units of 10 km.
using LinearAlgebra
using MPI
using Printf
using StaticArrays
using dmplex

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

function load_primal(mesh, resolution)
    mesh == "hexagon" && return create_hexagon_mesh(resolution)
    mesh == "triangle" && return create_spherical_mesh(resolution)
    mesh == "cubed" && return create_cubed_sphere_mesh(resolution)
    return read_grid(mesh)
end

function main(args)
    mesh = length(args) >= 1 ? args[1] : "hexagon"
    resolution = length(args) >= 2 ? parse(Int, args[2]) : 4
    days = length(args) >= 3 ? parse(Float64, args[3]) : 1.0
    perturbed = length(args) >= 4 ? parse(Bool, args[4]) : true

    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    nranks, root = MPI.Comm_size(comm), MPI.Comm_rank(comm) == 0
    primal = load_primal(mesh, resolution)
    model = NonlinearKiteShallowWater(primal, 1; gravity=G, coriolis=x -> 2x[3], comm=nranks > 1 ? comm : nothing)
    kite_size = sqrt(4π / num_cells(model.kites))
    root && @printf("%d primal cells, %d kites (≈ %.0f km), %d + %d dofs, %d ranks\n", num_cells(primal),
                    num_cells(model.kites), kite_size * EARTH_RADIUS / 1000, num_dofs(model.U), num_dofs(model.H),
                    nranks)

    hb = balanced_height()
    height(x) = hb(latitude(x)) + (perturbed ? perturbation(x) : 0.0)
    y = initial_state(model, velocity, height)
    y0 = copy(y)

    wave_speed = sqrt(G * 1.1) + U_MAX
    dt = 0.15 * kite_size / wave_speed
    nsteps = ceil(Int, days * DAY / dt)
    dt = days * DAY / nsteps
    root && @printf("dt = %.1f s, %d steps\n", dt / OMEGA, nsteps)

    cache = SSPRK3Cache(y)
    mass0, energy0 = total_mass(model, y), total_energy(model, y)
    elapsed = @elapsed for step in 1:nsteps
        ssprk3_step!(y, dt, tendency!, cache, model)
        if step % max(1, nsteps ÷ 10) == 0 || step == nsteps
            mass, energy = total_mass(model, y), total_energy(model, y)
            state = gather_state(model, y)
            dh = (state - y0)[height_dofs(model)]
            root && @printf("day %5.2f  |h - h0| = %.3e  mass %.1e  energy %.1e\n", step * dt / DAY,
                            sqrt(dot(dh, model.Mh * dh)), mass - mass0, (energy - energy0) / energy0)
        end
    end
    root && @printf("%.1f s wall time, %.1f ms per step\n", elapsed, 1000elapsed / nsteps)

    # relative vorticity at kite centers (1/s) for ParaView, from the gathered state
    y = gather_state(model, y)
    ζ = mass_solver(model.Z; lumped=true) \ (-(model.R * y[velocity_dofs(model)]))
    if root
        center = SVector(0.0, 0.0)
        vorticity = [evaluate(model.Z, ζ, c, center)[2] * OMEGA for c in cell_range(model.kites)]
        filename = write_vtkhdf("galewsky_$(basename(mesh))_$(resolution).vtkhdf", model.kites; vorticity)
        println("wrote $filename")
    end
    return model, y
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
