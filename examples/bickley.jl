# Bickley jet on a doubly periodic plane with the nonlinear kite shallow-water
# model (RT1 × Q2 × CG2), without rotation and at Froude number 0.2:
#
#   julia --project examples/bickley.jl [quadrilateral|triangle] [n] [T]
#
# Domain [0, 4π)², u = sech²(y - 2π), constant depth 1, gravity wave speed 5,
# perturbed by a stream function in x-wavenumber 1 (close to the most unstable
# wavenumber of the jet), so the jet rolls up into a vortex street.
using LinearAlgebra
using Printf
using StaticArrays
using dmplex

const L = 4π
const WAVE_SPEED = 5.0
const EPSILON = 0.05

jet(y) = sech(y - L / 2)^2
# perturbation stream function ψ' = ε exp(-(y - y₀)²) cos(x), u' = ∂ψ'/∂y, v' = -∂ψ'/∂x
function velocity(x)
    y = x[2] - L / 2
    envelope = EPSILON * exp(-y^2)
    return SVector(jet(x[2]) - 2y * envelope * cos(x[1]), envelope * sin(x[1]), 0.0)
end

function main(args)
    cells = length(args) >= 1 ? Symbol(args[1]) : :quadrilateral
    n = length(args) >= 2 ? parse(Int, args[2]) : 32
    T = length(args) >= 3 ? parse(Float64, args[3]) : 40.0

    primal = create_periodic_plane(n, n; Lx=L, Ly=L, cells)
    model = NonlinearKiteShallowWater(primal, 1; gravity=WAVE_SPEED^2, coriolis=x -> 0.0)
    kite_size = L / sqrt(num_cells(model.kites))
    @printf("%d %s cells, %d kites, %d + %d dofs\n", num_cells(primal), cells, num_cells(model.kites),
            num_dofs(model.U), num_dofs(model.H))

    y = initial_state(model, velocity, x -> 1.0)
    dt = 0.15 * kite_size / (WAVE_SPEED + 1)
    nsteps = ceil(Int, T / dt)
    dt = T / nsteps
    cache = SSPRK3Cache(y)
    mass0, energy0 = total_mass(model, y), total_energy(model, y)
    elapsed = @elapsed for step in 1:nsteps
        ssprk3_step!(y, dt, tendency!, cache, model)
        if step % max(1, nsteps ÷ 8) == 0 || step == nsteps
            @printf("t = %6.2f  mass %.1e  energy %.1e\n", step * dt, total_mass(model, y) - mass0,
                    (total_energy(model, y) - energy0) / energy0)
        end
    end
    @printf("%d steps, %.1f s, %.1f ms per step\n", nsteps, elapsed, 1000elapsed / nsteps)
    return model, y
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
