# Baroclinic wave in a Cartesian f-plane channel (Ullrich, Reed and Jablonowski 2015,
# `baroclinic_channel`) with the 3D kite model on a doubly periodic box: 40000 km in x and
# twice the channel width of 6000 km in y, the channel and its mirror image with the easterly
# jet, so that v = 0 at the walls y = 0 and 6000 km by symmetry. Flat ground, lid at 30 km,
# f0 = 1.0313e-4 1/s. Only the kite model runs on planar meshes (TRiSK and ICON need the
# spherical grids).
#
#   julia --project examples/baroclinic_channel.jl [nx] [days] [dt] [cells] [theta] [perturbation] [advection]
#
# nx: cells along x (default 80, 500 km; ny = 3nx/10); cells: quadrilateral (default) or
# triangle; theta: horizontal_theta of the kite model (dg_upwind by default, see
# KiteNonhydrostatic); perturbation: true (default) or false for the steady state; advection:
# horizontal_advection of KiteNonhydrostatic, dg (default) or vector_invariant.
# 30 equidistant layers, Rayleigh damping of w above 22 km. Prints twice a day the
# minimum and maximum surface pressure, the largest |w|, the amplitude of zonal wavenumber 10 of
# the surface pressure in the channel and its growth rate over the last half day (linear QG theory:
# 6.59e-6 1/s for k = 10, the most unstable wave of 4000 km), and the relative mass change.
using Printf
using dmplex

const NH = dmplex.Nonhydrostatic

"Surface pressure (Pa) from the lowest layer, extrapolated hydrostatically."
function surface_pressure(m, y)
    ρ, ρθ = Array(y.ρ), Array(y.ρθ)
    π1 = (NH.Rd .* ρθ[:, 1] ./ NH.P0) .^ (NH.Rd / NH.Cvd)
    T1 = π1 .* ρθ[:, 1] ./ ρ[:, 1]
    return NH.P0 .* π1 .^ (NH.Cpd / NH.Rd) .* exp.(NH.GRAV * m.vgrid.z_full[1] ./ (NH.Rd .* T1))
end

"""
Amplitude (Pa) of the zonal wavenumber `k` of the surface pressure in the channel `0 < y < Ly`
(the mirror half carries the mirror image), area weighted over the columns at `x`.
"""
function wave_amplitude(ps, x, W, Lx, Ly, k)
    w = W .* [p[2] < Ly for p in x]
    p̄ = sum(w .* ps) / sum(w)
    return 2 * abs(sum(w .* (ps .- p̄) .* cis.(-2π * k .* first.(x) ./ Lx))) / sum(w)
end

function main(args)
    nx = length(args) >= 1 ? parse(Int, args[1]) : 80
    days = length(args) >= 2 ? parse(Float64, args[2]) : 12.0
    dt = length(args) >= 3 ? parse(Float64, args[3]) : 150.0 * 80 / nx
    cells = length(args) >= 4 ? Symbol(args[4]) : :quadrilateral
    theta = length(args) >= 5 ? Symbol(args[5]) : :dg_upwind
    perturbation = length(args) >= 6 ? parse(Bool, args[6]) : true
    advection = length(args) >= 7 ? Symbol(args[7]) : :dg
    Lx, Ly, f0 = 4e7, 6e6, 1.0313e-4
    ny = 3nx ÷ 10
    primal = create_periodic_plane(nx, ny; Lx, Ly=2Ly, cells)
    m = KiteNonhydrostatic(primal, VerticalGrid(30, 30000.0), 1; radius=1.0, coriolis=x -> f0,
                           horizontal_theta=theta, horizontal_advection=advection,
                           damping_height=22000.0, damping_rate=0.05)
    y = initial_state(m, (x, yy, z) -> baroclinic_channel(x, yy, z; Ly, f0, perturbation))
    ys = zero_state(m)
    W = column_weights(m)
    mass(y) = sum(W' * y.ρ .* m.vgrid.dz')
    m0 = mass(y)
    x = column_points(m)
    A0 = wave_amplitude(surface_pressure(m, y), x, W, Lx, Ly, 10)
    @printf("channel %d × %d %ss (%.0f km), θ %s, advection %s, %d columns × 30 layers, dt = %.0f s\n", nx, ny,
            cells, Lx / nx / 1e3, theta, advection, num_columns(m), dt)
    t = 0.0
    for n in 1:round(Int, days * 86400 / dt)
        t += @elapsed step!(m, y, dt, ys)
        if n % round(Int, 43200 / dt) == 0
            ps = surface_pressure(m, y)
            A = wave_amplitude(ps, x, W, Lx, Ly, 10)
            @printf("day %5.2f  p_s min %7.2f max %7.2f hPa  max|w| %.2e m/s  k=10: %.2e Pa (growth %.2e 1/s)  mass %.1e  %.2f s/step\n",
                    n * dt / 86400, minimum(ps) / 100, maximum(ps) / 100, maximum(abs, y.w), A, log(A / A0) / 43200,
                    mass(y) / m0 - 1, t / n)
            A0 = A
            flush(stdout)
            all(isfinite, y.ρ) || break
        end
    end
    return m, y
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
