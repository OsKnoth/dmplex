# Steady rotating wind in thermal-wind balance (`thermal_wind`) with the
# nonhydrostatic TRiSK model or the 3D kite model on the extruded hexagonal mesh:
# a test of how well a model keeps a balanced, stratified, sheared flow at rest.
#
#   julia --project examples/thermal_wind.jl trisk [resolution] [days] [dt]
#   julia --project examples/thermal_wind.jl kite [resolution] [days] [dt] [velocity] [theta]
#
# velocity: rt (RT1 × Q2, default) or dual_radau (dual Radau 1 × Q1); theta: upwind
# (default) or central trace of θ on the primal faces. 30 equidistant layers up to
# 30 km, Rayleigh damping of w above 22 km. Prints twice a day the largest |w|, the
# largest change of θ and of the surface pressure.
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

function main(args)
    model = length(args) >= 1 ? args[1] : "trisk"
    resolution = length(args) >= 2 ? parse(Int, args[2]) : (model == "trisk" ? 4 : 3)
    days = length(args) >= 3 ? parse(Float64, args[3]) : 5.0
    dt = length(args) >= 4 ? parse(Float64, args[4]) : (model == "trisk" ? 300.0 : 150.0)
    v = VerticalGrid(30, 30000.0)
    if model == "trisk"
        m = TRiSKNonhydrostatic(TRiSKGrid(hilbert_order(create_hexagon_mesh(resolution))), v;
                                damping_height=22000.0, damping_rate=0.05)
    else
        velocity = length(args) >= 5 ? Symbol(args[5]) : :rt
        theta = length(args) >= 6 ? Symbol(args[6]) : :upwind
        m = KiteNonhydrostatic(create_hexagon_mesh(resolution), v, 1; height_order=velocity == :rt ? 2 : 1,
                               velocity, horizontal_theta=theta, damping_height=22000.0, damping_rate=0.05)
    end
    y = initial_state(m, (λ, φ, z) -> thermal_wind(λ, φ, z))
    ys = zero_state(m)
    θ0, ps0 = y.ρθ ./ y.ρ, surface_pressure(m, y)
    @printf("%s, hexagon %d, dt = %.0f s\n", model, resolution, dt)
    for n in 1:round(Int, days * 86400 / dt)
        step!(m, y, dt, ys)
        if n % round(Int, 43200 / dt) == 0
            @printf("day %4.1f  max|w| %.2e m/s  max|θ - θ0| %.2e K  max|Δp_s| %.2e hPa\n", n * dt / 86400,
                    maximum(abs, y.w), maximum(abs, y.ρθ ./ y.ρ .- θ0),
                    maximum(abs, surface_pressure(m, y) .- ps0) / 100)
        end
    end
    return m, y
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
