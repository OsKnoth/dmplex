# Steady rotating wind in thermal-wind balance (`thermal_wind`) with the
# nonhydrostatic TRiSK model or the 3D kite model on the extruded hexagonal mesh,
# or the ICON model on the extruded triangle mesh: a test of how well a model
# keeps a balanced, stratified, sheared flow at rest.
#
#   julia --project examples/thermal_wind.jl trisk [resolution] [days] [dt]
#   julia --project examples/thermal_wind.jl icon [resolution] [days] [dt]
#   julia --project examples/thermal_wind.jl kite [resolution] [days] [dt] [velocity] [theta] [mesh] [advection]
#
# velocity: rt (RT1 × Q2, default) or dual_radau (dual Radau 1 × Q1); theta: the
# horizontal_theta of KiteNonhydrostatic, dg_upwind (default), dg, upwind or central;
# mesh: the primal mesh of the kites, hexagon (default), cubed (cubed sphere with
# `resolution` cells per panel edge) or triangles; advection: the horizontal_advection of
# KiteNonhydrostatic, dg (default) or vector_invariant. 30 equidistant layers up to
# 30 km, Rayleigh damping of w above 22 km. Prints twice a day the largest |w|, the
# largest change of θ and of the surface pressure.
using Printf
using dmplex

const NH = dmplex.Nonhydrostatic

"Surface pressure (Pa) from the lowest layer, extrapolated hydrostatically."
function surface_pressure(m, y)
    lowest(x) = m isa KiteNonhydrostatic ? x[1, :] : x[:, 1]     # the kite arrays are layers first
    ρ, ρθ = lowest(Array(y.ρ)), lowest(Array(y.ρθ))
    π1 = (NH.Rd .* ρθ ./ NH.P0) .^ (NH.Rd / NH.Cvd)
    T1 = π1 .* ρθ ./ ρ
    return NH.P0 .* π1 .^ (NH.Cpd / NH.Rd) .* exp.(NH.GRAV * m.vgrid.z_full[1] ./ (NH.Rd .* T1))
end

function main(args)
    model = length(args) >= 1 ? args[1] : "trisk"
    resolution = length(args) >= 2 ? parse(Int, args[2]) : (model == "kite" ? 3 : 4)
    days = length(args) >= 3 ? parse(Float64, args[3]) : 5.0
    dt = length(args) >= 4 ? parse(Float64, args[4]) : (model == "kite" ? 150.0 : 300.0)
    v = VerticalGrid(30, 30000.0)
    if model == "trisk"
        m = MPASGassmannNonhydrostatic(TRiSKGrid(hilbert_order(create_hexagon_mesh(resolution))), v;
                                damping_height=22000.0, damping_rate=0.05)
    elseif model == "icon"
        m = ICONNonhydrostatic(TRiSKGrid(hilbert_order(create_spherical_mesh(resolution))), v;
                               damping_height=22000.0, damping_rate=0.05)
    else
        velocity = length(args) >= 5 ? Symbol(args[5]) : :rt
        theta = length(args) >= 6 ? Symbol(args[6]) : :dg_upwind
        mesh = length(args) >= 7 ? args[7] : "hexagon"
        advection = length(args) >= 8 ? Symbol(args[8]) : :dg
        primal = mesh == "hexagon" ? create_hexagon_mesh(resolution) :
                 mesh == "cubed" ? create_cubed_sphere_mesh(resolution) : create_spherical_mesh(resolution)
        m = KiteNonhydrostatic(primal, v, 1; height_order=velocity == :rt ? 2 : 1,
                               velocity, horizontal_theta=theta, horizontal_advection=advection,
                               damping_height=22000.0, damping_rate=0.05)
    end
    y = initial_state(m, (λ, φ, z) -> thermal_wind(λ, φ, z))
    ys = zero_state(m)
    θ0, ps0 = y.ρθ ./ y.ρ, surface_pressure(m, y)
    @printf("%s, %s %d, dt = %.0f s\n", model, model == "kite" ? (length(args) >= 7 ? args[7] : "hexagon") :
            model == "icon" ? "triangles" : "hexagon", resolution, dt)
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
