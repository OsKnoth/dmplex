# Baroclinic wave (Ullrich et al. 2014) with the kite model, RT1 × Q2, DG transport; prints twice a day
# and saves p_s and T of the lowest layer at days 6, 8, 10 to data/:
#
#   julia --project baroclinic_wave_run.jl hexagon|cubed|triangles resolution days dt [advection]
using dmplex, Printf, LinearAlgebra, Serialization
const NH = dmplex.Nonhydrostatic
mesh, res, days, dt = ARGS[1], parse(Int, ARGS[2]), parse(Float64, ARGS[3]), parse(Float64, ARGS[4])
adv = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :dg
primal = mesh == "hexagon" ? create_hexagon_mesh(res) : mesh == "cubed" ? create_cubed_sphere_mesh(res) : create_spherical_mesh(res)
m = KiteNonhydrostatic(primal, VerticalGrid(30, 30000.0), 1; height_order=2, velocity=:rt, damping_height=22000.0,
                       damping_rate=0.05, horizontal_theta=:dg_upwind, horizontal_advection=adv)
y = initial_state(m, (λ, φ, z) -> baroclinic_wave(λ, φ, z)); ys = zero_state(m)
W = column_weights(m); mass(y) = sum((y.ρ * W) .* m.vgrid.dz)
function ps(y)
    π1 = (NH.Rd .* y.ρθ[1, :] ./ NH.P0) .^ (NH.Rd / NH.Cvd); T1 = π1 .* y.ρθ[1, :] ./ y.ρ[1, :]
    NH.P0 .* π1 .^ (NH.Cpd / NH.Rd) .* exp.(NH.GRAV * m.vgrid.z_full[1] ./ (NH.Rd .* T1))
end
m0 = mass(y)
@printf("%s %d, advection %s: %d columns × 30 layers, dt %.0f\n", mesh, res, adv, num_columns(m), dt)
nday, n = round(Int, 86400 / dt), round(Int, days * 86400 / dt); t = 0.0
for i in 1:n
    global t += @elapsed step!(m, y, dt, ys)
    if i % (nday ÷ 2) == 0
        @printf("day %5.2f  p_s min %7.2f max %7.2f hPa  max|w| %.2e  mass %.1e  %.2f s/step\n", i * dt / 86400,
                minimum(ps(y)) / 100, maximum(ps(y)) / 100, maximum(abs, y.w), mass(y) / m0 - 1, t / i)
        flush(stdout)
        all(isfinite, y.ρ) || break
        d = i * dt / 86400
        if isapprox(d, round(d)) && round(Int, d) in (6, 8, 10)
            ll = [rad2deg.(NH.lonlat(x)) for x in column_points(m)]
            π1 = (NH.Rd .* y.ρθ[1, :] ./ NH.P0) .^ (NH.Rd / NH.Cvd)
            serialize(joinpath(@__DIR__, "data", "bw_$(mesh)$(res)_$(adv)_day$(round(Int, d)).jls"),
                      (lon=first.(ll), lat=last.(ll), ps=ps(y), T=π1 .* y.ρθ[1, :] ./ y.ρ[1, :]))
        end
    end
end
