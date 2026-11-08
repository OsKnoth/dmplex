# The 3D test cases of the paper with every pair of kite spaces: thermal wind (tw), baroclinic wave on
# the sphere (bw) or baroclinic channel (ch), 30 layers to 30 km, DG transport with upwind traces;
# prints the diagnostics twice a day (four times for ch) and saves the series and snapshots to data/:
#
#   julia --project kite_case.jl tw|bw|ch space mesh resolution days dt [vertical_theta]
#
# space: rt1q1, rt1q2, bdm1q1, bdm1q2, wess1, wess2 (KiteDualRadau(k) × Q_k); mesh: hexagon, cubed or
# triangles on the sphere, quadrilateral or triangle cells of the channel (resolution = cells along x);
# vertical_theta: upwind (default) or central.
using dmplex, Printf, Serialization
include(joinpath(@__DIR__, "..", "..", "..", "examples", "baroclinic_channel.jl"))

const SPACES = Dict("rt1q1" => (:rt, 1, 1), "rt1q2" => (:rt, 1, 2), "bdm1q1" => (:bdm, 1, 1),
                    "bdm1q2" => (:bdm, 1, 2), "wess1" => (:dual_radau, 1, 1), "wess2" => (:dual_radau, 2, 2))

case, space, mesh = ARGS[1], ARGS[2], ARGS[3]
res, days, dt = parse(Int, ARGS[4]), parse(Float64, ARGS[5]), parse(Float64, ARGS[6])
vt = length(ARGS) >= 7 ? Symbol(ARGS[7]) : :upwind
velocity, k, p = SPACES[space]
Lx, Ly, f0 = 4e7, 6e6, 1.0313e-4
primal = case == "ch" ? create_periodic_plane(res, 3res ÷ 10; Lx, Ly=2Ly, cells=Symbol(mesh)) :
         mesh == "hexagon" ? create_hexagon_mesh(res) : mesh == "cubed" ? create_cubed_sphere_mesh(res) :
         create_spherical_mesh(res)
plane = case == "ch" ? (radius=1.0, coriolis=x -> f0) : (;)
m = KiteNonhydrostatic(primal, VerticalGrid(30, 30000.0), k; height_order=p, velocity, plane...,
                       damping_height=22000.0, damping_rate=0.05, horizontal_theta=:dg_upwind,
                       horizontal_advection=:dg, vertical_theta=vt)
profile = case == "tw" ? (λ, φ, z) -> thermal_wind(λ, φ, z) : case == "bw" ? (λ, φ, z) -> baroclinic_wave(λ, φ, z) :
          (x, yy, z) -> baroclinic_channel(x, yy, z; Ly, f0)
y = initial_state(m, profile); ys = zero_state(m)
x, W = column_points(m), column_weights(m)
mass(y) = sum((y.ρ * W) .* m.vgrid.dz)
lowest_T(y) = (π1 = (NH.Rd .* y.ρθ[1, :] ./ NH.P0) .^ (NH.Rd / NH.Cvd); π1 .* y.ρθ[1, :] ./ y.ρ[1, :])
m0, ps0, θ0 = mass(y), surface_pressure(m, y), y.ρθ ./ y.ρ
tag = "$(case)_$(space)_$(mesh)$(res)_$(vt)"
nu, nh, nlev = size(y.u, 2), num_columns(m), num_layers(m)
@printf("%s: %s %s %d, vertical_theta %s, dt %.0f s\n", tag, space, mesh, res, vt, dt)
@printf("dofs: u %d, columns %d (%d layers), total %d\n", nu, nh, nlev, nu * nlev + nh * (3nlev - 1))
every = round(Int, (case == "ch" ? 21600 : 43200) / dt)
snapdays = case == "bw" ? (6, 8, 10) : case == "ch" ? (8, 10, 12, 14) : ()
rows = []
t = 0.0
for n in 1:round(Int, days * 86400 / dt)
    global t += @elapsed step!(m, y, dt, ys)
    n % every == 0 || continue
    d = n * dt / 86400
    ps = surface_pressure(m, y)
    row = (day=d, psmin=minimum(ps) / 100, psmax=maximum(ps) / 100, dps=maximum(abs, ps .- ps0) / 100,
           dθ=maximum(abs, y.ρθ ./ y.ρ .- θ0), wmax=maximum(abs, y.w), mass=mass(y) / m0 - 1,
           A10=case == "ch" ? wave_amplitude(ps, x, W, Lx, Ly, 10) : NaN)
    push!(rows, row)
    @printf("day %5.2f  p_s min %7.2f max %7.2f hPa  max|Δp_s| %.2e  max|Δθ| %.2e  max|w| %.2e  mass %.1e%s  %.2f s/step\n",
            d, row.psmin, row.psmax, row.dps, row.dθ, row.wmax, row.mass,
            case == "ch" ? @sprintf("  k=10 %.3e Pa", row.A10) : "", t / n)
    flush(stdout)
    serialize(joinpath(@__DIR__, "data", "$(tag)_series.jls"), (; nu, nh, nlev, rows))
    all(isfinite, y.ρ) || break
    if isapprox(d, round(d)) && round(Int, d) in snapdays
        pts = case == "ch" ? (x=getindex.(x, 1), y=getindex.(x, 2)) :
              (ll = [rad2deg.(NH.lonlat(xi)) for xi in x]; (lon=first.(ll), lat=last.(ll)))
        serialize(joinpath(@__DIR__, "data", "$(tag)_day$(round(Int, d)).jls"), (; pts..., ps, T=lowest_T(y)))
    end
end
