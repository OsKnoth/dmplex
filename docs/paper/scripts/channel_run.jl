# Baroclinic channel (examples/baroclinic_channel.jl) with snapshots of p_s and T of the lowest layer
#
#   julia --project channel_run.jl 60 14 200 quadrilateral|triangle
using Printf, Serialization, dmplex
include(joinpath(@__DIR__, "..", "..", "..", "examples", "baroclinic_channel.jl"))
nx, days, dt, cells = parse(Int, ARGS[1]), parse(Float64, ARGS[2]), parse(Float64, ARGS[3]), Symbol(ARGS[4])
Lx, Ly, f0 = 4e7, 6e6, 1.0313e-4
m = KiteNonhydrostatic(create_periodic_plane(nx, 3nx ÷ 10; Lx, Ly=2Ly, cells), VerticalGrid(30, 30000.0), 1;
                       radius=1.0, coriolis=x -> f0, horizontal_theta=:dg_upwind, horizontal_advection=:dg,
                       damping_height=22000.0, damping_rate=0.05)
y = initial_state(m, (x, yy, z) -> baroclinic_channel(x, yy, z; Ly, f0)); ys = zero_state(m)
x, W = column_points(m), column_weights(m)
rows = []
A0 = wave_amplitude(surface_pressure(m, y), x, W, Lx, Ly, 10)
push!(rows, (t=0.0, A=A0, psmin=minimum(surface_pressure(m, y))))
for n in 1:round(Int, days * 86400 / dt)
    step!(m, y, dt, ys)
    if n % round(Int, 21600 / dt) == 0
        d = n * dt / 86400
        ps = surface_pressure(m, y)
        push!(rows, (t=d, A=wave_amplitude(ps, x, W, Lx, Ly, 10), psmin=minimum(ps), wmax=maximum(abs, y.w)))
        @printf("day %5.2f  p_s min %7.2f  k=10 %.3e Pa  max|w| %.2e\n", d, minimum(ps) / 100, rows[end].A, rows[end].wmax)
        flush(stdout)
        snap = isapprox(d, round(d)) && round(Int, d) in (8, 10, 12, 14)
        π1 = (NH.Rd .* y.ρθ[:, 1] ./ NH.P0) .^ (NH.Rd / NH.Cvd)
        snap && serialize(joinpath(@__DIR__, "data", "channel_$(cells)$(nx)_day$(round(Int, d)).jls"),
                          (ps=ps, T=π1 .* y.ρθ[:, 1] ./ y.ρ[:, 1], z1=m.vgrid.z_full[1]))
        serialize(joinpath(@__DIR__, "data", "channel_$(cells)$(nx)_series.jls"), rows)
    end
end
