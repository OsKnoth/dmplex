# Baroclinic channel: p_s and T of the lowest layer at days 8, 10, 12, 14 (from channel_run.jl)
include("figtools.jl")
Lx, Ly = 4e7, 6e6
xg, yg = 0:50e3:Lx, 0:50e3:Ly
"Field on the (x, y) grid from samples in the doubly periodic box (the channel half y < Ly)."
function xy_grid(xs, vs)
    q = [SVector(x, y, 0.0) for y in yg for x in xg]
    p = [SVector(x[1], x[2], 0.0) for x in xs]
    # periodic copies in x so that the ends of the channel see their neighbours
    pp = vcat(p, [x + SVector(Lx, 0, 0) for x in p], [x - SVector(Lx, 0, 0) for x in p])
    return permutedims(reshape(idw(pp, repeat(vs, 3), q), length(xg), length(yg)))
end
grids = Dict()
isfile(joinpath(@__DIR__, "data", "channel_grids.jls")) && (grids = deserialize(joinpath(@__DIR__, "data", "channel_grids.jls")))
isempty(grids) && for (cells, label) in ((:quadrilateral, "quadrilaterals"), (:triangle, "triangles"))
    m = KiteNonhydrostatic(create_periodic_plane(60, 18; Lx, Ly=2Ly, cells), VerticalGrid(1, 30000.0), 1; radius=1.0)
    for day in (8, 10, 12, 14)
        s = deserialize(joinpath(@__DIR__, "data", "channel_$(cells)60_day$day.jls"))
        for (f, v) in ((:ps, s.ps ./ 100), (:T, s.T))
            xs, vs = kite_samples(m, v)
            grids[(cells, day, f)] = xy_grid(xs, vs)
        end
        @printf("%s day %d: min p_s %.2f hPa (columns)\n", cells, day, minimum(s.ps) / 100)
        grids[(cells, day, :psmin)] = minimum(s.ps) / 100
    end
end
isfile(joinpath(@__DIR__, "data", "channel_grids.jls")) || serialize(joinpath(@__DIR__, "data", "channel_grids.jls"), grids)
for (cells, label) in ((:quadrilateral, "60 × 18 quadrilaterals"), (:triangle, "60 × 18 triangles"))
    ps = []
    for day in (8, 10, 12, 14), f in (:ps, :T)
        z = grids[(cells, day, f)]
        lev, cm, ct = f == :ps ? (952:4:1028, cgrad(:viridis), "surface pressure (hPa)") :
                      (266:2:306, cgrad(:RdYlBu, rev=true), "temperature at 500 m (K)")
        zc = clamp.(z, first(lev) + 1e-6, last(lev) - 1e-6)
        p = contourf(xg ./ 1e3, yg ./ 1e3, zc; levels=lev, c=cm, clims=(first(lev), last(lev)), lw=0,
                     colorbar_title=ct, xticks=(0:5000:40000, string.(0:5000:40000)), yticks=0:2000:6000, xlims=(0, 40000), ylims=(0, 6000),
                     title=f == :ps ? @sprintf("%s, day %d, min %.1f hPa", label, day, grids[(cells, day, :psmin)]) :
                                      @sprintf("%s, day %d", label, day),
                     xlabel=day == 14 ? "x (km)" : "", ylabel="y (km)")
        contour!(p, xg ./ 1e3, yg ./ 1e3, z; levels=lev, c=:black, lw=0.35, colorbar_entry=false)
        push!(ps, p)
    end
    p = plot(ps...; layout=(4, 2), size=(1600, 820), left_margin=5Plots.mm, right_margin=3Plots.mm,
             bottom_margin=3Plots.mm)
    save(p, "kite3d_channel_$(cells)")
end
