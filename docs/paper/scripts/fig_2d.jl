# Galewsky and Bickley jets: relative vorticity of the kite shallow-water model
#
#   julia --project fig_2d.jl galewsky galewsky_hexagon5_day6.jls
#   julia --project fig_2d.jl bickley bickley_quadrilateral64.jls
include("figtools.jl")
const OMEGA = 7.292e-5
"Samples of the CG vorticity at n × n points of every kite."
function vorticity_samples(model, ζ; n=6)
    t = [-1 + (2i - 1) / n for i in 1:n]
    xs, vs = SVector{3, Float64}[], Float64[]
    for c in cell_range(model.kites), a in t, b in t
        x, v = evaluate(model.Z, ζ, c, SVector(a, b))
        push!(xs, x); push!(vs, v)
    end
    return xs, vs
end
what = ARGS[1]
if what == "galewsky"
    d = deserialize(joinpath(@__DIR__, "data", ARGS[2]))
    primal = create_hexagon_mesh(d.res)
    model = NonlinearKiteShallowWater(primal, 1; gravity=1.0, coriolis=x -> 2x[3])
    xs, vs = vorticity_samples(model, d.ζ)
    lon, lat = -180:0.2:180, 0:0.2:90
    z = lonlat_grid(xs, vs .* OMEGA, lon, lat)
    lev = -1.6e-4:2e-5:1.6e-4
    p = contourf(lon, lat, clamp.(z, -1.6e-4 + 1e-12, 1.6e-4 - 1e-12); levels=lev, c=:balance, lw=0, clims=(-1.6e-4, 1.6e-4),
                 colorbar_title="relative vorticity (1/s)", xlabel="longitude (°E)", ylabel="latitude (°N)",
                 xticks=-180:30:180, yticks=0:15:90, xlims=(-180, 180), ylims=(10, 80), size=(1400, 500),
                 title=@sprintf("Galewsky jet after %d days: kite RT1 × Q2 × CG2, %d hexagons, %d kites (%.0f km)", d.days,
                                d.nprimal, d.nkites, sqrt(4π / d.nkites) * 6371.22))
    contour!(p, lon, lat, z; levels=filter(!iszero, collect(lev)), c=:black, lw=0.3, colorbar_entry=false)
    save(plot(p; left_margin=5Plots.mm, bottom_margin=6Plots.mm), "kite_galewsky")
else
    d = deserialize(joinpath(@__DIR__, "data", ARGS[2]))
    L = 4π
    model = NonlinearKiteShallowWater(create_periodic_plane(d.n, d.n; Lx=L, Ly=L, cells=:quadrilateral), 1;
                                      gravity=25.0, coriolis=x -> 0.0)
    xs, vs = vorticity_samples(model, d.ζ)
    xs = [SVector(mod(x[1], L), mod(x[2], L), 0.0) for x in xs]
    # periodic copies near the seams, so the interpolation sees both sides
    ext = [(x + SVector(i * L, j * L, 0.0), v) for (x, v) in zip(xs, vs) for i in -1:1, j in -1:1
           if -0.3 < x[1] + i * L < L + 0.3 && -0.3 < x[2] + j * L < L + 0.3]
    g = range(0, L; length=1201)
    z = permutedims(reshape(idw(first.(ext), last.(ext), [SVector(a, b, 0.0) for b in g for a in g]), length(g), length(g)))
    lim = maximum(abs, z)
    p = heatmap(g, g, z; c=:balance, clims=(-lim, lim), aspect_ratio=:equal, xlims=(0, L), ylims=(0, L),
                xticks=(0:π:4π, ["0", "π", "2π", "3π", "4π"]), yticks=(0:π:4π, ["0", "π", "2π", "3π", "4π"]),
                xlabel="x", ylabel="y", colorbar_title="relative vorticity", size=(900, 800),
                title=@sprintf("Bickley jet at t = %g: periodic %d × %d quadrilaterals, %d kites", d.T, d.n, d.n, d.nkites))
    save(p, "kite_bickley")
end
