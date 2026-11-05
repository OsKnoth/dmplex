# Steady jet (unperturbed baroclinic wave, 10 layers): growth of max|w| and w at 10.6 km after 42 h
#
#   julia --project fig_jet.jl <snapshot dir>   (snapshots of the steady-jet runs of 2026-11-04)
include("figtools.jl")
const K = ARGS[1]
h6 = 6:6:42
series = [
    ("RT1 × Q2, central, hexagon 3", 6:6:48, [6.25e-3, 1.06e-2, 1.75e-2, 4.29e-2, 9.72e-2, 0.1745, 0.5525, 0.9185], :firebrick, :solid),
    ("RT1 × Q2, upwind θ, hexagon 3", h6, [3.3e-3, 4.4e-3, 5.5e-3, 8.9e-3, 1.4e-2, 1.5e-2, 2.9e-2], :royalblue3, :solid),
    ("RT1 × Q2, central, hexagon 4", 1:6, [1e-3, 7.1e-3, 0.042, 0.23, 0.94, 4.1], :firebrick, :dash),
    ("RT1 × Q2, upwind θ, hexagon 4", 1:12, [1.7e-3, 2.6e-3, 1.9e-3, 1.7e-3, 2.2e-3, 2.5e-3, 3.1e-3, 3.2e-3, 3.2e-3, 3.0e-3, 3.7e-3, 4.0e-3], :royalblue3, :dash),
    ("Q2 × dual Radau 2, central, hexagon 3", h6, [1.5e-3, 5.5e-3, 3.4e-2, 0.2, 0.95, 1.2, 1.6], :darkorange2, :solid),
    ("Q2 × dual Radau 2, upwind θ, hexagon 3", h6, [1.2e-3, 1.6e-3, 2.7e-3, 6.8e-3, 3.2e-2, 0.13, 0.45], :purple, :solid),
    ("Q1 × dual Radau 1, central, hexagon 3", h6, [1.4e-2, 1.3e-2, 1.3e-2, 1.2e-2, 1.5e-2, 1.9e-2, 2.4e-2], :seagreen, :solid),
]
p = plot(; yscale=:log10, xlabel="time (h)", ylabel="max |w| (m/s)", legend=:outerright, size=(1050, 450),
         title="unperturbed baroclinic jet, 10 layers: central and upwind θ trace on the primal faces",
         ylims=(5e-4, 10), xlims=(0, 48), yticks=10.0 .^ (-3:1))
for (l, t, w, c, s) in series
    plot!(p, t, w; label=l, color=c, ls=s, marker=:circle, ms=3, lw=2)
end
save(plot(p; left_margin=5Plots.mm, bottom_margin=5Plots.mm), "kite3d_jet_growth")

m = KiteNonhydrostatic(create_hexagon_mesh(3), VerticalGrid(1, 30000.0), 1; height_order=2, velocity=:rt)
lon, lat = -180:0.25:180, -90:0.25:90
maps = map((("snap.jls", "central θ trace"), ("snap_upwind.jls", "upwind θ trace"))) do (f, name)
    d = deserialize(joinpath(K, f))
    s = d.snaps[42]
    j = argmax(abs.(deserialize(joinpath(K, "snap.jls")).snaps[42].w))[2]
    w = s.w[:, j]
    lim = maximum(abs, w)
    xs, vs = kite_samples(m, w)
    z = lonlat_grid(xs, vs, lon, lat)
    heatmap(lon, lat, z; c=:balance, clims=(-lim, lim), colorbar_title="w (m/s)", xlabel="longitude (°E)",
            ylabel="latitude (°N)", xticks=-180:60:180, yticks=-90:30:90, aspect_ratio=:equal, xlims=(-180, 180), ylims=(-90, 90),
            title=@sprintf("%s: w at %.1f km after 42 h, max |w| = %.3f m/s", name, d.z_half[j] / 1000, lim))
end
save(plot(maps...; layout=(2, 1), size=(1000, 1000)), "kite3d_jet_w")
