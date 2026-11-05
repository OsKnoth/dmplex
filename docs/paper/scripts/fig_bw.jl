# Spherical baroclinic wave: p_s and T(500 m) at days 8 and 10 on the three kite meshes
#
#   julia --project fig_bw.jl        (after baroclinic_wave_run.jl on the three meshes; then plot_bw.jl)
include("figtools.jl")
meshes = (("hexagon", 3, "hexagons"), ("cubed", 12, "cubed sphere"), ("triangles", 3, "triangles"))
lon, lat = 0:0.25:360, 0:0.25:90
fields = Dict()
for (mesh, res, label) in meshes
    primal = mesh == "hexagon" ? create_hexagon_mesh(res) : mesh == "cubed" ? create_cubed_sphere_mesh(res) : create_spherical_mesh(res)
    m = KiteNonhydrostatic(primal, VerticalGrid(1, 30000.0), 1; height_order=2, velocity=:rt)
    for day in (8, 10)
        s = deserialize(joinpath(@__DIR__, "data", "bw_$(mesh)$(res)_dg_day$day.jls"))
        for (f, v) in ((:ps, s.ps ./ 100), (:T, s.T))
            xs, vs = kite_samples(m, v)
            fields[(mesh, day, f)] = lonlat_grid(xs, vs, lon, lat)
        end
        @printf("%s day %d: min p_s %.2f hPa (columns), %.2f (gridded)\n", mesh, day, minimum(s.ps) / 100,
                minimum(fields[(mesh, day, :ps)]))
    end
end
serialize(joinpath(@__DIR__, "data", "bw_grids.jls"), (lon=lon, lat=lat, fields=fields))
