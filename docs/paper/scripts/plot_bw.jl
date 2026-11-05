#
#   julia --project plot_bw.jl
include("figtools.jl")
g = deserialize(joinpath(@__DIR__, "data", "bw_grids.jls"))
lon, lat, F = g.lon, g.lat, g.fields
meshes = (("hexagon", "hexagons (23 682 columns)"), ("cubed", "cubed sphere (21 600)"), ("triangles", "triangles (24 320)"))
const PSMIN = Dict(("hexagon", 8) => 968.30, ("hexagon", 10) => 925.82, ("cubed", 8) => 966.85, ("cubed", 10) => 926.34,
                   ("triangles", 8) => 967.30, ("triangles", 10) => 925.39)   # columns, from the run logs
const LON, LAT = (0, 240), (20, 80)
function panel(mesh, label, day, f; bar)
    il, jl = findall(l -> LON[1] <= l <= LON[2], lon), findall(l -> LAT[1] <= l <= LAT[2], lat)
    z = F[(mesh, day, f)][jl, il]
    lev, cm, ct = f == :ps ? (921:6:1029, cgrad(:viridis), "surface pressure (hPa)") :
                  (222:4:302, cgrad(:RdYlBu, rev=true), "temperature at 500 m (K)")
    zc = clamp.(z, first(lev) + 1e-6, last(lev) - 1e-6)
    p = contourf(lon[il], lat[jl], zc; levels=lev, c=cm, clims=(first(lev), last(lev)), lw=0, colorbar=bar,
                 colorbar_title=bar ? ct : "", xticks=0:30:240, yticks=20:20:80, xlims=LON, ylims=LAT,
                 title=f == :ps ? @sprintf("%s, day %d, min %.1f hPa", label, day, PSMIN[(mesh, day)]) : @sprintf("%s, day %d", label, day),
                 xlabel=mesh == "triangles" ? "longitude (°E)" : "", ylabel=day == 8 ? "latitude (°N)" : "")
    contour!(p, lon[il], lat[jl], z; levels=lev, c=:black, lw=0.35, colorbar_entry=false)
    return p
end
for f in (:ps, :T)
    ps = [panel(m, l, d, f; bar=d == 10) for (m, l) in meshes for d in (8, 10)]
    p = plot(ps...; layout=grid(3, 2; widths=[0.455, 0.545]), size=(1400, 820), left_margin=4Plots.mm,
             right_margin=4Plots.mm, bottom_margin=3Plots.mm)
    save(p, "kite3d_baroclinic_wave_$(f)")
end
