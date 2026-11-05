#
#   julia --project plot_series.jl <snapshot dir> <log dir>   (logs of thermal_wind.jl and baroclinic_channel.jl)
include("figtools.jl")
const P, R = ARGS[1], ARGS[2]   # kite3d/pics and kite3d/runs
const COL = Dict("hexagon" => :royalblue3, "cubed" => :darkorange2, "triangles" => :seagreen, "quadrilateral" => :royalblue3, "triangle" => :seagreen)
const LAB = Dict("hexagon" => "hexagons", "cubed" => "cubed sphere", "triangles" => "triangles")
"Columns `(t, max|w|, max|θ′|)` of a thermal-wind log."
function twlog(f)
    rows = [split(l) for l in eachline(f) if startswith(l, "day")]
    return parse.(Float64, getindex.(rows, 2)), parse.(Float64, getindex.(rows, 4)), parse.(Float64, getindex.(rows, 9))
end

# --- spectra
sp = deserialize(joinpath(@__DIR__, "data", "spectra.jls"))
ps = [begin
          λ = sp[(cells, name)]
          scatter(real(λ), imag(λ); ms=2, msw=0, c=:black, label=false, xlims=(-10, 7), ylims=(-30, 30),
                  title=@sprintf("%s, %s: max Re λ = %.3f", cells == :quadrilateral ? "quadrilaterals" : "triangles", name, maximum(real, λ)),
                  xlabel=cells == :triangle ? "Re λ Δx/|u|" : "", ylabel=name == "vector invariant" ? "Im λ Δx/|u|" : "", titlefontsize=9)
          vline!([0]; c=:gray, ls=:dash, label=false)
      end for cells in (:quadrilateral, :triangle) for name in ("vector invariant", "DG, central", "DG, upwind")]
save(plot(ps...; layout=(2, 3), size=(1300, 700), left_margin=4Plots.mm, bottom_margin=4Plots.mm), "kite_spectra")

# --- thermal wind
tr = deserialize(joinpath(P, "thermal_trisk_4.jls")).rows
old = deserialize(joinpath(P, "thermal_kite_3_1_rt_upwind.jls")).rows
pw = plot(; yscale=:log10, xlabel="time (days)", ylabel="max |w| (m/s)", legend=:outerright, ylims=(1e-5, 2), yticks=10.0 .^ (-5:0),
          xlims=(0, 5), title="thermal wind, 30 layers, max |w|", size=(1000, 450))
plot!(pw, [r.t for r in tr], [r.w for r in tr]; c=:black, lw=2, label="TRiSK, hexagon 4")
plot!(pw, [r.t for r in old], [r.w for r in old]; c=:gray, lw=1.5, ls=:dot, label="kite, upwind θ trace, hexagons")
for mesh in ("hexagon", "cubed", "triangles")
    t, w, _ = twlog(joinpath(R, "tw_$(mesh)_dg_upwind.log"))
    plot!(pw, t, w; c=COL[mesh], lw=1.5, ls=:dash, marker=:circle, ms=2.5, label="kite, DG θ, vector-invariant u, $(LAB[mesh])")
    t, w, _ = twlog(joinpath(R, "twdg_$(mesh).log"))
    plot!(pw, t, w; c=COL[mesh], lw=2.5, marker=:circle, ms=3, label="kite, DG θ and DG u, $(LAB[mesh])")
end
save(plot(pw; left_margin=5Plots.mm, bottom_margin=5Plots.mm), "kite3d_thermal_wind")

# --- channel: growth of the k = 10 wave
pc = plot(; yscale=:log10, xlabel="time (days)", ylabel="amplitude of wavenumber 10 in surface pressure (Pa)", yticks=10.0 .^ (-1:3), legend=:topleft,
          title="baroclinic channel, 60 × 18 cells (667 km), 30 layers", size=(800, 500))
for cells in ("quadrilateral", "triangle")
    rows = [split(l) for l in eachline(joinpath(R, "chdg60_$cells.log")) if startswith(l, "day")]
    t = parse.(Float64, getindex.(rows, 2))
    A = [parse(Float64, r[findfirst(==("k=10:"), r) + 1]) for r in rows]
    cells == "quadrilateral" ? plot!(pc, t, A; c=COL[cells], lw=2.5, marker=:circle, ms=3, label="quadrilaterals") :
        scatter!(pc, t, A; c=:white, msc=COL[cells], msw=1.5, marker=:utriangle, ms=5, label="triangles")
end
t = 2:0.5:12
plot!(pc, t, 0.45 .* exp.(6.59e-6 * 86400 .* (t .- 2)); c=:black, ls=:dash, lw=1.5, label="linear QG growth, 6.59e-6 1/s")
save(plot(pc; left_margin=5Plots.mm, bottom_margin=5Plots.mm), "kite3d_channel_growth")
