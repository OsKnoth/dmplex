# Runs the Galewsky jet (examples/galewsky.jl) and draws the relative vorticity
# seen from above the jet as docs/images/galewsky.svg:
#
#   julia --project docs/galewsky_figure.jl [mesh] [resolution] [days]
include(joinpath(@__DIR__, "make_figures.jl"))
include(joinpath(@__DIR__, "..", "examples", "galewsky.jl"))

mesh = length(ARGS) >= 1 ? ARGS[1] : "hexagon"
resolution = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
days = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 6.0
model, y = main([mesh, string(resolution), string(days), "true"])

ζ = mass_solver(model.Z; lumped=true) \ (-(model.R * y[velocity_dofs(model)]))
vorticity(c, ξ, _) = evaluate(model.Z, ζ, c, ξ)[2] * OMEGA
scale = 1.5e-4  # 1/s, the usual Galewsky color range
view = normalize(SVector(cosd(50) * cosd(0), cosd(50) * sind(0), sind(50)))
mkpath(OUT)
svg("galewsky.svg", 640, 400) do io
    text(io, 24, 34, @sprintf("Galewsky jet, relative vorticity after %g days", days); anchor="start", weight="bold")
    text(io, 24, 54, @sprintf("Kite method, RT1 × Q2 × CG2, %d primal cells, %d kites", num_cells(model.primal),
                              num_cells(model.kites)); size=12, color=INK_SECONDARY, anchor="start")
    field_globe(io, model.kites, vorticity, 230, 225, 155, v -> diverging(v / scale); m=1, view)
    x0, y0, w, hgt = 450, 130, 18, 190
    for j in 0:49
        println(io, """<rect x="$x0" y="$(y0 + j * hgt / 50)" width="$w" height="$(hgt / 50 + 0.5)" fill="$(diverging(1 - 2j / 49))"/>""")
    end
    for (t, label) in ((1, "+1.5·10⁻⁴"), (0, "0"), (-1, "−1.5·10⁻⁴"))
        ty = y0 + (1 - t) / 2 * hgt
        println(io, """<line x1="$(x0 + w)" y1="$ty" x2="$(x0 + w + 5)" y2="$ty" stroke="$INK_SECONDARY"/>""")
        text(io, x0 + w + 9, ty + 4, label; size=12, color=INK_SECONDARY, anchor="start")
    end
    text(io, x0, y0 - 12, "ζ (1/s)"; size=12, color=INK_SECONDARY, anchor="start")
end
