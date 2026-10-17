# Runs the Bickley jet (examples/bickley.jl) and draws the relative vorticity
# as docs/images/bickley.svg:
#
#   julia --project docs/bickley_figure.jl [quadrilateral|triangle] [n] [T]
include(joinpath(@__DIR__, "make_figures.jl"))
include(joinpath(@__DIR__, "..", "examples", "bickley.jl"))

cells = length(ARGS) >= 1 ? ARGS[1] : "quadrilateral"
n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 32
T = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 40.0
model, y = main([cells, string(n), string(T)])
ζ, _, _ = diagnostics(model, y[velocity_dofs(model)], y[height_dofs(model)])

size_px, x0, y0 = 300.0, 30.0, 70.0
to_screen(p) = @sprintf("%.1f,%.1f", x0 + p[1] / L * size_px, y0 + (1 - p[2] / L) * size_px)
scale = maximum(abs, ζ)
mkpath(OUT)
svg("bickley.svg", 640, 400) do io
    text(io, 24, 34, @sprintf("Bickley jet, relative vorticity at t = %g", T); anchor="start", weight="bold")
    text(io, 24, 54, @sprintf("Kite method, RT1 × Q2 × CG2, periodic %d × %d %s mesh, %d kites", n, n, cells,
                              num_cells(model.kites)); size=12, color=INK_SECONDARY, anchor="start")
    println(io, """<clipPath id="box"><rect x="$x0" y="$y0" width="$size_px" height="$size_px"/></clipPath><g clip-path="url(#box)">""")
    for c in cell_range(model.kites), (points, ξ) in subcells(model.kites, c, 2)
        fill = diverging(evaluate(model.Z, ζ, c, ξ)[2] / scale)
        println(io, """<polygon points="$(join(to_screen.(points), " "))" fill="$fill" stroke="$fill" stroke-width="0.4"/>""")
    end
    println(io, "</g>")
    println(io, """<rect x="$x0" y="$y0" width="$size_px" height="$size_px" fill="none" stroke="$INK_SECONDARY"/>""")
    bx, by, w, hgt = 370, 110, 18, 190
    for j in 0:49
        println(io, """<rect x="$bx" y="$(by + j * hgt / 50)" width="$w" height="$(hgt / 50 + 0.5)" fill="$(diverging(1 - 2j / 49))"/>""")
    end
    for (t, label) in ((1, @sprintf("+%.2f", scale)), (0, "0"), (-1, @sprintf("−%.2f", scale)))
        ty = by + (1 - t) / 2 * hgt
        println(io, """<line x1="$(bx + w)" y1="$ty" x2="$(bx + w + 5)" y2="$ty" stroke="$INK_SECONDARY"/>""")
        text(io, bx + w + 9, ty + 4, label; size=12, color=INK_SECONDARY, anchor="start")
    end
    text(io, bx, by - 12, "ζ"; size=12, color=INK_SECONDARY, anchor="start")
    text(io, bx, by + hgt + 40, "doubly periodic [0, 4π)², no rotation,"; size=12, color=INK_SECONDARY, anchor="start")
    text(io, bx, by + hgt + 58, "Froude number 0.2, u = sech²(y − 2π)"; size=12, color=INK_SECONDARY, anchor="start")
end
