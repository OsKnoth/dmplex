# Generates the README figures as plain SVG (no plotting dependency):
#
#   julia --project docs/make_figures.jl
#
# Meshes and fields are drawn in orthographic projection; cells are painted back
# to front with great-circle edges.
using LinearAlgebra
using Printf
using StaticArrays
using dmplex

const OUT = joinpath(@__DIR__, "images")
const SURFACE = "#fcfcfb"
const INK = "#0b0b0b"
const INK_SECONDARY = "#52514e"
const CELL_FILL = "#e3ecf7"
const VIEW = normalize(SVector(cosd(25) * cosd(20), cosd(25) * sind(20), sind(25)))

# --- projection ---------------------------------------------------------------

function screen_basis(view)
    e1 = normalize(cross(SVector(0.0, 0.0, 1.0), view))
    return e1, cross(view, e1)
end

"Boundary of `cell` as points on the sphere, edges subdivided along great circles."
function cell_outline(dm, cell; segments=4)
    x = [get_coordinates(dm, v) for v in get_cell_vertices(dm, cell)]
    points = SVector{3, Float64}[]
    for i in eachindex(x), s in 0:(segments - 1)
        t = s / segments
        push!(points, normalize((1 - t) * x[i] + t * x[mod1(i + 1, length(x))]))
    end
    return points
end

"Painter's order: cells facing the viewer, back to front."
function visible_cells(dm, view)
    depth = Dict(c => dot(normalize(sum(get_coordinates(dm, v) for v in get_cell_vertices(dm, c))), view)
                 for c in cell_range(dm))
    return sort([c for c in cell_range(dm) if depth[c] > -0.2]; by=c -> depth[c])
end

function polygon(points, cx, cy, radius, basis; fill, stroke, width)
    e1, e2 = basis
    coords = join((@sprintf("%.1f,%.1f", cx + radius * dot(p, e1), cy - radius * dot(p, e2)) for p in points), " ")
    return """<polygon points="$coords" fill="$fill" stroke="$stroke" stroke-width="$width" stroke-linejoin="round"/>"""
end

function globe(io, dm, cx, cy, radius; fill=c -> CELL_FILL, stroke=INK_SECONDARY, width=0.6)
    basis = screen_basis(VIEW)
    for c in visible_cells(dm, VIEW)
        println(io, polygon(cell_outline(dm, c), cx, cy, radius, basis; fill=fill(c), stroke=stroke, width))
    end
    println(io, """<circle cx="$cx" cy="$cy" r="$radius" fill="none" stroke="$INK_SECONDARY" stroke-width="1"/>""")
end

text(io, x, y, s; size=14, color=INK, anchor="middle", weight="normal") =
    println(io, """<text x="$x" y="$y" font-size="$size" fill="$color" text-anchor="$anchor" font-weight="$weight">$s</text>""")

function svg(f, filename, width, height)
    open(joinpath(OUT, filename), "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $width $height" width="$width" height="$height" font-family="Helvetica, Arial, sans-serif">""")
        println(io, """<rect width="$width" height="$height" rx="12" fill="$SURFACE"/>""")
        f(io)
        println(io, "</svg>")
    end
    println("wrote ", joinpath(OUT, filename))
end

# --- colors (OKLab interpolation for the diverging scale) -----------------------

hex_to_rgb(h) = SVector((parse(Int, h[i:(i + 1)]; base=16) / 255 for i in (2, 4, 6))...)
rgb_to_hex(c) = "#" * join(string(round(Int, clamp(x, 0, 1) * 255); base=16, pad=2) for x in c)
to_linear(c) = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055)^2.4
to_srgb(c) = c <= 0.0031308 ? 12.92c : 1.055c^(1 / 2.4) - 0.055

function oklab(rgb)
    r, g, b = to_linear.(rgb)
    l = cbrt(0.4122214708r + 0.5363325363g + 0.0514459929b)
    m = cbrt(0.2119034982r + 0.6806995451g + 0.1073969566b)
    s = cbrt(0.0883024619r + 0.2817188376g + 0.6299787005b)
    return SVector(0.2104542553l + 0.7936177850m - 0.0040720468s,
                   1.9779984951l - 2.4285922050m + 0.4505937099s,
                   0.0259040371l + 0.7827717662m - 0.8086757660s)
end

function srgb(lab)
    L, a, b = lab
    l = (L + 0.3963377774a + 0.2158037573b)^3
    m = (L - 0.1055613458a - 0.0638541728b)^3
    s = (L - 0.0894841775a - 1.2914855480b)^3
    return to_srgb.(SVector(4.0767416621l - 3.3077115913m + 0.2309699292s,
                            -1.2684380046l + 2.6097574011m - 0.3413193965s,
                            -0.0041960771l - 0.7034186147m + 1.7076147010s))
end

# blue arm (sequential blue 400 → 650), neutral gray midpoint, red arm of equal length
const DIVERGING = oklab.(hex_to_rgb.(["#104281", "#3987e5", "#f0efec", "#e34948", "#9e2a2b"]))

"Color for `t ∈ [-1, 1]` on the diverging scale (negative blue, positive red)."
function diverging(t)
    s = (clamp(t, -1, 1) + 1) / 2 * (length(DIVERGING) - 1)
    i = min(floor(Int, s), length(DIVERGING) - 2)
    return rgb_to_hex(srgb((1 - (s - i)) * DIVERGING[i + 1] + (s - i) * DIVERGING[i + 2]))
end

# --- figures --------------------------------------------------------------------

function mesh_gallery()
    meshes = [("Icosahedral", "320 triangles", create_spherical_mesh(2)),
              ("Cubed sphere", "6 × 6² quadrilaterals", create_cubed_sphere_mesh(6)),
              ("Hexagonal (dual)", "162 hexagons/pentagons", create_hexagon_mesh(2)),
              ("HEALPix", "12 × 4² quadrilaterals", create_healpix_mesh(4))]
    svg("meshes.svg", 960, 300) do io
        for (i, (name, detail, dm)) in enumerate(meshes)
            cx = 120 + 240 * (i - 1)
            globe(io, dm, cx, 140, 100)
            text(io, cx, 270, name; weight="bold")
            text(io, cx, 288, detail; size=12, color=INK_SECONDARY)
        end
    end
end

"""
Sub-polygons of `cell` (an `m × m` split of the reference cell) as points on
the sphere, each with its reference center, so DG/CG fields can be drawn with
their variation inside the cell.
"""
function subcells(dm, cell, m)
    cmap = CellMap(dm, cell)
    on_sphere(ξ) = map_point(cmap, SVector{2, Float64}(ξ))[1]
    pieces = Tuple{Vector{SVector{3, Float64}}, SVector{2, Float64}}[]
    if get_cone_size(dm, cell) == 4
        g = range(-1, 1; length=m + 1)
        for j in 1:m, i in 1:m
            corners = [(g[i], g[j]), (g[i + 1], g[j]), (g[i + 1], g[j + 1]), (g[i], g[j + 1])]
            push!(pieces, (on_sphere.(corners), SVector((g[i] + g[i + 1]) / 2, (g[j] + g[j + 1]) / 2)))
        end
    else  # triangle (-1,-1), (1,-1), (-1,1): lattice of up and down triangles
        node(a, b) = SVector(-1 + 2a / m, -1 + 2b / m)
        for b in 0:(m - 1), a in 0:(m - 1 - b)
            up = [node(a, b), node(a + 1, b), node(a, b + 1)]
            push!(pieces, (on_sphere.(up), sum(up) / 3))
            if a + b < m - 1
                down = [node(a + 1, b), node(a + 1, b + 1), node(a, b + 1)]
                push!(pieces, (on_sphere.(down), sum(down) / 3))
            end
        end
    end
    return pieces
end

"Draw a scalar field `value(cell, ξ, x)` on the visible hemisphere, painted back to front."
function field_globe(io, dm, value, cx, cy, radius, color; m=2)
    basis = screen_basis(VIEW)
    pieces = Tuple{Float64, Vector{SVector{3, Float64}}, String}[]
    for c in cell_range(dm), (points, ξ) in subcells(dm, c, m)
        center = normalize(sum(points))
        depth = dot(center, VIEW)
        depth > -0.1 && push!(pieces, (depth, points, color(value(c, ξ, center))))
    end
    for (_, points, fill) in sort!(pieces; by=first)
        println(io, polygon(points, cx, cy, radius, basis; fill, stroke=fill, width=0.5))
    end
    println(io, """<circle cx="$cx" cy="$cy" r="$radius" fill="none" stroke="$INK_SECONDARY" stroke-width="1"/>""")
end

function shallow_water_figure()
    dm = create_cubed_sphere_mesh(16)
    k = 1
    model = LinearShallowWater(dm, k)
    u0 = 0.5
    # bump inside a cube panel (30°N, 20°E), away from the panel corners
    bump = SVector(cosd(30) * cosd(20), cosd(30) * sind(20), sind(30))
    balanced(x) = -u0 * x[3]^2
    y = initial_state(model, x -> u0 * SVector(-x[2], x[1], 0.0),
                      x -> balanced(x) + 0.1 * exp(-30 * (1 - dot(x, bump))))
    cache = RK4Cache(y)
    T = 0.6
    nsteps = ceil(Int, T / (0.1 * sqrt(4π / num_cells(dm)) / (k + 1)^2))
    for _ in 1:nsteps
        rk4_step!(y, T / nsteps, tendency!, cache, model)
    end
    h = y[height_dofs(model)]
    anomaly(c, ξ, _) = (r = evaluate(model.H, h, c, ξ); r[2] - balanced(r[1]))
    scale = maximum(abs(anomaly(c, SVector(0.0, 0.0), nothing)) for c in cell_range(dm))

    svg("shallow_water.svg", 640, 360) do io
        text(io, 24, 34, "Height anomaly after an initial bump, t = $T"; anchor="start", weight="bold")
        text(io, 24, 54, "Linear rotating shallow water, RT$k × DG$k, cubed sphere 6 × 16²"; size=12,
             color=INK_SECONDARY, anchor="start")
        field_globe(io, dm, anomaly, 230, 205, 140, v -> diverging(v / scale))
        # color bar
        x0, y0, w, hgt = 450, 110, 18, 190
        for j in 0:49
            t = 1 - 2j / 49
            println(io, """<rect x="$x0" y="$(y0 + j * hgt / 50)" width="$w" height="$(hgt / 50 + 0.5)" fill="$(diverging(t))"/>""")
        end
        for (t, label) in ((1, @sprintf("+%.3f", scale)), (0, "0"), (-1, @sprintf("−%.3f", scale)))
            ty = y0 + (1 - t) / 2 * hgt
            println(io, """<line x1="$(x0 + w)" y1="$ty" x2="$(x0 + w + 5)" y2="$ty" stroke="$INK_SECONDARY"/>""")
            text(io, x0 + w + 9, ty + 4, label; size=12, color=INK_SECONDARY, anchor="start")
        end
        text(io, x0, y0 - 12, "h − h_balanced"; size=12, color=INK_SECONDARY, anchor="start")
    end
end

mkpath(OUT)
mesh_gallery()
shallow_water_figure()
