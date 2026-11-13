# Data for the figures of docs/polyhedral/polyhedral_doc.tex (plotted by plot_doc.py)
#   julia --project=. examples/polyhedral/figure_data.jl   -> docfig/*.txt
using dmplex, LinearAlgebra, StaticArrays, Printf, Random
PH = dmplex.Polyhedral
const D = "docfig"
mkpath(D)

"Polygons (one per line, x y z ...) and a value per polygon."
function write_polygons(path, polys, values)
    open(path, "w") do io
        for (P, v) in zip(polys, values)
            print(io, v)
            foreach(x -> print(io, " ", x[1], " ", x[2], " ", x[3]), P)
            println(io)
        end
    end
end

# 1. one layer of cells over the sphere: the bottom faces of three meshes
bottom_faces(cx) = [f for f in 1:num_poly_faces(cx) if all(x -> abs(norm(x) - 1) < 1e-12, PH.face_points(cx, f))]
bottom(cx) = [PH.face_points(cx, f) for f in bottom_faces(cx)]
bottom_cells(cx) = (fc = face_cells(cx); [max(fc[1, f], fc[2, f]) for f in bottom_faces(cx)])
for (name, mesh) in (("hexagons", create_hexagon_mesh(3)), ("triangles", create_spherical_mesh(3)), ("cubed", create_cubed_sphere_mesh(8)))
    cx = extruded_complex(ExtrudedPlex(mesh, 1), [0.0, 0.01]; radius=1.0)
    P = bottom(cx)
    write_polygons(joinpath(D, "mesh_$name.txt"), P, length.(P))
    @printf("%s: %d cells\n", name, num_poly_cells(cx))
end

# 2. solid-body rotation of a Gaussian hill, one revolution, degrees 1 and 2 (the test's setup)
for level in (4, 5)
    sh = extruded_complex(ExtrudedPlex(create_hexagon_mesh(level), 1), [0.0, 0.01]; radius=1.0)
    Ω = 2π * normalize(SVector(1.0, 0.0, 1.0))
    U = stream_function_flux(sh, r̂ -> dot(Ω, r̂); G=r -> r^3 / 3)
    for degree in (1, 2)
        m = PolyAdvection(sh, U; degree, basis=:tangential)
        q = [exp(-sum(abs2, normalize(x) - SVector(1.0, 0, 0)) / 0.16) for x in m.geometry.cell_centroid]
        q0, V = copy(q), m.geometry.volume
        nst = ceil(Int, 1 / (0.4 * upwind_time_step(m)))
        cache = SSPRK3Cache(q)
        for _ in 1:nst
            ssprk3_step!(q, 1 / nst, PH.tendency!, cache, m)
        end
        e2 = sqrt(sum(V .* (q .- q0) .^ 2) / sum(V .* q0 .^ 2))
        @printf("rotation level %d degree %d: %d cells, %d steps, l2 %.3e, max %.3f -> %.3f, min %.2e\n",
                level, degree, num_poly_cells(sh), nst, e2, maximum(q0), maximum(q), minimum(q))
        level == 5 && write_polygons(joinpath(D, "rotation_p$degree.txt"), bottom(sh), q[bottom_cells(sh)])
    end
end

# 3. cut cells: the jittered prisms of the test, cut by the mountain, before and after merging,
#    and the steady transport (the faces on the wall y = 0)
zs(x) = 0.03 + 0.25 * exp(-(x / 0.4)^2) * cos(π * x / 0.5)^2
level(x) = x[3] - zs(x[1])
ψ(x) = (x[3] - zs(x[1])) * (1 - x[3]) * sin(π * (x[1] + 1) / 2)
function grid(nx, nz; ny=2, Ly=0.25)
    rng = Random.MersenneTwister(1)
    id(i, j) = (j - 1) * (nx + 1) + i
    jx(i) = i == 1 || i == nx + 1 ? 0.0 : 0.3 * (rand(rng) - 0.5)
    jy(j) = j == 1 || j == ny + 1 ? 0.0 : 0.3 * (rand(rng) - 0.5)
    pts = [SVector(-1 + 2 * (i - 1 + jx(i)) / nx, Ly * (j - 1 + jy(j)) / ny, 0.0) for j in 1:(ny + 1) for i in 1:(nx + 1)]
    polys = reduce(vcat, [[[id(i, j), id(i + 1, j), id(i + 1, j + 1)], [id(i, j), id(i + 1, j + 1), id(i, j + 1)]]
                          for j in 1:ny for i in 1:nx])
    return extruded_complex(ExtrudedPlex(plex_from_polygons(polys, pts), nz), collect(range(0.0, 1.0; length=nz + 1)))
end
"The faces on y = 0 and their cells."
function wall(cx)
    fc = face_cells(cx)
    fs = [f for f in 1:num_poly_faces(cx) if all(x -> abs(x[2]) < 1e-14, PH.face_points(cx, f))]
    return [PH.face_points(cx, f) for f in fs], [max(fc[1, f], fc[2, f]) for f in fs]
end
let nx = 16, nz = 8
    cx = grid(nx, nz)
    g0 = PolyGeometry(cx)
    cut, parent, terrain = cut_complex(cx, level)
    g = PolyGeometry(cut)
    frac = g.volume ./ g0.volume[parent]
    P, c = wall(cut)
    write_polygons(joinpath(D, "cut_fraction.txt"), P, frac[c])
    merged, group = merge_small_cells(cut, g0.volume[parent]; fraction=0.5, geometry=g)
    P, c = wall(merged)
    write_polygons(joinpath(D, "cut_merged.txt"), P, c)
    @printf("cut: %d cells, %d cut, smallest fraction %.1e, merged %d cells\n", num_poly_cells(cut), length(terrain),
            minimum(frac), num_poly_cells(merged))
end
let nx = 32, nz = 16
    cx = grid(nx, nz)
    g0 = PolyGeometry(cx)
    cut, parent, _ = cut_complex(cx, level)
    merged, _ = merge_small_cells(cut, g0.volume[parent]; fraction=0.5)
    g = PolyGeometry(merged)
    U = vector_potential_flux(merged, x -> SVector(0.0, ψ(x), 0.0))
    m = PolyAdvection(merged, U; degree=2, geometry=g, boundary_oversampling=2.5)
    q = [exp(2ψ(x)) for x in g.cell_centroid]
    q0 = copy(q)
    nst = ceil(Int, 0.5 / (0.4 * upwind_time_step(m)))
    cache = SSPRK3Cache(q)
    for _ in 1:nst
        ssprk3_step!(q, 0.5 / nst, PH.tendency!, cache, m)
    end
    P, c = wall(merged)
    write_polygons(joinpath(D, "cut_steady.txt"), P, (q .- q0)[c])
    write_polygons(joinpath(D, "cut_q.txt"), P, q[c])
    @printf("steady 32×16: %d steps, max error %.2e\n", nst, maximum(abs, q - q0))
end
