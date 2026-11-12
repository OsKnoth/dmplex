@testset "polyhedral: cell complexes and the integration of monomials" begin
    PH = dmplex.Polyhedral
    B = MonomialBasis(4)
    @test num_monomials(B) == 35 && B.exponents[1] == (0, 0, 0)
    @test all(j -> all(i -> B.lower[i, j] == 0 || B.exponents[B.lower[i, j]] == B.exponents[j] .- ntuple(l -> l == i, 3), 1:3),
              1:num_monomials(B))
    prism(polygons, points, levels) =
        extruded_complex(ExtrudedPlex(plex_from_polygons(polygons, [SVector(p..., 0.0) for p in points]), length(levels) - 1),
                         levels)
    exact(cx, c, f) = cell_moments(cx, c, B; center=zero(SVector{3, Float64}), scale=1.0) ≈ map(f, B.exponents)
    # unit cube, triangular prism of height 2, non-convex L prism (three unit squares)
    cube = prism([[1, 2, 3, 4]], [(0, 0), (1, 0), (1, 1), (0, 1)], [0.0, 1.0])
    @test num_poly_cells(cube) == 1 && num_poly_faces(cube) == 6 && num_poly_edges(cube) == 12
    @test exact(cube, 1, α -> 1 / prod(α .+ 1))
    tri = prism([[1, 2, 3]], [(0, 0), (1, 0), (0, 1)], [0.0, 2.0])
    @test exact(tri, 1, α -> factorial(α[1]) * factorial(α[2]) / factorial(α[1] + α[2] + 2) * 2.0^(α[3] + 1) / (α[3] + 1))
    L = prism([[1, 2, 3, 4, 5, 6]], [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], [0.0, 1.0])
    square(x0, y0, α) = ((x0 + 1)^(α[1] + 1) - x0^(α[1] + 1)) / (α[1] + 1) * ((y0 + 1)^(α[2] + 1) - y0^(α[2] + 1)) / (α[2] + 1) /
                        (α[3] + 1)
    @test exact(L, 1, α -> square(0, 0, α) + square(1, 0, α) + square(0, 1, α))
    # two layers: the faces between them are shared with opposite signs; moments in a shifted, scaled frame
    two = prism([[1, 2, 3, 4]], [(0, 0), (1, 0), (1, 1), (0, 1)], [0.0, 0.5, 1.5])
    @test sum(c -> cell_moments(two, c, B; center=zero(SVector{3, Float64}), scale=1.0), 1:2) ≈
          map(α -> 1.5^(α[3] + 1) / prod(α .+ 1), B.exponents)
    I = cell_moments(cube, 1, B; center=SVector(0.5, 0.5, 0.5), scale=0.5)
    @test I ≈ map(α -> all(iseven, α) ? 1 / 8 * 8 / prod(α .+ 1) : 0.0, B.exponents) atol = 1e-14
    # divergence theorem for the monomials: ∫_C ∂_i y^α = Σ_F s_F n_F,i ∫_F y^α
    for cx in (cube, tri, L)
        g = PolyGeometry(cx)
        faces, signs = PH.cell_cone(cx, 1)
        o, h = zero(SVector{3, Float64}), 1.0
        IC = cell_moments(cx, 1, B; center=o, scale=h)
        for (j, α) in enumerate(B.exponents), i in 1:3
            sum(α) < B.degree || continue
            up = findfirst(==(ntuple(l -> α[l] + (l == i), 3)), B.exponents)
            lhs = (α[i] + 1) * IC[j]
            rhs = sum(s * g.normal[f][i] * face_moments(cx, f, B; center=o, scale=h)[up] for (f, s) in zip(faces, signs))
            @test lhs ≈ rhs atol = 1e-13
        end
    end
    # spherical shell over the icosahedral mesh: closed cells, shared faces, the volume of the shell
    a, d = 1.0, 0.1
    shell(level) = extruded_complex(ExtrudedPlex(create_spherical_mesh(level), 1), [0.0, d]; radius=a)
    cx = shell(3)
    g = PolyGeometry(cx)
    closure = map(1:num_poly_cells(cx)) do c
        faces, signs = PH.cell_cone(cx, c)
        norm(sum(s * g.area[f] * g.normal[f] for (f, s) in zip(faces, signs)))
    end
    @test maximum(closure) < 1e-15
    count_signs = zeros(Int, num_poly_faces(cx))
    for c in 1:num_poly_cells(cx)
        faces, signs = PH.cell_cone(cx, c)
        count_signs[faces] .+= signs
    end
    # lateral faces in two cells with opposite signs, top and bottom in one
    @test all(abs.(count_signs) .<= 1) && count(!iszero, count_signs) == 2 * num_poly_cells(cx)
    exact_volume = 4π / 3 * ((a + d)^3 - a^3)
    err(level) = abs(sum(PolyGeometry(shell(level)).volume) / exact_volume - 1)
    @test err(4) < err(3) / 3.5 && err(3) < 0.01
    @test all(c -> norm(g.cell_centroid[c]) > a && norm(g.cell_centroid[c]) < a + d, 1:num_poly_cells(cx))
    # hexagons on the sphere are not planar: the fan of planar_pieces, still closed cells
    hx = extruded_complex(ExtrudedPlex(create_hexagon_mesh(3), 1), [0.0, d]; radius=a)
    gh = PolyGeometry(hx)
    @test count(f -> length(PH.planar_pieces(PH.face_points(hx, f))) > 1, 1:num_poly_faces(hx)) > num_poly_cells(hx)
    @test maximum(1:num_poly_cells(hx)) do c
        faces, signs = PH.cell_cone(hx, c)
        norm(sum(s * gh.area[f] * gh.normal[f] for (f, s) in zip(faces, signs)))
    end < 1e-15
    @test abs(sum(gh.volume) / exact_volume - 1) < 0.01
    @test_throws ArgumentError extruded_complex(ExtrudedPlex(create_spherical_mesh(1), 1), [0.0, 1.0])
end

@testset "polyhedral: polynomial reconstruction and transport" begin
    PH = dmplex.Polyhedral
    # distorted triangles on the unit square, five layers: the face values reproduce polynomials
    rng = Random.MersenneTwister(3)
    n = 5
    id(i, j) = (j - 1) * (n + 1) + i
    jitter(i) = i == 1 || i == n + 1 ? 0.0 : 0.3 * (rand(rng) - 0.5)
    pts = [SVector((i - 1 + jitter(i)) / n, (j - 1 + jitter(j)) / n, 0.0) for j in 1:(n + 1) for i in 1:(n + 1)]
    polys = reduce(vcat, [[[id(i, j), id(i + 1, j), id(i + 1, j + 1)], [id(i, j), id(i + 1, j + 1), id(i, j + 1)]]
                          for j in 1:n for i in 1:n])
    cx = extruded_complex(ExtrudedPlex(plex_from_polygons(polys, pts), 5), [0.0, 0.1, 0.25, 0.35, 0.42, 0.55])
    g = PolyGeometry(cx)
    fc = face_cells(cx)
    o = zero(SVector{3, Float64})
    for r in 1:3
        rec = PolyReconstruction(cx, g, r)
        Q1, Q2 = face_value_matrices(cx, rec)
        B = MonomialBasis(r)
        avg(I) = sum(I[j] * (1 + j) for j in eachindex(I)) / I[1]       # q = Σ (1 + j) x^{α_j}
        ū = [avg(cell_moments(cx, c, B; center=o, scale=1.0)) for c in 1:num_poly_cells(cx)]
        exact = [avg(face_moments(cx, f, B; center=o, scale=1.0)) for f in 1:num_poly_faces(cx)]
        @test all(f -> fc[1, f] == 0 || abs((Q1 * ū)[f] - exact[f]) < 1e-10, 1:num_poly_faces(cx))
        @test all(f -> fc[2, f] == 0 || abs((Q2 * ū)[f] - exact[f]) < 1e-10, 1:num_poly_faces(cx))
        x = g.cell_centroid[17] + SVector(0.01, -0.02, 0.005)
        @test reconstruct(rec, 17, ū, x) ≈ sum(prod(x .^ α) * (1 + j) for (j, α) in enumerate(B.exponents))
    end
    # one layer on the sphere: the tangential basis reproduces polynomials of the tangent plane
    sh = extruded_complex(ExtrudedPlex(create_hexagon_mesh(3), 1), [0.0, 0.01]; radius=1.0)
    gs = PolyGeometry(sh)
    rec = PolyReconstruction(sh, gs, 2; basis=:tangential)
    @test all(R -> size(R, 1) == 6, rec.coefficients)
    c = 100
    Q, xc, h = rec.rotation[c], rec.center[c], rec.scale[c]
    p(x) = (y = Q * (x - xc) / h; 1 + y[1] - 2y[2] + 3y[1] * y[2] + y[1]^2)
    B2 = MonomialBasis(2)
    coef = Dict((0, 0, 0) => 1.0, (1, 0, 0) => 1.0, (0, 1, 0) => -2.0, (1, 1, 0) => 3.0, (2, 0, 0) => 1.0)
    ū = map(1:num_poly_cells(sh)) do s
        I = cell_moments(sh, s, B2; center=xc, scale=h, rotation=Q)
        sum(get(coef, α, 0.0) * I[j] for (j, α) in enumerate(B2.exponents)) / I[1]
    end
    @test reconstruct(rec, c, ū, gs.cell_centroid[c]) ≈ p(gs.cell_centroid[c])
    # transport by solid-body rotation: divergence-free fluxes, mass to round-off
    Ω = 2π * normalize(SVector(1.0, 0.0, 1.0))
    U = stream_function_flux(sh, r̂ -> dot(Ω, r̂))
    fcs = face_cells(sh)
    div = zeros(num_poly_cells(sh))
    for f in eachindex(U)
        fcs[1, f] > 0 && (div[fcs[1, f]] += U[f])
        fcs[2, f] > 0 && (div[fcs[2, f]] -= U[f])
    end
    @test maximum(abs, div) < 1e-14 * maximum(abs, U)
    m = PolyAdvection(sh, U; degree=2, basis=:tangential)
    q = [exp(-sum(abs2, normalize(x) - SVector(1.0, 0, 0)) / 0.16) for x in m.geometry.cell_centroid]
    mass = sum(m.geometry.volume .* q)
    dt = 0.4 * upwind_time_step(m)
    cache = SSPRK3Cache(q)
    for _ in 1:20
        ssprk3_step!(q, dt, PH.tendency!, cache, m)
    end
    @test abs(sum(m.geometry.volume .* q) / mass - 1) < 1e-13
    # steady flow in the unit square (q = exp(2ψ) is steady for u = ∇ψ × ẑ): the fluxes at the
    # quadrature points of the faces give the order of the reconstruction, degree 3: 4
    ψ(x) = sin(π * x[1]) * sin(π * x[2]) / π
    u(x) = SVector(sin(π * x[1]) * cos(π * x[2]), -cos(π * x[1]) * sin(π * x[2]), 0.0)
    function steady_error(nx)
        rng = Random.MersenneTwister(1)
        id(i, j) = (j - 1) * (nx + 1) + i
        jit(i) = i == 1 || i == nx + 1 ? 0.0 : 0.3 * (rand(rng) - 0.5)
        pts = [SVector((i - 1 + jit(i)) / nx, (j - 1 + jit(j)) / nx, 0.0) for j in 1:(nx + 1) for i in 1:(nx + 1)]
        polys = reduce(vcat, [[[id(i, j), id(i + 1, j), id(i + 1, j + 1)], [id(i, j), id(i + 1, j + 1), id(i, j + 1)]]
                              for j in 1:nx for i in 1:nx])
        cx = extruded_complex(ExtrudedPlex(plex_from_polygons(polys, pts), 1), [0.0, 0.1])
        m = PolyAdvection(cx, u; degree=3, basis=:tangential, up=x -> SVector(0.0, 0.0, 1.0))
        # cell averages of exp(2ψ) by the degree-5 rule on the four subtriangles of the base triangle
        bary, w = PH.TRIANGLE_RULE
        q = map(1:num_poly_cells(cx)) do c
            a, b, d = [x for x in cx.coordinates[PH.cell_vertices(cx, c)] if x[3] == 0.0]
            sub = ((a, (a + b) / 2, (a + d) / 2), ((a + b) / 2, b, (b + d) / 2), ((a + d) / 2, (b + d) / 2, d),
                   ((a + b) / 2, (b + d) / 2, (a + d) / 2))
            sum(sum(wk * exp(2ψ(λ[1] * t[1] + λ[2] * t[2] + λ[3] * t[3])) for (λ, wk) in zip(bary, w)) for t in sub) / 4
        end
        q0, V = copy(q), m.geometry.volume
        nst = ceil(Int, 0.5 / (0.4 * upwind_time_step(m)))
        cache = SSPRK3Cache(q)
        for _ in 1:nst
            ssprk3_step!(q, 0.5 / nst, PH.tendency!, cache, m)
        end
        return sqrt(sum(V .* (q .- q0) .^ 2) / sum(V .* q0 .^ 2))
    end
    @test log2(steady_error(8) / steady_error(16)) > 3.3
end

@testset "polyhedral: interaction regions and shallow water" begin
    PH = dmplex.Polyhedral
    rng = Random.MersenneTwister(3)
    n = 5
    id(i, j) = (j - 1) * (n + 1) + i
    jit(i) = i == 1 || i == n + 1 ? 0.0 : 0.3 * (rand(rng) - 0.5)
    pts = [SVector((i - 1 + jit(i)) / n, (j - 1 + jit(j)) / n, 0.0) for j in 1:(n + 1) for i in 1:(n + 1)]
    polys = reduce(vcat, [[[id(i, j), id(i + 1, j), id(i + 1, j + 1)], [id(i, j), id(i + 1, j + 1), id(i, j + 1)]]
                          for j in 1:n for i in 1:n])
    cx = extruded_complex(ExtrudedPlex(plex_from_polygons(polys, pts), 4), [0.0, 0.1, 0.25, 0.32, 0.45])
    g = PolyGeometry(cx)
    G, T = interaction_operators(cx, g)
    fc = face_cells(cx)
    nf = num_poly_faces(cx)
    # linear fields: exact at the faces whose vertices are inside the domain
    boundary = Set(v for f in 1:nf if fc[1, f] == 0 || fc[2, f] == 0 for v in PH.face_loop(cx, f))
    inner = [f for f in 1:nf if all(v -> v ∉ boundary, PH.face_loop(cx, f))]
    @test length(inner) > 50
    a = SVector(1.3, -0.7, 2.1)
    u = [1 + dot(a, x) for x in g.cell_centroid]
    @test maximum(abs, (G * u)[inner] .- [dot(a, g.normal[f]) for f in inner]) < 1e-12
    @test maximum(abs, (T * u)[inner] .- [1 + dot(a, g.face_centroid[f]) for f in inner]) < 1e-12
    @test maximum(abs, G * ones(num_poly_cells(cx))) < 1e-12
    # constant vectors from their normal components and back
    vc = cell_vectors(cx, g, [dot(a, g.normal[f]) for f in 1:nf])
    @test maximum(v -> norm(v - a), vc) < 1e-12
    @test maximum(abs, face_normal_components(cx, g, vc) .- [dot(a, g.normal[f]) for f in 1:nf]) < 1e-12
    # Williamson et al. (1992) case 2 for one day on one layer of prisms over hexagons
    R, Ω, grav = 6.37122e6, 7.292e-5, 9.80616
    u0, gh0 = 2π * R / (12 * 86400), 2.94e4
    hexact(x) = (r = normalize(x); (gh0 - (R * Ω * u0 + u0^2 / 2) * r[3]^2) / grav)
    uexact(x) = u0 * cross(SVector(0.0, 0.0, 1.0), normalize(x))
    # hexagons (Perot's cell vectors) and triangles (the weighted linear fit)
    for (mesh, perot) in ((create_hexagon_mesh(3), true), (create_spherical_mesh(3), false))
        sh = extruded_complex(ExtrudedPlex(mesh, 1), [0.0, 1000.0]; radius=R)
        m = PolyShallowWater(sh; gravity=grav, omega=Ω, degree=2)
        @test (m.cell_vector == :perot) == perot
        geo = m.geometry
        y = initial_state(m, uexact, hexact)
        h0, mass0 = [hexact(x) for x in geo.cell_centroid], total_mass(m, y)
        steps = ceil(Int, 86400 / shallow_water_time_step(m, y))
        cache = SSPRK3Cache(y)
        for _ in 1:steps
            ssprk3_step!(y, 86400 / steps, tendency!, cache, m)
        end
        h = y[1:num_poly_cells(sh)]
        @test sqrt(sum(geo.volume .* (h .- h0) .^ 2) / sum(geo.volume .* h0 .^ 2)) < 2e-3
        @test abs(total_mass(m, y) / mass0 - 1) < 1e-13
        @test @allocated(tendency!(cache.k, y, m)) < 10_000
    end
    # 1 and 3 ranks give the same state (gathers only, no atomics: bitwise)
    script = joinpath(@__DIR__, "mpi_poly.jl")
    project = dirname(@__DIR__)
    for mesh in ("triangles", "hexagons", "cubed")
        results = map((1, 3)) do n
            file = tempname()
            run(`$(MPI.mpiexec()) -n $n $(Base.julia_cmd()) --project=$project $script $file $mesh`)
            lines = readlines(file)
            parse(Float64, split(lines[1])[1]), parse.(Float64, lines[2:end])
        end
        @test results[2][2] == results[1][2]
        @test abs(results[2][1]) < 1e-13
    end
    @test_throws ArgumentError PolyShallowWater(cx; cell_vector=:rbf)
end
