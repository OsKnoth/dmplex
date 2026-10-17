const SHAPES = (Triangle(), Quadrilateral())

families_up_to(k) = [[Lagrange(p) for p in 1:k]; [Discontinuous(p) for p in 0:k];
                     [RaviartThomas(p) for p in 0:k]; [Nedelec(p) for p in 0:k]]

@testset "Quadrature" begin
    for n in 1:8
        x, w = gauss_legendre(n)
        @test sum(w) ≈ 2
        @test sum(w .* x .^ (2n - 2)) ≈ 2 / (2n - 1)
    end
    for n in 2:8
        x, w = gauss_lobatto(n)
        @test x[1] == -1 && x[end] == 1 && issorted(x)
        @test sum(w .* x .^ (2n - 4)) ≈ 2 / (2n - 3)
    end
    for shape in SHAPES, degree in 0:8
        rule = quadrature_rule(shape, degree)
        @test sum(rule.weights) ≈ reference_area(shape)
        # ∫ (1 + x)^a (1 + y)^b over the reference cell
        a, b = degree ÷ 2, degree - degree ÷ 2
        exact = shape isa Quadrilateral ? 2^(a + 1) / (a + 1) * 2^(b + 1) / (b + 1) :
                2^(a + b + 2) * factorial(a) * factorial(b) / factorial(a + b + 2)
        @test sum(w * (1 + ξ[1])^a * (1 + ξ[2])^b for (ξ, w) in zip(rule.points, rule.weights)) ≈ exact
    end
end

@testset "Reference elements: $family on $shape" for family in families_up_to(4), shape in SHAPES
    el = ReferenceElement(family, shape)
    nv = length(reference_vertices(shape))
    @test entity_dofs(el) == (family(2, nv), family(1, 2), family(0, 0))
    @test num_dofs(el) == sum(entity_dofs(el) .* (1, nv, nv))

    if function_space(family) in (:H1, :L2)
        values, _ = tabulate(el, el.nodes)
        @test values[:, 1, :] ≈ I atol=1e-10
        points = [SVector(-0.9, -0.8), SVector(-0.2, 0.1), SVector(0.3, -0.6)]
        @test vec(sum(first(tabulate(el, points)); dims=1)) ≈ ones(3)  # partition of unity
    end

    # traces: on edge i only the dofs of edge i (and its two vertices) are seen
    first_edge_dof = entity_dofs(el)[1]
    for i in 1:nv, t in (-0.83, -0.1, 0.47)
        values, _ = tabulate(el, [edge_point(shape, i, t)])
        owned = collect(first_edge_dof + (i - 1) * entity_dofs(el)[2] .+ (1:entity_dofs(el)[2]))
        if function_space(family) == :H1
            vertex_base = first_edge_dof + nv * entity_dofs(el)[2]
            append!(owned, vertex_base .+ (i, mod1(i + 1, nv)))
            trace = values[:, 1, 1]
        elseif function_space(family) == :Hdiv
            trace = values[:, :, 1] * edge_normal(shape, i)
        elseif function_space(family) == :Hcurl
            trace = values[:, :, 1] * edge_tangent(shape, i)
        else
            continue
        end
        @test all(abs.(trace[setdiff(1:num_dofs(el), owned)]) .< 1e-10)
    end
end

@testset "Sphere map" begin
    for dm in (create_spherical_mesh(1), create_cubed_sphere_mesh(3))
        rule = Dict(n => quadrature_rule(reference_shape(n), 12) for n in (3, 4))
        area = 0.0
        for c in cell_range(dm)
            cmap = CellMap(dm, c)
            r = rule[get_cone_size(dm, c)]
            area += sum(w * map_point(cmap, ξ)[3] for (ξ, w) in zip(r.points, r.weights))
            x, J, _ = map_point(cmap, SVector(-0.2, -0.3))
            @test norm(x) ≈ 1
            @test norm(J' * x) < 1e-12  # Jacobian columns are tangent
            @test dot(cross(J[:, 1], J[:, 2]), x) > 0  # counter-clockwise from outside
        end
        @test area ≈ 4π rtol=1e-7
    end
end

@testset "Global conformity on the sphere" begin
    Random.seed!(42)
    for dm in (create_spherical_mesh(1), create_cubed_sphere_mesh(2))
        for family in [Lagrange(1), Lagrange(3), RaviartThomas(0), RaviartThomas(2), Nedelec(0), Nedelec(2)]
            V = FESpace(dm, family)
            u = randn(num_dofs(V))
            space = function_space(V)
            jump = 0.0
            for e in edge_range(dm)
                left, right = get_edge_cells(dm, e)
                il, ir = findfirst(==(e), get_cone(dm, left)), findfirst(==(e), get_cone(dm, right))
                p1, p2 = (get_coordinates(dm, v) for v in get_cone(dm, e))
                normal = normalize(cross(p1, p2))
                for t in (-0.7, 0.3, 0.9)
                    # the right cell traverses the shared edge in the opposite direction
                    xl, ul = evaluate(V, u, left, edge_point(reference_shape(dm, left), il, t))
                    xr, ur = evaluate(V, u, right, edge_point(reference_shape(dm, right), ir, -t))
                    @test xl ≈ xr atol=1e-14
                    jump = max(jump, space == :H1 ? abs(ul - ur) :
                                     space == :Hdiv ? abs(dot(ul - ur, normal)) :
                                     abs(dot(ul - ur, cross(normal, xl))))
                end
            end
            @test jump < 1e-12
        end
    end
end

@testset "Mass matrices" begin
    for dm in (create_spherical_mesh(1), create_cubed_sphere_mesh(2)), family in families_up_to(2)
        M = assemble_mass(FESpace(dm, family))
        @test norm(M - M', Inf) < 1e-12
        @test isposdef(Symmetric(Matrix(M)))
        if function_space(family) in (:H1, :L2)
            @test sum(M) ≈ 4π rtol=1e-3
        end
    end
    # vector-valued Lagrange: one scalar block per Cartesian component
    dm = create_cubed_sphere_mesh(2)
    M1 = assemble_mass(FESpace(dm, Lagrange(2)))
    M3 = assemble_mass(FESpace(dm, Lagrange(2); ncomp=3))
    @test M3[1:3:end, 1:3:end] ≈ M1
    @test iszero(M3[1:3:end, 2:3:end])
end

@testset "Discrete divergence theorem" begin
    Random.seed!(7)
    for dm in (create_spherical_mesh(1), create_cubed_sphere_mesh(2)), k in 0:2
        V = FESpace(dm, RaviartThomas(k))
        u = randn(num_dofs(V))
        total = 0.0
        for c in cell_range(dm)
            el = reference_element(V, c)
            rule = quadrature_rule(el.shape, 2k + 2)
            _, grads = tabulate(el, rule.points)
            indices, signs = cell_dofs(V, c)
            # Piola: div u dA = div̂ φ̂ dξ, so the reference divergence integrates directly
            for (q, w) in enumerate(rule.weights), a in eachindex(indices)
                total += w * signs[a] * u[indices[a]] * (grads[a, 1, 1, q] + grads[a, 2, 2, q])
            end
        end
        @test abs(total) < 1e-10
    end
end

@testset "Mixed spaces share one section" begin
    dm = create_cubed_sphere_mesh(2)
    s = Section(dm, raviart_thomas(1; name="u"), discontinuous(1; name="h"); ordering=:field)
    U, H = FESpace(dm, s, "u"), FESpace(dm, s, "h")
    c = first(cell_range(dm))
    @test all(i -> i in field_range(s, 1), first(cell_dofs(U, c)))
    @test all(i -> i in field_range(s, 2), first(cell_dofs(H, c)))
    @test num_dofs(reference_element(U, c)) == length(first(cell_dofs(U, c)))
    @test_throws ArgumentError FESpace(create_hexagon_mesh(1), Lagrange(1))
end
