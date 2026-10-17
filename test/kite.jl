@testset "Kite elements" begin
    x, w = gauss_radau(4)
    @test x[end] == 1 && issorted(x) && sum(w) ≈ 2
    @test sum(w .* x .^ 6) ≈ 2 / 7  # exact for degree 2n - 2
    for k in 0:3
        el = ReferenceElement(KiteRadau(k), Quadrilateral())
        @test num_dofs(el) == (k + 1)^2
        values, _ = tabulate(el, el.nodes)
        @test values[:, 1, :] ≈ I atol=1e-10
        @test sum(el.nodal_weights) ≈ 4
        # no nodes on the primal side (x = -1 or y = -1)
        @test all(ξ -> ξ[1] > -1 && ξ[2] > -1, el.nodes)
        @test num_dofs(ReferenceElement(KiteBrokenRT(k), Quadrilateral())) == 2 * (k + 1) * (k + 2)
    end
    @test_throws ArgumentError ReferenceElement(KiteRadau(1), Triangle())
end

"Largest jump of a random field of `V` across kite edges of the given type."
function kite_jump(V, edge_type)
    dm = V.dm
    u = randn(num_dofs(V))
    space = function_space(V)
    jump = 0.0
    for e in edge_range(dm)
        get_label_value(dm, "kite_edge_type", e) == edge_type || continue
        left, right = get_edge_cells(dm, e)
        il, ir = findfirst(==(e), get_cone(dm, left)), findfirst(==(e), get_cone(dm, right))
        a, b = (get_coordinates(dm, v) for v in get_cone(dm, e))
        n = normalize(cross(a, b))
        for t in (-0.6, 0.2, 0.7)
            _, ul = evaluate(V, u, left, edge_point(Quadrilateral(), il, t))
            _, ur = evaluate(V, u, right, edge_point(Quadrilateral(), ir, -t))
            jump = max(jump, space == :Hdiv ? abs(dot(ul - ur, n)) : abs(ul - ur))
        end
    end
    return jump
end

@testset "Kite spaces and operators" begin
    Random.seed!(11)
    for primal in (create_hexagon_mesh(1), create_cubed_sphere_mesh(2)), k in 0:2
        dm = create_kite_mesh(primal)
        H, U = FESpace(dm, KiteRadau(k)), FESpace(dm, KiteBrokenRT(k))
        @test num_dofs(H) == sum(1 + get_cone_size(primal, c) * (k + k^2) for c in cell_range(primal))
        # h: continuous inside primal cells, broken across primal edges (unless constant per cell)
        @test kite_jump(H, 2) < 1e-12
        k > 0 && @test kite_jump(H, 1) > 1e-3
        # u: normal-continuous across primal half edges, broken across dual-cell boundaries
        @test kite_jump(U, 1) < 1e-12
        @test kite_jump(U, 2) > 1e-3

        B = assemble_kite_divergence(H, U)
        G = assemble_kite_gradient(U, H)
        @test norm(G + B', Inf) < 1e-12
        @test norm(ones(num_dofs(H))' * B, Inf) < 1e-12

        # mass matrices: diagonal for h (exact for k = 0, Radau-lumped otherwise) and
        # for the vorticity (GLL-lumped), one block per dual cell for u
        @test block_size(mass_solver(H; lumped=k > 0)) == 1
        @test block_size(mass_solver(FESpace(dm, Lagrange(k + 1)); lumped=true)) == 1
        dual_cell_size = maximum(length(get_label_stratum(dm, "kite_dual_cell", vertex_index(primal, v)))
                                 for v in vertex_range(primal))
        @test block_size(mass_solver(U)) == dual_cell_size * (2 * (k + 1) + 2k * (k + 1)) + dual_cell_size * (k + 1)
    end
end
