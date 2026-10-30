@testset "extruded plex" begin
    for (name, mesh, levels) in MESHES, n in levels[1:2]
        dm = mesh(n)
        ext = ExtrudedPlex(dm, 3)
        @testset "$name $n" begin
            counts = [count(_ -> true, get_depth_stratum(ext, d)) for d in 0:3]
            @test counts[4] == 3 * num_cells(dm)
            @test counts[1] == 4 * num_vertices(dm)
            @test counts[1] - counts[2] + counts[3] - counts[4] == 2      # χ(S² × I)
            B1, B2, B3 = (boundary_matrix(ext, d) for d in 1:3)
            @test iszero(B1 * B2)
            @test iszero(B2 * B3)
            @test all(q -> extruded_point(ext, base_point(ext, q)...) == q, get_chart(ext))
            @test all(q -> all(r -> q in get_support(ext, r), get_cone(ext, q)), get_chart(ext))
            c = first(cell_range(dm))
            cell = extruded_point(ext, c, 2, :layer)
            @test get_depth(ext, cell) == 3
            @test get_cone(ext, cell)[1:2] == [extruded_point(ext, c, 2, :level), extruded_point(ext, c, 3, :level)]
            @test length(get_cone(ext, cell)) == 2 + get_cone_size(dm, c)
        end
    end
end

@testset "nonhydrostatic TRiSK" begin
    g = TRiSKGrid(create_hexagon_mesh(3))
    v = VerticalGrid(10, 30000.0; stretch=3)
    m = TRiSKNonhydrostatic(g, v; damping_height=22000.0, damping_rate=0.05)
    mass(y) = dmplex.Nonhydrostatic.total_mass(m, y)
    # a horizontally homogeneous column in discrete hydrostatic balance stays at rest
    rest(λ, φ, z) = (baroclinic_wave(0.0, 0.0, z; perturbation=false)[1:2]..., 0.0, 0.0)
    y = initial_state(m, rest)
    M0 = mass(y)
    ys = zero_state(m)
    for _ in 1:5
        step!(m, y, 300.0, ys)
    end
    @test maximum(abs, y.vn) == 0
    @test maximum(abs, y.w) < 1e-12
    @test mass(y) ≈ M0 rtol = 1e-14
    # the unperturbed baroclinic wave is a steady state: small changes, mass conserved
    y = initial_state(m, (λ, φ, z) -> baroclinic_wave(λ, φ, z; perturbation=false))
    vn0, M0 = copy(y.vn), mass(y)
    for _ in 1:12
        step!(m, y, 300.0, ys)
    end
    @test maximum(abs, y.vn .- vn0) < 2
    @test maximum(abs, y.w) < 0.05
    @test mass(y) ≈ M0 rtol = 1e-13
end
