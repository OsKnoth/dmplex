@testset "Operators" begin
    for dm in (create_spherical_mesh(1), create_cubed_sphere_mesh(2)), k in 0:2
        U, H = FESpace(dm, RaviartThomas(k)), FESpace(dm, Discontinuous(k))
        D = assemble_divergence(H, U)
        @test size(D) == (num_dofs(H), num_dofs(U))
        # DG bases sum to one, so the column sums are ∫ ∇⋅φ_j = 0 on the closed sphere
        @test maximum(abs, sum(D; dims=1)) < 1e-12
        C = assemble_coriolis(U, x -> 2x[3])
        @test norm(C + C', Inf) < 1e-12
        @test norm(assemble_bilinear((t, u, a, b, q) -> dot(t.φ[a, q], u.φ[b, q]) * t.dA[q], U, U) -
                   assemble_mass(U), Inf) < 1e-12
    end
    # L2 projection reproduces constants in DG and Lagrange spaces
    dm = create_cubed_sphere_mesh(2)
    @test project(FESpace(dm, Discontinuous(1)), x -> 1.0) ≈ ones(4 * num_cells(dm))
    @test project(FESpace(dm, Lagrange(2)), x -> 1.0) ≈ ones(num_dofs(FESpace(dm, Lagrange(2))))
end

@testset "RK4" begin
    for dt in (0.1, 0.05)
        y = [1.0, 0.0]
        rk4_step!(y, dt, (dy, y) -> (dy .= (y[2], -y[1])), RK4Cache(y))
        @test abs(y[1] - cos(dt)) < dt^5 / 100
    end
end

@testset "Linear shallow water: $name, k = $k" for (name, dm) in (("triangles", create_spherical_mesh(1)),
                                                                 ("quadrilaterals", create_cubed_sphere_mesh(3))),
                                                    k in 0:2
    # steady geostrophic balance: u = u0 ẑ × x, f = 2z, g h = -u0 z²
    u0 = 0.5
    m = LinearShallowWater(dm, k)
    y = initial_state(m, x -> u0 * SVector(-x[2], x[1], 0.0), x -> -u0 * x[3]^2)
    y0 = copy(y)
    mass0, energy0 = total_mass(m, y), total_energy(m, y)
    cache = RK4Cache(y)
    dt = 0.1 * sqrt(4π / num_cells(dm)) / (k + 1)^2
    for _ in 1:20
        rk4_step!(y, dt, tendency!, cache, m)
    end
    @test abs(total_mass(m, y) - mass0) < 1e-13
    @test abs(total_energy(m, y) - energy0) / energy0 < 1e-7
    dh = (y - y0)[height_dofs(m)]
    @test sqrt(dot(dh, m.Mh * dh)) < (k == 2 ? 1e-3 : 5e-2)
end
