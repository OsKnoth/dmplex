@testset "Metrics and upwind advection: $name" for (name, builder, resolutions) in MESHES
    for resolution in resolutions
        dm = builder(resolution)
        metrics = compute_mesh_metrics(dm)
        tracer = init_tracer(dm, metrics)
        velocity = init_edge_velocity(dm, metrics)
        rhs = compute_advection_rhs!(similar(tracer), tracer, dm, metrics, velocity)

        @test all(>(0.0), metrics.cell_areas)
        @test sum(metrics.cell_areas) ≈ 4π atol=1e-10
        @test all(n -> norm(n) ≈ 1, metrics.edge_normals)
        @test all(i -> abs(dot(metrics.edge_normals[i], metrics.edge_midpoints[i])) < 1e-12, eachindex(metrics.edge_normals))
        # the edge normal points from the left cell towards the right cell
        @test all(edge_range(dm)) do e
            left, right = get_edge_cells(dm, e)
            i = edge_index(dm, e)
            dot(metrics.edge_normals[i],
                metrics.cell_centers[cell_index(dm, right)] - metrics.cell_centers[cell_index(dm, left)]) > 0
        end
        @test sum(rhs .* metrics.cell_areas) ≈ 0.0 atol=1e-12
        @test all(c -> 0 <= c <= 1, tracer)
    end
end

@testset "TRiSK grid and operators" begin
    dm = create_hexagon_mesh(3)
    g = TRiSKGrid(dm)
    m = TRiSKShallowWater(g)
    nc, ne, nv = num_cells(dm), num_edges(dm), num_vertices(dm)
    @test sum(g.cell_areas) ≈ 4π && sum(g.vertex_areas) ≈ 4π
    @test all(i -> sum(g.cell_kites[:, i]) ≈ g.cell_areas[i], 1:nc)   # kite fractions add up to one
    @test all(e -> dot(g.edge_normals[e], g.centers[g.edge_cells[2, e]] - g.centers[g.edge_cells[1, e]]) > 0, 1:ne)

    # antisymmetric weights: the PV flux term does no work
    W = zeros(ne, ne)
    for e in 1:ne, k in axes(g.trisk_edges, 1)
        W[e, g.trisk_edges[k, e]] += g.trisk_weights[k, e]
    end
    @test maximum(abs, W + W') < 1e-14

    Random.seed!(5)
    u = randn(ne)
    cell_sum(x, weights) = [sum(weights[k, i] * x[g.cell_edges[k, i]] for k in axes(g.cell_edges, 1)) for i in 1:nc]
    D(F) = cell_sum(F, g.cell_signs .* g.primal_lengths[g.cell_edges]) ./ g.cell_areas
    # TRiSK property: the dual divergence of the perpendicular fluxes is the kite average of the divergence
    perp = tangential_velocity(m, u) .* g.dual_lengths
    dual_div = [sum(g.vertex_signs[k, v] * perp[g.vertex_edges[k, v]] for k in axes(g.vertex_edges, 1))
                for v in 1:nv] ./ g.vertex_areas
    Du = D(u)
    average = [sum(g.vertex_kites[k, v] * Du[g.vertex_cells[k, v]] for k in axes(g.vertex_cells, 1))
               for v in 1:nv] ./ g.vertex_areas
    @test maximum(abs, dual_div + average) < 1e-12 * maximum(abs, average)
    # curl ∘ gradient = 0 and divergence ∘ skew gradient = 0
    φ, ψ = randn(nc), randn(nv)
    gradient = [(φ[g.edge_cells[2, e]] - φ[g.edge_cells[1, e]]) / g.dual_lengths[e] for e in 1:ne]
    @test maximum(abs, relative_vorticity(m, [gradient; zeros(nc)])) < 1e-12
    skew = [(ψ[g.edge_vertices[2, e]] - ψ[g.edge_vertices[1, e]]) / g.primal_lengths[e] for e in 1:ne]
    @test maximum(abs, D(skew)) < 1e-11
    # curl of solid-body rotation, tangential reconstruction (TRiSK is not consistent in the max norm)
    ω = SVector(0.3, -0.2, 1.0)
    solid = [dot(cross(ω, x), n) for (x, n) in zip(g.edge_points, g.edge_normals)]
    @test maximum(abs, relative_vorticity(m, [solid; zeros(nc)]) - [2dot(ω, x) for x in g.vertices]) < 0.03
    tangential = [dot(cross(ω, x), cross(x, n)) for (x, n) in zip(g.edge_points, g.edge_normals)]
    @test maximum(abs, tangential_velocity(m, solid) - tangential) < 0.15

    # Gassmann's reconstruction: antisymmetric with the same PV, and consistent
    E, P, Wg = gassmann_coefficients(g)
    l, d = g.primal_lengths, g.dual_lengths
    @test all(CartesianIndices(E)) do I
        k, e = Tuple(I)
        f = E[k, e]
        j = findfirst(j -> E[j, f] == e && P[j, f] == P[k, e], axes(E, 1))
        j !== nothing && abs(l[e] * d[e] * Wg[k, e] + l[f] * d[f] * Wg[j, f]) < 1e-15
    end
    gassmann = [sum(Wg[k, e] * solid[E[k, e]] for k in axes(E, 1)) for e in 1:ne]
    @test maximum(abs, gassmann - tangential) < 0.02
end

@testset "TRiSK shallow water: Williamson test case 2" begin
    u0 = 0.2
    velocity(x) = u0 * SVector(-x[2], x[1], 0.0)
    height(x) = 1 - (u0 + u0^2 / 2) * x[3]^2
    errors = map(Iterators.product((3, 4), (:trisk, :gassmann, :gassmann_mean))) do (n, pv_flux)
        dm = create_hexagon_mesh(n)
        m = TRiSKShallowWater(dm; pv_flux)
        y = initial_state(m, velocity, height)
        y0 = copy(y)

        # the semi-discretization conserves energy
        Random.seed!(3)
        z = y .+ 0.01 .* randn(length(y))
        dz = tendency!(similar(z), z, m)
        ε = 1e-6
        dEdt = (total_energy(m, z .+ ε .* dz) - total_energy(m, z .- ε .* dz)) / 2ε
        @test abs(dEdt) < 1e-8 * norm(dz)

        mass0, energy0 = total_mass(m, y), total_energy(m, y)
        dt = 0.2 * sqrt(4π / num_cells(dm))
        cache = RK4Cache(y)
        for _ in 1:ceil(Int, 1 / dt)
            rk4_step!(y, dt, tendency!, cache, m)
        end
        @test abs(total_mass(m, y) - mass0) < 1e-13
        @test abs(total_energy(m, y) - energy0) / energy0 < 1e-8
        dh = (y - y0)[height_dofs(m)]
        sqrt(sum(m.grid.cell_areas .* dh .^ 2) / 4π)
    end
    @test all(<(2e-3), errors[1, :])
    @test all(errors[2, :] .< errors[1, :] ./ 2.5)   # about second order in the L2 norm
end

@testset "ICON triangle shallow water" begin
    g = TRiSKGrid(create_spherical_mesh(3))   # circumcenters
    @test dmplex.orthogonality_defect(g.dm, g.centers) < 1e-12
    # RBF tangential velocity: exact for constant vectors along r̂ × n_e by construction, consistent
    E, C = rbf_coefficients(g)
    ω = SVector(0.3, -0.2, 1.0)
    solid = [dot(cross(ω, x), n) for (x, n) in zip(g.edge_points, g.edge_normals)]
    tangential = [dot(cross(ω, x), cross(x, n)) for (x, n) in zip(g.edge_points, g.edge_normals)]
    @test maximum(abs, [sum(C[k, e] * solid[E[k, e]] for k in 1:4) for e in 1:num_edges(g)] - tangential) < 0.01
    # bilinear weights: partition of unity, no first moment
    B = dmplex.FiniteVolumes.bilinear_weights(g)
    @test all(sum(B; dims=1) .≈ 1)
    @test all(i -> norm(cross(g.centers[i], sum(B[k, i] * g.edge_points[g.cell_edges[k, i]] for k in 1:3))) < 1e-14,
              1:num_cells(g))

    u0 = 0.2
    velocity(x) = u0 * SVector(-x[2], x[1], 0.0)
    height(x) = 1 - (u0 + u0^2 / 2) * x[3]^2
    errors = map((3, 4)) do n
        m = ICONShallowWater(create_spherical_mesh(n))
        y = initial_state(m, velocity, height)
        y0 = copy(y)
        mass0, energy0 = total_mass(m, y), total_energy(m, y)
        dt = 0.2 * minimum(m.grid.dual_lengths)
        cache = RK4Cache(y)
        for _ in 1:ceil(Int, 1 / dt)
            rk4_step!(y, dt, tendency!, cache, m)
        end
        @test abs(total_mass(m, y) - mass0) < 1e-13
        @test abs(total_energy(m, y) - energy0) / energy0 < 1e-5
        dh = (y - y0)[height_dofs(m)]
        sqrt(sum(m.grid.cell_areas .* dh .^ 2) / 4π)
    end
    @test errors[1] < 1e-4
    @test errors[2] < errors[1] / 3
end

@testset "TRiSK distribution (MPI)" begin
    # ownership without MPI: every cell and edge owned exactly once, owned edges have an owned cell
    g = TRiSKGrid(create_hexagon_mesh(3))
    part = equal_area_partition(g.dm, 3; centers=g.centers)
    distributions = [TRiSKDistribution(g, part, 3, rank) for rank in 0:2]
    for (field, n) in ((:cells, num_cells(g)), (:edges, num_edges(g)))
        ids = reduce(vcat, (getfield(d, field).global_ids[owned(getfield(d, field))] for d in distributions))
        @test sort(ids) == 1:n
    end
    @test all(distributions) do d
        all(e -> any(part[g.edge_cells[s, e]] == d.rank + 1 for s in 1:2), d.edges.global_ids[owned(d.edges)])
    end
    @test sum(d.n_owned_vertices for d in distributions) == num_vertices(g)

    # 1 and 3 ranks give the same state (gathers only, no atomics: bitwise)
    script = joinpath(@__DIR__, "mpi_trisk.jl")
    project = dirname(@__DIR__)
    for pv_flux in ("trisk", "gassmann", "gassmann_mean", "icon")
        results = map((1, 3)) do n
            file = tempname()
            run(`$(MPI.mpiexec()) -n $n $(Base.julia_cmd()) --project=$project $script $file $pv_flux`)
            lines = readlines(file)
            parse.(Float64, split(lines[1])), parse.(Float64, lines[2:end])
        end
        ((dmass1, denergy1, enstrophy1, _), y1), ((dmass3, denergy3, enstrophy3, _), y3) = results
        @test y3 == y1
        @test abs(dmass3) < 1e-13
        @test abs(denergy3) < (pv_flux == "icon" ? 1e-6 : 1e-9)   # ICON's scheme does not conserve energy
        @test abs(denergy3 - denergy1) < 1e-14     # summation order of the reduction only
        @test enstrophy3 ≈ enstrophy1
    end
end
