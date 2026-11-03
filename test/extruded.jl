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

@testset "ICON-IAP momentum advection" begin
    NH = dmplex.Nonhydrostatic
    rng = Random.MersenneTwister(1)
    g = TRiSKGrid(create_hexagon_mesh(2))
    v = VerticalGrid(6, 10000.0; stretch=3)
    ne, nc, nv, nl = num_edges(g), num_cells(g), num_vertices(g), 6
    # the PV usage of i_cori_method = 4 is symmetric in (e, e')
    P = trisk_pv_edges(g)
    @test all(1:ne) do e
        all(k -> g.trisk_weights[k, e] == 0 ||
                     P[k, e] in P[findall(==(e), g.trisk_edges[:, g.trisk_edges[k, e]]), g.trisk_edges[k, e]],
            axes(P, 1))
    end
    Ae = g.primal_lengths .* g.dual_lengths     # weight of vn²/2 in the energy
    vn, w = randn(rng, ne, nl), randn(rng, nc, nl + 1)
    w[:, 1] .= 0
    w[:, end] .= 0
    ρ = 1 .+ 0.3 .* rand(rng, nc, nl)
    ρe = (g.cell_distances[1, :] .* ρ[g.edge_cells[1, :], :] .+ g.cell_distances[2, :] .* ρ[g.edge_cells[2, :], :]) ./
         g.dual_lengths
    ρh = zeros(nc, nl + 1)
    for j in 2:nl
        ρh[:, j] .= v.wlow[j] .* ρ[:, j - 1] .+ (1 - v.wlow[j]) .* ρ[:, j]
    end
    for (pv_flux, vertical_advection) in ((:iap, :iap), (:trisk, :vector_invariant))
        m = TRiSKNonhydrostatic(g, v; radius=1.0, pv_flux, vertical_advection, trisk_weights=:exact)
        W, mw = m.work, vertical_advection == :iap
        fill!(W.K, 0)
        fill!(W.π, 0)
        fill!(W.θ, 1)
        NH.launch!(m, NH.vertex_pv_kernel!, (nv, nl), W.qv, W.ζv, W.ρv, vn, ρ, m.vertex_edges, m.curl,
                   m.vertex_cells, m.vertex_weight, m.coriolis)
        NH.launch!(m, NH.edge_pv_kernel!, (ne, nl), W.qe, W.qv, W.ζv, W.ρv, m.edge_vertices, m.rhombus,
                   m.edge_coriolis, m.iap_pv)
        NH.launch!(m, NH.edge_density_kernel!, (ne, nl), W.ρe, W.Fs, vn, ρ, m.edge_cells, m.edge_weight)
        tendency(qe, w, η) = (NH.launch!(m, NH.vn_tendency_kernel!, (ne, nl), W.tvn, vn, W.Fs, w, ρ, W.θ, W.π, W.K, qe,
                                         η, m.edge_cells, m.edge_weight, m.inverse_dual, m.trisk_index, m.trisk,
                                         m.dz, m.dzh, m.wlow, nl, m.vertical); copy(W.tvn))
        # the PV flux does no work
        F = ρe .* vn
        t = tendency(W.qe, zero(w), zero(W.η))
        @test abs(sum(Ae .* (F .* t) * v.dz)) < 1e-12 * sum(Ae .* abs.(F .* t) * v.dz)
        # the vorticity term exchanges horizontal and vertical kinetic energy without loss
        NH.launch!(m, NH.eta_kernel!, (ne, nl + 1), W.η, vn, w, m.edge_cells, m.inverse_dual, m.dzh)
        NH.launch!(m, NH.w_tendency_vi_kernel!, (nc, nl + 1), W.tw, W.η, vn, W.ρe, W.K, m.cell_edges, m.inner,
                   m.dzh, m.wlow, mw)
        t = tendency(zero(W.qe), w, W.η)
        a = sum(Ae .* ((mw ? ρe : 1) .* vn .* t) * v.dz)
        b = sum(g.cell_areas .* ((mw ? ρh : 1) .* w .* W.tw) * v.dzh)
        @test abs(a + b) < 1e-12 * abs(a)
    end
end

@testset "Smagorinsky diffusion" begin
    NH = dmplex.Nonhydrostatic
    rng = Random.MersenneTwister(2)
    g = TRiSKGrid(create_hexagon_mesh(3))
    v = VerticalGrid(4, 10000.0)
    a = 6.37122e6
    m = TRiSKNonhydrostatic(g, v)
    ne, nc, nl = num_edges(g), num_cells(g), 4
    y = zero_state(m)
    y.vn .= randn(rng, ne, nl)
    y.ρ .= 1 .+ 0.2 .* rand(rng, nc, nl)
    y.ρθ .= 300 .* y.ρ
    hd, hρθ = zeros(ne, nl), zeros(nc, nl)
    NH.smagorinsky!(hd, hρθ, m, y)
    ρe = (g.cell_distances[1, :] .* y.ρ[g.edge_cells[1, :], :] .+ g.cell_distances[2, :] .* y.ρ[g.edge_cells[2, :], :]) ./
         g.dual_lengths
    # the loss of kinetic energy heats the cells
    dE = sum((a^2 .* g.primal_lengths .* g.dual_lengths) .* (ρe .* y.vn .* hd) * v.dz)
    Q = sum((a^2 .* g.cell_areas) .* (NH.Cpd .* (NH.Rd .* y.ρθ ./ NH.P0) .^ (NH.Rd / NH.Cvd) .* hρθ) * v.dz)
    @test dE < 0
    @test abs(dE + Q) < 1e-12 * abs(dE)
    # a solid-body rotation is no deformation: small coefficient
    y.vn .= [20 * dot(cross([0, 0, 1.0], g.edge_points[e]), g.edge_normals[e]) for e in 1:ne, _ in 1:nl]
    NH.smagorinsky!(hd, hρθ, m, y)
    @test maximum(m.work.Ke) < 0.05 * 0.15 * 8π * a^2 / ne * 20 / a
end

@testset "upwind θ at the interfaces" begin
    NH = dmplex.Nonhydrostatic
    Random.seed!(3)
    g = TRiSKGrid(create_hexagon_mesh(1))
    v = VerticalGrid(12, 30000.0; stretch=3)
    m = TRiSKNonhydrostatic(g, v)
    nc, nl = length(g.centers), 12
    ρ = 1 .+ rand(nc, nl)
    w = randn(nc, nl + 1)
    θi, A, θm = zeros(nc, nl + 1), zeros(nc, nl, 5), zeros(nc, nl)
    run(θ, τ) = NH.launch!(m, NH.theta_interface_kernel!, nc, θi, A, θm, θ, ρ, ρ .* θ, ρ, w, m.dz, m.dzh, m.wlow,
                           τ, nl, true)
    # without advection the upwind values are exact for θ linear in z, either sign of w
    θ = repeat((300 .+ 0.004 .* v.z_full)', nc)
    run(θ, 0.0)
    θlin = [v.wlow[j] * θ[i, j - 1] + (1 - v.wlow[j]) * θ[i, j] for i in 1:nc, j in 2:nl]
    @test maximum(abs, θi[:, 2:nl] .- θlin) < 1e-10
    # the implicit advection keeps a constant
    run(fill(300.0, nc, nl), 600.0)
    @test maximum(abs, θi[:, 2:nl] .- 300) < 1e-10
    # and solves θm + τ/(2ρ dz) [M_{k+1}(θ_{k+1} - θm_k) - M_k(θ_k - θm_k)] = θ0
    θ0 = 300 .+ randn(nc, nl)
    run(θ0, 600.0)
    M = [2 <= j <= nl ? (v.wlow[j] * ρ[i, j - 1] + (1 - v.wlow[j]) * ρ[i, j]) * w[i, j] : 0.0 for i in 1:nc, j in 1:nl + 1]
    r = [θm[i, k] + 300 / (ρ[i, k] * v.dz[k]) * (M[i, k + 1] * (θi[i, k + 1] - θm[i, k]) - M[i, k] * (θi[i, k] - θm[i, k])) -
         θ0[i, k] for i in 1:nc, k in 1:nl]
    @test maximum(abs, r) < 1e-9
end

@testset "kite nonhydrostatic: spaces and mass divergence" begin
    for level in (2, 3)
        m = KiteNonhydrostatic(create_hexagon_mesh(level), VerticalGrid(6, 12000.0; stretch=2))
        y, v = zero_state(m), m.vgrid
        nh, nu, nl = num_columns(m), size(y.u, 1), num_layers(v)
        @test size(y.w) == (nh, nl + 1) && size(y.ρ) == (nh, nl) && size(y.ρθ) == (nh, nl)
        @test nh == num_dofs(m.horizontal.H) && nu == num_dofs(m.horizontal.U)
        # the total mass is conserved with W = 0 at the ground and the top
        rng = MersenneTwister(level)
        F, W = randn(rng, nu, nl), randn(rng, nh, nl + 1)
        W[:, 1] .= 0
        W[:, end] .= 0
        dρ = similar(y.ρ)
        mass_divergence!(dρ, m, F, W)
        M = column_weights(m)
        @test abs(sum((M' * dρ) .* v.dz')) < 1e-14 * sum(abs, M .* dρ .* v.dz')
        # a vertical flux changes every column by its boundary fluxes
        W .= randn(rng, nh, nl + 1)
        mass_divergence!(dρ, m, zero(F), W)
        @test dρ * v.dz ≈ W[:, 1] .- W[:, end]
        # fields linear in z are exact at the layer centres; a solid-body rotation has a small divergence
        @test project_layers(m, (x, z) -> 1.2 - 1e-4z) ≈ repeat((1.2 .- 1e-4 .* v.z_full)', nh)
        @test size(project_layers(m, (x, z) -> z; levels=:half)) == (nh, nl + 1)
        U = project_velocity(m, (x, z) -> 20.0 * SVector(-x[2], x[1], 0.0))
        mass_divergence!(dρ, m, U, zero(W))
        @test maximum(abs, dρ) < 0.1 * 20.0 / m.radius
    end
end

@testset "kite nonhydrostatic: column solver" begin
    m = KiteNonhydrostatic(create_hexagon_mesh(2), VerticalGrid(10, 30000.0; stretch=3); damping_height=22000.0,
                           damping_rate=0.05)
    # the columns sit at the Radau nodes of the height dofs
    x = column_points(m)
    @test length(x) == num_columns(m) && all(p -> norm(p) ≈ 1, x)
    @test maximum(abs, project_layers(m, (x, z) -> x[3])[:, 1] .- getindex.(x, 3)) < 1e-4
    # a column in discrete hydrostatic balance stays at rest
    rest(λ, φ, z) = (baroclinic_wave(0.0, 0.0, z; perturbation=false)[1:2]..., 0.0, 0.0)
    y = initial_state(m, rest)
    ρ0, ys = copy(y.ρ), zero_state(m)
    for _ in 1:5
        vertical_step!(m, y, 300.0, ys)
    end
    @test maximum(abs, y.w) < 1e-12
    @test y.ρ ≈ ρ0 rtol = 1e-14
    # a warm layer rises, every column keeps its mass
    y.ρθ .*= 1 .+ (m.vgrid.z_full' .< 3000) ./ 300
    column = y.ρ * m.vgrid.dz
    for _ in 1:5
        vertical_step!(m, y, 300.0, ys)
    end
    @test maximum(y.w) > 0.01
    @test y.ρ * m.vgrid.dz ≈ column rtol = 1e-14
end
