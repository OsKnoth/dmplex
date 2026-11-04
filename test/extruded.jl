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

@testset "kite nonhydrostatic: horizontal terms" begin
    NH = dmplex.Nonhydrostatic
    m = KiteNonhydrostatic(create_hexagon_mesh(2), VerticalGrid(6, 30000.0; stretch=3))
    v, Mh = m.vgrid, column_weights(m)
    y = initial_state(m, (λ, φ, z) -> baroclinic_wave(λ, φ, z))
    dy = map(similar, (u=y.u, ρ=y.ρ, ρθ=y.ρθ))
    horizontal_tendency!(dy, m, y)
    # kinetic energy, PV and pressure terms exchange energy without loss, mass is conserved
    rate = scale = 0.0
    for k in 1:num_layers(v)
        F, rhs_K = similar(y.u[:, k]), zeros(length(Mh))
        NH.weighted_projection!(F, m, y.u[:, k], y.ρ[:, k]; kinetic=rhs_K)
        π = (NH.Rd .* y.ρθ[:, k] ./ NH.P0) .^ (NH.Rd / NH.Cvd)
        p = [F .* (m.horizontal.Mu * dy.u[:, k]); (rhs_K .+ NH.GRAV * v.z_full[k] .* Mh) .* dy.ρ[:, k];
             NH.Cpd .* π .* Mh .* dy.ρθ[:, k]]
        rate += v.dz[k] * sum(p)
        scale += v.dz[k] * sum(abs, p)
    end
    @test abs(rate) < 1e-14 * scale
    # the velocity mass uses the rule of the projections: Π_U(c v) = c v on the non-affine kites,
    # so the ρθ flux Π_U(θ̄ F) of a layer with constant θ̄ is θ̄ F and creates no θ′
    u = randn(MersenneTwister(3), size(y.u, 1))
    Πu = similar(u)
    NH.weighted_projection!(Πu, m, u, fill(300.0, num_columns(m)))
    @test Πu ≈ 300 .* u rtol = 1e-13
    # the two vertical coupling terms are adjoint: Σ_k dz_k u_kᵀ M_u (η w)_k = Σ_j dzh_j w_jᵀ M_h (η ⋅ ṽ)_j
    y.w .= randn(MersenneTwister(4), size(y.w))
    y.w[:, 1] .= 0
    y.w[:, end] .= 0
    horizontal_tendency!(dy, m, y)
    W, nl = m.work, num_layers(v)
    lhs = rhs = 0.0
    p, d = similar(y.u[:, 1]), zeros(num_columns(m))
    for j in 2:nl
        NH.weighted_projection!(p, m, W.η[:, j], y.w[:, j])
        lhs += v.dzh[j] / 2 * dot(y.u[:, j - 1] .+ y.u[:, j], m.horizontal.Mu * p)
        ṽ = (y.u[:, j - 1] .+ y.u[:, j]) ./ 2
        NH.dot_product!(d, m, W.η[:, j] .+ ṽ, W.η[:, j] .- ṽ)
        rhs += v.dzh[j] * dot(y.w[:, j], Mh .* d)
    end
    @test lhs ≈ rhs rtol = 1e-10
    @test abs(sum((Mh' * dy.ρ) .* v.dz')) < 1e-14 * sum(abs, Mh .* dy.ρ .* v.dz')
    # a horizontally homogeneous column in balance stays at rest
    rest(λ, φ, z) = (baroclinic_wave(0.0, 0.0, z; perturbation=false)[1:2]..., 0.0, 0.0)
    y = initial_state(m, rest)
    ys = zero_state(m)
    for _ in 1:3
        step!(m, y, 150.0, ys)
    end
    @test maximum(abs, y.u) < 1e-10 && maximum(abs, y.w) < 1e-10
end

@testset "kite nonhydrostatic: upwind θ on the primal faces" begin
    NH = dmplex.Nonhydrostatic
    m = KiteNonhydrostatic(create_hexagon_mesh(2), VerticalGrid(4, 30000.0); horizontal_theta=:upwind)
    sw, up = m.horizontal, m.upwind
    rng = MersenneTwister(5)
    F, θ = randn(rng, NH.num_velocity_dofs(m)), randn(rng, num_columns(m))
    # both sides of every face point see the same normal flux
    @test all(eachindex(up.cK)) do i
        fK, fE = sw.faces[up.fK[i]], sw.faces[up.fE[i]]
        nK = dot(dmplex.ShallowWater.gather_vector(F, fK.φ, sw.iu, sw.su, up.cK[i], up.qK[i]), fK.ν)
        nE = dot(dmplex.ShallowWater.gather_vector(F, fE.φ, sw.iu, sw.su, up.cE[i], up.qE[i]), fE.ν)
        abs(nK + nE) < 1e-12
    end
    th = NH.upwind_theta!(zeros(num_columns(m)), m, F, θ, 1.0)
    @test abs(sum(th)) < 1e-12 * sum(abs, th)                 # conserves ρθ
    @test dot(θ, th) < 0                                       # damps the jumps of θ
    @test maximum(abs, NH.upwind_theta!(zeros(num_columns(m)), m, F, fill(300.0, num_columns(m)), 1.0)) <
          1e-12 * 300 * sum(abs, F)                            # vanishes for constant θ
    # a horizontally homogeneous column in balance stays at rest
    rest(λ, φ, z) = (baroclinic_wave(0.0, 0.0, z; perturbation=false)[1:2]..., 0.0, 0.0)
    y, ys = initial_state(m, rest), zero_state(m)
    for _ in 1:3
        step!(m, y, 150.0, ys)
    end
    @test maximum(abs, y.u) < 1e-10 && maximum(abs, y.w) < 1e-10
end

@testset "thermal wind: analytic balance" begin
    NH = dmplex.Nonhydrostatic
    a, Ω, h = 6.37122e6, 7.292e-5, 1e-4
    for φ in (-1.2, -0.5, 0.1, 0.7, 1.3), z in (0.0, 5000.0, 15000.0, 28000.0)
        T, p, u, v = thermal_wind(0.3, φ, z)
        ρ = p / (NH.Rd * T)
        dpdz = (thermal_wind(0.3, φ, z + h)[2] - thermal_wind(0.3, φ, z - h)[2]) / 2h
        dpdφ = (thermal_wind(0.3, φ + h, z)[2] - thermal_wind(0.3, φ - h, z)[2]) / 2h
        @test dpdz ≈ -ρ * NH.GRAV rtol = 1e-6                          # hydrostatic
        @test (2Ω * sin(φ) + u * tan(φ) / a) * u ≈ -dpdφ / (ρ * a) atol = 1e-9   # gradient wind
        @test v == 0
    end
    @test thermal_wind(0.0, 0.5, 0.0)[3] == 0                            # no wind at the ground
    @test 20 < thermal_wind(0.0, 0.0, 15000.0)[3] < 30
    @test thermal_wind(0.0, 0.0, 29000.0)[3] - thermal_wind(0.0, 0.0, 25000.0)[3] < 0.5   # no shear aloft
end

@testset "ICON nonhydrostatic" begin
    NH = dmplex.Nonhydrostatic
    v = VerticalGrid(10, 30000.0)
    m = ICONNonhydrostatic(create_spherical_mesh(2), v; damping_height=22000.0, damping_rate=0.05)
    @test_throws ArgumentError ICONNonhydrostatic(create_hexagon_mesh(1), v)
    # a horizontally homogeneous column in balance stays at rest
    rest(λ, φ, z) = (baroclinic_wave(0.0, 0.0, z; perturbation=false)[1:2]..., 0.0, 0.0)
    y, ys = initial_state(m, rest), zero_state(m)
    for _ in 1:5
        step!(m, y, 300.0, ys)
    end
    @test maximum(abs, y.vn) < 1e-10 && maximum(abs, y.w) < 1e-10
    # the steady jet keeps its mass to round-off and stays near the steady state
    y = initial_state(m, (λ, φ, z) -> baroclinic_wave(λ, φ, z; perturbation=false))
    m0, θ0 = NH.total_mass(m, y), y.ρθ ./ y.ρ
    for _ in 1:24
        step!(m, y, 300.0, ys)
    end
    @test abs(NH.total_mass(m, y) / m0 - 1) < 1e-14
    @test maximum(abs, y.w) < 0.05
    @test maximum(abs, y.ρθ ./ y.ρ .- θ0) < 1.0
end

@testset "kite nonhydrostatic: one layer is kite shallow water" begin
    # θ = 1, p = g (ρθ)²/2: the horizontal terms of one layer are those of NonlinearKiteShallowWater
    u0 = 0.2
    velocity(x) = u0 * SVector(-x[2], x[1], 0.0)
    height(x) = 1 - (u0 + u0^2 / 2) * x[3]^2
    for primal in (create_hexagon_mesh(2), create_cubed_sphere_mesh(4))
        m2 = NonlinearKiteShallowWater(primal, 1; gravity=1.0, coriolis=x -> 2x[3])
        m3 = KiteNonhydrostatic(primal, VerticalGrid(1, 1.0), 1; radius=1.0, omega=1.0,
                                equation_of_state=:shallow_water, gravity=1.0)
        y2 = initial_state(m2, velocity, height) .+ 1e-3 .* randn(MersenneTwister(3), num_dofs(m2.U) + num_dofs(m2.H))
        y3 = zero_state(m3)
        y3.u[:, 1] .= y2[velocity_dofs(m2)]
        y3.ρ[:, 1] .= y2[height_dofs(m2)]
        y3.ρθ .= y3.ρ
        d2 = tendency!(similar(y2), y2, m2)
        d3 = horizontal_tendency!(m3.work.dy, m3, y3)
        @test d3.u[:, 1] ≈ d2[velocity_dofs(m2)] rtol = 1e-12
        @test d3.ρ[:, 1] ≈ d2[height_dofs(m2)] rtol = 1e-12
        @test d3.ρθ ≈ d3.ρ rtol = 1e-13
    end
end

@testset "kite nonhydrostatic: tracer with mixing ratio one" begin
    m = KiteNonhydrostatic(create_hexagon_mesh(2), VerticalGrid(6, 30000.0); damping_height=22000.0, damping_rate=0.05,
                           tracer=true)
    y = initial_state(m, (λ, φ, z) -> baroclinic_wave(λ, φ, z; perturbation=false))
    @test y.ρq == y.ρ
    ys = zero_state(m)
    @test haskey(ys, :ρq)
    for _ in 1:4
        step!(m, y, 150.0, ys)
    end
    @test maximum(abs, y.ρq ./ y.ρ .- 1) < 1e-13        # ρq = ρ: the tracer equations are those of ρ for q = 1
    q0 = (λ, φ, z) -> 1 + 0.1 * sin(φ)
    @test initial_state(m, (λ, φ, z) -> baroclinic_wave(λ, φ, z; perturbation=false); tracer=q0).ρq != y.ρ
end

@testset "kite nonhydrostatic: DG transport of ρθ and ρq" begin
    NH = dmplex.Nonhydrostatic
    for primal in (create_hexagon_mesh(1), create_cubed_sphere_mesh(3), create_spherical_mesh(1)), ht in (:dg, :dg_upwind)
        m = KiteNonhydrostatic(primal, VerticalGrid(1, 1.0), 1; radius=1.0, horizontal_theta=ht)
        nh, nu = NH.num_columns(m), NH.num_velocity_dofs(m)
        F, θ, π = randn(nu), 1 .+ 0.1 .* randn(nh), randn(nh)
        th, tu, BF = zeros(nh), zeros(nu), zeros(nh)
        NH.dg_divergence!(th, m, F, θ, 1.0)
        NH.dg_gradient!(tu, m, F, θ, π, 1.0)
        @test dot(π, th) ≈ dot(F, tu) rtol = 1e-12       # adjoint: the energy exchange is exact
        dmplex.FEM.row_mul!(BF, m.horizontal.B_owned, F)
        @test NH.dg_divergence!(th, m, F, ones(nh), 1.0) ≈ BF rtol = 1e-12   # q = 1 gives the mass flux
        @test abs(sum(NH.dg_divergence!(th, m, F, θ, 1.0))) < 1e-10 * norm(th)  # conservative
        # DG advection u ⋅ ∇w (the w tendency subtracts it) and its adjoint in the momentum equation:
        # exact energy exchange for the central trace, the upwind penalty only removes energy
        u, wv = randn(nu), randn(nh)
        @test norm(NH.w_advection!(th, m, u, fill(2.0, nh), 1.0)) < 1e-12           # a constant w is not advected
        adv, flux = dot(wv, NH.w_advection!(th, m, u, wv, 1.0)), dot(u, NH.w_advection_flux!(tu, m, wv, 1.0))
        ht == :dg ? (@test adv ≈ flux rtol = 1e-12) : (@test adv > flux)
    end
    m = KiteNonhydrostatic(create_hexagon_mesh(2), VerticalGrid(6, 30000.0); damping_height=22000.0, damping_rate=0.05,
                           tracer=true, horizontal_theta=:dg_upwind)
    y = initial_state(m, (λ, φ, z) -> baroclinic_wave(λ, φ, z; perturbation=false)); ys = zero_state(m)
    for _ in 1:4
        step!(m, y, 150.0, ys)
    end
    @test maximum(abs, y.ρq ./ y.ρ .- 1) < 1e-13
end

@testset "baroclinic channel: balance and planar kite model" begin
    f0 = 1.0313e-4
    for (y, z) in ((1.5e6, 3000.0), (3e6, 8000.0), (9e6, 3000.0))
        p(yy, zz) = baroclinic_channel(0.0, yy, zz; perturbation=false)[2]
        T, pr, u, _ = baroclinic_channel(0.0, y, z; perturbation=false)
        ρ = pr / (dmplex.Nonhydrostatic.Rd * T)
        @test -(p(y + 1, z) - p(y - 1, z)) / (2 * f0 * ρ) ≈ u rtol = 1e-5                       # geostrophic
        @test (p(y, z + 1) - p(y, z - 1)) / 2 ≈ -ρ * dmplex.Nonhydrostatic.GRAV rtol = 1e-6   # hydrostatic
    end
    @test baroclinic_channel(0.0, 1.0, 5000.0; perturbation=false)[1] ≈ baroclinic_channel(0.0, 1.2e7 - 1.0, 5000.0; perturbation=false)[1]
    primal = create_periodic_plane(10, 3; Lx=4e7, Ly=1.2e7)
    m = KiteNonhydrostatic(primal, VerticalGrid(4, 30000.0), 1; radius=1.0, coriolis=x -> f0, horizontal_theta=:dg_upwind)
    y = initial_state(m, (x, yy, z) -> baroclinic_channel(x, yy, z; perturbation=false)); ys = zero_state(m)
    W = column_weights(m); mass(y) = sum(W' * y.ρ .* m.vgrid.dz')
    m0 = mass(y)
    step!(m, y, 300.0, ys)
    @test all(isfinite, y.ρ)
    @test abs(mass(y) / m0 - 1) < 1e-13
end
