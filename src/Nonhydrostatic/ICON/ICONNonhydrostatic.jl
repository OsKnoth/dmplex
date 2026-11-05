"""
    ICONNonhydrostatic(grid::TRiSKGrid, vgrid::VerticalGrid; radius=6.37122e6, omega=7.292e-5,
                       beta=0.65, damping_height=Inf, damping_rate=0, itime_scheme=4,
                       divdamp_order=4, divdamp_type=3, divdamp_fac=0.0025, divdamp_fac2=0.004,
                       divdamp_fac3=0.004, divdamp_fac4=0.004, divdamp_z=32500, divdamp_z2=40000,
                       divdamp_z3=60000, divdamp_z4=80000, divdamp_fac_o2=0, veladv_offctr=0.25,
                       rhotheta_offctr=-0.1, exner_expol=1/3, rbf_scale=0.5, hdiff_order=-1,
                       ndyn_substeps=5, hdiff_efdt_ratio=36, hdiff_w_efdt_ratio=15, hdiff_smag_fac=0.015,
                       hdiff_smag_fac2=0.07137, hdiff_smag_fac3=0, hdiff_smag_fac4=1, hdiff_smag_z=32500,
                       hdiff_smag_z2=60686, hdiff_smag_z3=50000, hdiff_smag_z4=90000, lhdiff_vn=true,
                       lhdiff_w=true, lhdiff_temp=true, rbf_vec_scale_v=0.5, hyperdiffusion=0,
                       backend=CPU())

Dry compressible Euler equations on the extruded triangular C grid of ICON
(Zängl et al. 2015, `mo_solve_nonhydro`, `mo_velocity_advection`), shallow
atmosphere, flat ground. `grid` is the `TRiSKGrid` of a triangle mesh with
circumcentres. Lorenz staggering: `vn` at the edges × layers, `w` at the cells
× interfaces, `ρ`, `ρθ` at the cells × layers (state `(vn, w, ρ, ρθ)` as for
`MPASGassmannNonhydrostatic`).

**Velocity tendencies** (`mo_velocity_advection`), horizontal operators of
`ICONShallowWater` per layer:

    vt_e    = Σ_k c_{e,k} vn_k                     tangential velocity, vector RBF (4 edges)
    K_e     = (vn_e² + vt_e²)/2,  K_i = Σ_e b_{i,e} K_e      bilinear weights (e_bln_c_s)
    ζ_e     = (ζ_{v1} + ζ_{v2})/2,  ζ_v = (1/A_v) Σ t_{e,v} d_e vn_e
    ∂vn/∂t  = (f_e + ζ_e) vt_e - [K_e (a₁ - a₂) + a₂ K_r - a₁ K_l] - w_e ∂vn/∂z
    ∂w/∂t   = -Σ_e b_{i,e} (vn_ie ∂w/∂n + vt_ie ∂w/∂t)_e - w ∂w/∂z

with the kinetic-energy gradient of ICON (`coeff_gradekin`), `vn_ie` the
interpolation of `vn` to the interfaces (`wgtfac_e`), `w_e` the full-level `w`
at the edges (`c_lin_e`), and `∂w/∂t` along the edge from `w` at the vertices
(area weights, `cells_aw_verts`).

**Fast waves** (`solve_nh`): horizontal pressure gradient `-c_p θ_e ∂π/∂n`,
mass flux `ρ_e vn` and `ρθ` flux `ρ_e θ_e vn` with `ρ_e`, `θ_e` from the
second-order upwind reconstruction of Miura (2007): the value of the upwind
cell plus its Green–Gauss gradient times the distance from the cell centre to
the backtrajectory point `x_e - (Δt/2)(vn n_e + vt t_e)` (ICON
`upwind_hflux_miura` with `iadv_rhotheta = 2`). The vertical terms are
implicit per column with the off-centring `beta` (`0.5 + vwind_offctr`),
centred θ and ρ at the interfaces, linearized around the Exner pressure of
the old state: the column solver of `MPASGassmannNonhydrostatic` with
`vertical_theta = :centred` and the lumped vertical mass.

**Time stepping** (`step!`): the two-time-level predictor–corrector of ICON
(`solve_nh`, `istep = 1, 2`). Predictor: the velocity tendencies of `n` (with
the default `itime_scheme = 4` those of the last corrector, computed at the
predictor state of the previous step, recomputed only in the first step;
`itime_scheme = 6` recomputes them every step), the pressure gradient of the
Exner pressure extrapolated in time, `π^n + exner_expol (π^n - π^{n-1})`, and
`ρ_e`, `θ_e` of `n`; `vn* = vn^n + Δt (T_vn^n - c_p θ_e ∂π/∂n)`, the fluxes of
`vn*` and the column solve with `ρ`, `θ` of `n` linearly interpolated to the
interfaces give `w*`, `ρ*`, `ρθ*`. Corrector: velocity tendencies at the
predictor state, weighted `(0.5 - veladv_offctr, 0.5 + veladv_offctr)`; the
same pressure gradient and `ρ_e`, `θ_e`;
`vn^{n+1} = vn^n + Δt (Σ w T_vn - c_p θ_e ∂π/∂n)` plus the divergence damping of
the predictor state; then the fluxes of `vn^{n+1}` and a second column solve
with the weighted `w` tendency and `ρ`, `θ` at the interfaces from the time
average `(0.5 - rhotheta_offctr) x^n + (0.5 + rhotheta_offctr) x*`, upwind
biased along the backtrajectory over `Δt/2` with `w*`.

**Divergence damping** of the predictor state, added to `vn^{n+1}`: the
divergence `D = ∇⋅vn* + ∂w*/∂z` (`divdamp_type = 3`, 3D; `2` drops `∂w/∂z`;
`32` blends from 3D to 2D between `divdamp_trans_start` and `divdamp_trans_end`),
fourth order `-k₄(z) Ā² ∇(∇⋅∇D)` (`divdamp_order = 4`) with ICON's profile
`k₄(z)`: `divdamp_fac` up to `divdamp_z`, linear to `divdamp_fac2` at
`divdamp_z2`, quadratic through `divdamp_fac3` at `divdamp_z3` to
`divdamp_fac4` at `divdamp_z4`, constant above (`Ā` the mean cell area);
second order `divdamp_fac_o2 Ā ∇D` (`divdamp_order = 2`); or both
(`divdamp_order = 24`, `k₄` reduced by `divdamp_fac_o2/4`, the fourth-order
term only for `divdamp_fac_o2 ≤ 4 divdamp_fac`). ICON sets `divdamp_fac_o2`
by its spin-up schedule only in real-data runs and to zero in test cases, so
24 is 4 by default.

**Diffusion** (`mo_nh_diffusion`, `hdiff_order = 5`), after every
`ndyn_substeps` steps over their sum `Δt_a = ndyn_substeps Δt` (ICON's
advective time step): the velocity at the vertices by vector RBF
(`rbf_vec_scale_v`); at the edges the Smagorinsky coefficient
`K = max(0, s(z) Δt_a |D| - k₄ ndyn_substeps/4)`, limited by `1/8 - 4k`, of the
deformation `|D|` on the quadrilateral of the edge (`s(z)` the profile of
`hdiff_smag_fac…fac4` over `hdiff_smag_z…z4`, as the divergence damping), and
`vn += A_e (K ∇²vn - A_e k ∇⁴vn)` with `k = min(1/128, k₄ ndyn_substeps/3)`,
`k₄ = 1/(64 hdiff_efdt_ratio)` (`lhdiff_vn`); `w -= k_w A² ∇⁴w` with
`k_w = min(1/48, ndyn_substeps/(36 hdiff_w_efdt_ratio))` plus the `∇²w` of ICON
above `damping_height`, both capped so that the factor `1 - k_w ρ² - n2w ρ`
of the highest mode stays non-negative, `ρ = 2 max Σ l/d` (`lhdiff_w`); `θ += A ∇⋅(K ∇θ)` at constant `ρ`, with
`K` raised over cold pools in the lowest two layers (`lhdiff_temp`).
It is off by default (`hdiff_order = -1`): on our meshes, which are not
spring-optimized like ICON's, it lets a grid-scale mode of `w` grow near the
pentagons within two days at R2B4. `hyperdiffusion = ν₄` adds `-ν₄ L(L(vn))`
to every dynamics step instead.

Differences to ICON: `ρθ` is prognostic and `π` follows from the equation of
state in every step (ICON carries `exner` and `θ_v` and updates `exner` with
the linearized equation); no reference state (with flat ground its horizontal
gradient vanishes); no tracer or physics coupling (`ndyn_substeps` only sets
when the diffusion is called); no nest-boundary terms; the Rayleigh damping of `w` as in
`MPASGassmannNonhydrostatic`. Serial.
"""
struct ICONNonhydrostatic{M, V, I, BE} <: CGridNonhydrostatic
    grid::TRiSKGrid
    vgrid::VerticalGrid
    cell_edges::I
    divergence::M        # n_{e,i} l_e / (a A_i)
    bilinear::M          # e_bln_c_s
    edge_cells::I
    edge_vertices::I
    edge_height::M       # c_lin_e
    gradient::M          # coeff_gradekin / a
    inverse_dual::V      # 1 / (a d_e)
    inverse_primal::V    # 1 / (a l_e)
    tangent_sign::V      # ±1: t_e = x_e × n_e points from vertex 1 to vertex 2 (+1) or back
    rbf_edges::I
    rbf::M
    vertex_edges::I
    curl::M              # t_{e,v} d_e / (a A_v)
    vertex_cells::I
    vertex_weight::M     # K_{i,v} / A_v (cells_aw_verts)
    edge_coriolis::V
    green_gauss::NTuple{3, M}   # n_{e,i} l_e n_e / A_i on the unit sphere: Green–Gauss gradient at the cells
    edge_vectors::M      # 12 × ne: x_e - x_l, x_e - x_r, n_e, t_e (unit sphere)
    cell_areas::V        # a² A_i
    dz::V
    dzh::V
    wlow::V
    rayleigh::V
    work::NamedTuple
    radius::Float64
    beta::Float64
    divdamp::V           # -k₄(z) Ā² per layer, zero without fourth-order damping (scal_divdamp)
    divdamp_o2::V        # divdamp_fac_o2 Ā per layer, zero without second-order damping
    divdamp_3d::V        # weight of ∂w/∂z in the damped divergence per layer (scalfac_dd3d)
    weights::NTuple{2, Float64}   # velocity tendencies of n and of the predictor in the corrector
    rth_weights::NTuple{2, Float64}   # ρ, θ of n and of the predictor at the interfaces in the corrector
    itime_scheme::Int    # 4: predictor tendencies of the last corrector, 6: recomputed
    exner_expol::Float64
    hyperdiffusion::Float64
    hdiff_order::Int
    ndyn_substeps::Int
    diffusion::NamedTuple
    vertical_theta::Int  # 0: centred, as ICON
    backend::BE
end

function ICONNonhydrostatic(grid::TRiSKGrid, vgrid::VerticalGrid; radius::Real=6.37122e6, omega::Real=7.292e-5,
                            beta::Real=0.65, damping_height::Real=Inf, damping_rate::Real=0.0,
                            itime_scheme::Integer=4, divdamp_order::Integer=4, divdamp_type::Integer=3,
                            divdamp_fac::Real=0.0025, divdamp_fac2::Real=0.004, divdamp_fac3::Real=0.004,
                            divdamp_fac4::Real=0.004, divdamp_z::Real=32500.0, divdamp_z2::Real=40000.0,
                            divdamp_z3::Real=60000.0, divdamp_z4::Real=80000.0, divdamp_fac_o2::Real=0.0,
                            divdamp_trans_start::Real=12500.0, divdamp_trans_end::Real=17500.0,
                            veladv_offctr::Real=0.25, rhotheta_offctr::Real=-0.1, exner_expol::Real=1 / 3,
                            rbf_scale::Real=0.5, hdiff_order::Integer=-1, ndyn_substeps::Integer=5,
                            hdiff_efdt_ratio::Real=36.0, hdiff_w_efdt_ratio::Real=15.0, hdiff_smag_fac::Real=0.015,
                            hdiff_smag_fac2::Real=2e-6 * (1600 + 25000 + sqrt(1600 * (1600 + 50000))),
                            hdiff_smag_fac3::Real=0.0, hdiff_smag_fac4::Real=1.0, hdiff_smag_z::Real=32500.0,
                            hdiff_smag_z2::Real=1600 + 50000 + sqrt(1600 * (1600 + 50000)),
                            hdiff_smag_z3::Real=50000.0, hdiff_smag_z4::Real=90000.0, lhdiff_vn::Bool=true,
                            lhdiff_w::Bool=true, lhdiff_temp::Bool=true, rbf_vec_scale_v::Real=0.5,
                            hyperdiffusion::Real=0.0, backend=KernelAbstractions.CPU())
    size(grid.cell_edges, 1) == 3 || throw(ArgumentError("ICONNonhydrostatic needs a triangle mesh"))
    itime_scheme in (4, 6) || throw(ArgumentError("itime_scheme must be 4 or 6"))
    divdamp_order in (2, 4, 24) || throw(ArgumentError("divdamp_order must be 2, 4 or 24"))
    divdamp_type in (2, 3, 32) || throw(ArgumentError("divdamp_type must be 2, 3 or 32"))
    hdiff_order in (-1, 5) || throw(ArgumentError("hdiff_order must be -1 or 5"))
    ndyn_substeps >= 1 || throw(ArgumentError("ndyn_substeps must be positive"))
    a = Float64(radius)
    nc, ne, nv, nlev = num_cells(grid), num_edges(grid), num_vertices(grid), num_layers(vgrid)
    l, d, dc = grid.primal_lengths, grid.dual_lengths, grid.cell_distances
    divergence = zeros(3, nc)
    green_gauss = ntuple(_ -> zeros(3, nc), 3)
    for i in 1:nc, k in 1:3
        e = grid.cell_edges[k, i]
        divergence[k, i] = grid.cell_signs[k, i] * l[e] / (a * grid.cell_areas[i])
        g = grid.cell_signs[k, i] * l[e] / grid.cell_areas[i] .* grid.edge_normals[e]
        foreach(c -> green_gauss[c][k, i] = g[c], 1:3)
    end
    edge_height = [dc[2, :]'; dc[1, :]'] ./ d'
    gradient = [(dc[2, :] ./ (dc[1, :] .* d))'; (dc[1, :] ./ (dc[2, :] .* d))'] ./ a
    curl = similar(grid.vertex_signs)
    for v in 1:nv, k in axes(curl, 1)
        curl[k, v] = grid.vertex_signs[k, v] * d[grid.vertex_edges[k, v]] / (a * grid.vertex_areas[v])
    end
    rbf_edges, rbf = rbf_coefficients(grid; scale=rbf_scale)
    tangent_sign = zeros(ne)
    edge_vectors = zeros(12, ne)
    for e in 1:ne
        x, n = grid.edge_points[e], grid.edge_normals[e]
        t = cross(x, n)
        v1, v2 = grid.vertices[grid.edge_vertices[1, e]], grid.vertices[grid.edge_vertices[2, e]]
        tangent_sign[e] = sign(dot(t, v2 - v1))
        edge_vectors[1:3, e] .= x - grid.centers[grid.edge_cells[1, e]]
        edge_vectors[4:6, e] .= x - grid.centers[grid.edge_cells[2, e]]
        edge_vectors[7:9, e] .= n
        edge_vectors[10:12, e] .= t
    end
    z_top = vgrid.z_half[end]
    rayleigh = map(vgrid.z_half) do z
        z > damping_height ? damping_rate * sin(π / 2 * (z - damping_height) / (z_top - damping_height))^2 : 0.0
    end
    mean_area = 4π * a^2 / nc
    o4 = divdamp_order == 4 || (divdamp_order == 24 && divdamp_fac_o2 <= 4 * divdamp_fac)
    o2 = divdamp_order == 2 || (divdamp_order == 24 && divdamp_fac_o2 * mean_area > 1e-6)
    k4 = height_profile(vgrid, divdamp_fac, divdamp_fac2, divdamp_fac3, divdamp_fac4, divdamp_z, divdamp_z2,
                         divdamp_z3, divdamp_z4)
    divdamp_order == 24 && (k4 = max.(0.0, k4 .- divdamp_fac_o2 / 4))
    divdamp = o4 ? -k4 .* mean_area^2 : zeros(nlev)
    divdamp_o2 = fill(o2 ? Float64(divdamp_fac_o2) * mean_area : 0.0, nlev)
    divdamp_3d = map(1:nlev) do k
        zf = (vgrid.z_half[k] + vgrid.z_half[k + 1]) / 2
        divdamp_type == 2 ? 0.0 : divdamp_type == 3 ? 1.0 :
        clamp((divdamp_trans_end - zf) / (divdamp_trans_end - divdamp_trans_start), 0.0, 1.0)
    end
    dev(x) = to_backend(backend, x)
    z(n, k) = KernelAbstractions.zeros(backend, Float64, n, k)
    work = (θ=z(nc, nlev), π=z(nc, nlev), πprev=z(nc, nlev), first=Ref(true), K=z(nc, nlev), Ke=z(ne, nlev),
            vt=z(ne, nlev), ζv=z(nv, nlev), wv=z(nv, nlev + 1), gw=z(ne, nlev + 1),
            tvn_now=z(ne, nlev), tvn_new=z(ne, nlev), tw_now=z(nc, nlev + 1), tw_new=z(nc, nlev + 1),
            tw=z(nc, nlev + 1), grad=ntuple(_ -> z(nc, nlev), 6), ρe=z(ne, nlev), θe=z(ne, nlev),
            pg=z(ne, nlev), F=z(ne, nlev), Fθ=z(ne, nlev), hρθ=z(nc, nlev), ρexpl=z(nc, nlev), ρθexpl=z(nc, nlev),
            θi=z(nc, nlev + 1), θm=z(nc, nlev), band=KernelAbstractions.zeros(backend, Float64, nc, nlev, 2),
            cprime=z(nc, nlev + 1), dprime=z(nc, nlev + 1), wmass=dev(vertical_mass(vgrid, :lumped)),
            ρi=z(nc, nlev + 1), π0=z(nc, nlev), γ0=z(nc, nlev), δ=z(nc, nlev), δ2=z(nc, nlev), G=z(ne, nlev),
            ζ=z(nv, nlev), Lvn=z(ne, nlev), uvert=ntuple(_ -> z(nv, nlev), 3), uvert2=ntuple(_ -> z(nv, nlev), 3),
            kh=z(ne, nlev), n2e=z(ne, nlev), Lw=z(nc, nlev + 1), substep=Ref(0),
            hd=z(ne, nlev), y0=(vn=z(ne, nlev), w=z(nc, nlev + 1), ρ=z(nc, nlev), ρθ=z(nc, nlev)))
    # mo_nh_diffusion: vertex RBF, the quadrilateral of each edge, coefficients
    rbf_v = rbf_vertex_coefficients(grid; scale=rbf_vec_scale_v)
    quad = zeros(Int, 4, ne)
    for e in 1:ne
        v1, v2 = grid.edge_vertices[1, e], grid.edge_vertices[2, e]
        quad[1:2, e] .= tangent_sign[e] > 0 ? (v1, v2) : (v2, v1)
        for s in 1:2
            i = grid.edge_cells[s, e]
            quad[2 + s, e] = only(v for v in grid.cell_vertices[:, i] if v != v1 && v != v2)
        end
    end
    # n_e and t_e projected onto the tangent planes of the four vertices
    quad_vectors = zeros(24, ne)
    for e in 1:ne, j in 1:4
        p = grid.vertices[quad[j, e]]
        for (o, b) in ((0, edge_vectors[7:9, e]), (3, edge_vectors[10:12, e]))
            q = b .- dot(b, p) .* p
            quad_vectors[(6(j - 1) + o + 1):(6(j - 1) + o + 3), e] .= q ./ norm(q)
        end
    end
    inverse_vv = [1 / (a * arc_length(grid.vertices[quad[3, e]], grid.vertices[quad[4, e]])) for e in 1:ne]
    k4d = hdiff_efdt_ratio > 0 ? 1 / (64 * hdiff_efdt_ratio) : 0.0
    kvn = min(1 / 128, k4d * ndyn_substeps / 3)
    jb = findlast(<(damping_height), vgrid.z_half)
    zh = vgrid.z_half
    # the factor 1 - k4w ρ² - n2w ρ of the w diffusion for the highest mode (ρ: Gershgorin bound of the
    # spectral radius of A∇², 10.4 for equilateral triangles) kept non-negative: a negative factor flips
    # the checkerboard of w against ρ and θ every call and drives it up near the top. ICON's
    # k4w = 5/540 gives k4w ρ² = 1 on equilateral triangles, so the caps are just active there
    ρ = 2 * maximum(i -> sum(l[e] / d[e] for e in grid.cell_edges[:, i]), 1:nc)
    k4w = hdiff_w_efdt_ratio > 0 ? min(1 / 48, ndyn_substeps / (36 * hdiff_w_efdt_ratio), 1 / ρ^2) : 0.0
    n2w_max = max(0.0, (1 - k4w * ρ^2) / ρ)
    n2w = [jb !== nothing && jb < j <= nlev ? min(n2w_max, ((zh[j] - zh[jb]) / (zh[nlev] - zh[jb]))^4 / 12) : 0.0
           for j in 1:(nlev + 1)]
    on = hdiff_order == 5
    diffusion = (rbf_edges=dev(rbf_v[1]), rbf=map(dev, rbf_v[2:4]), quad=dev(quad), quad_vectors=dev(quad_vectors),
                 inverse_vv=dev(inverse_vv),
                 edge_area=dev(a^2 .* l .* d),
                 smag=dev(height_profile(vgrid, hdiff_smag_fac, hdiff_smag_fac2, hdiff_smag_fac3, hdiff_smag_fac4,
                                         hdiff_smag_z, hdiff_smag_z2, hdiff_smag_z3, hdiff_smag_z4)),
                 offset=k4d * ndyn_substeps / 4, limit=dev(fill(1 / 8 - 4kvn, nlev)), k4=kvn,
                 k4w=k4w,
                 n2w=dev(n2w), lhdiff_vn=on && lhdiff_vn, lhdiff_w=on && lhdiff_w, lhdiff_temp=on && lhdiff_temp)
    return ICONNonhydrostatic(grid, vgrid, dev(grid.cell_edges), dev(divergence), dev(bilinear_weights(grid)),
                              dev(grid.edge_cells), dev(grid.edge_vertices), dev(edge_height), dev(gradient),
                              dev(1 ./ (a .* d)), dev(1 ./ (a .* l)), dev(tangent_sign), dev(rbf_edges), dev(rbf),
                              dev(grid.vertex_edges), dev(curl), dev(grid.vertex_cells),
                              dev(grid.vertex_kites ./ grid.vertex_areas'), dev([2omega * x[3] for x in grid.edge_points]),
                              map(dev, green_gauss), dev(edge_vectors), dev(a^2 .* grid.cell_areas),
                              dev(vgrid.dz), dev(vgrid.dzh), dev(vgrid.wlow), dev(rayleigh), work, a, Float64(beta),
                              dev(divdamp), dev(divdamp_o2), dev(divdamp_3d),
                              (0.5 - veladv_offctr, 0.5 + veladv_offctr),
                              (0.5 - rhotheta_offctr, 0.5 + rhotheta_offctr), Int(itime_scheme),
                              Float64(exner_expol), Float64(hyperdiffusion), Int(hdiff_order), Int(ndyn_substeps),
                              diffusion, 0, backend)
end

"""
    height_profile(vgrid, fac, fac2, fac3, fac4, z, z2, z3, z4) -> factor per layer

ICON's height profile of the fourth-order divergence damping and of the
Smagorinsky coefficient at the layer centres: `fac` up to `z`, linear to `fac2` at `z2`, the quadratic through
`fac2` at `z2`, `fac3` at `z3` and `fac4` at `z4` between `z2` and `z4`, `fac4`
above.
"""
function height_profile(vgrid::VerticalGrid, fac, fac2, fac3, fac4, z, z2, z3, z4)
    z < z2 < z4 && z2 != z3 != z4 || throw(ArgumentError("profile heights need z < z2 < z4, z3 ∉ {z2, z4}"))
    slope = (fac2 - fac) / (z2 - z)
    d3, h3, d4, h4 = fac3 - fac2, z3 - z2, fac4 - fac2, z4 - z2
    b = (d4 * h3 - d3 * h4) / (h3 * h4 * (h4 - h3))
    c = d3 / h3 - b * h3
    return map(1:num_layers(vgrid)) do k
        zf = (vgrid.z_half[k] + vgrid.z_half[k + 1]) / 2
        dl, dq = clamp(zf - z, 0, z2 - z), clamp(zf - z2, 0, z4 - z2)
        fac + dl * slope + dq * (c + dq * b)
    end
end

ICONNonhydrostatic(dm::Plex, vgrid::VerticalGrid; kwargs...) = ICONNonhydrostatic(TRiSKGrid(dm), vgrid; kwargs...)
