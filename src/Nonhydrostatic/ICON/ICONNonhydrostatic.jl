"""
    ICONNonhydrostatic(grid::TRiSKGrid, vgrid::VerticalGrid; radius=6.371229e6, omega=7.29212e-5,
                       coriolis=x -> 2omega * x[3],
                       beta=0.65, damping_height=45000, damping_rate=0.05, itime_scheme=4,
                       divdamp_order=4, divdamp_type=3, divdamp_fac=0.0025, divdamp_fac2=0.004,
                       divdamp_fac3=0.004, divdamp_fac4=0.004, divdamp_z=32500, divdamp_z2=40000,
                       divdamp_z3=60000, divdamp_z4=80000, divdamp_fac_o2=0, veladv_offctr=0.25,
                       rhotheta_offctr=-0.1, exner_expol=1/3, rbf_scale=0.5, hdiff_order=5,
                       ndyn_substeps=5, hdiff_efdt_ratio=36, hdiff_w_efdt_ratio=15, hdiff_smag_fac=0.015,
                       hdiff_smag_fac2=0.07137, hdiff_smag_fac3=0, hdiff_smag_fac4=1, hdiff_smag_z=32500,
                       hdiff_smag_z2=60686, hdiff_smag_z3=50000, hdiff_smag_z4=90000, lhdiff_vn=true,
                       lhdiff_w=true, lhdiff_temp=true, rbf_vec_scale_v=0.5, divavg_cntrwgt=0.5,
                       hyperdiffusion=0, surface_height=nothing, z_half=nothing, coordinate=:sleve,
                       flat_height=16000, decay_scale=4000, decay_exponent=1.35, pressure_gradient=:levels,
                       backend=CPU())

Dry compressible Euler equations on the extruded triangular C grid of ICON
(Zängl et al. 2015, `mo_solve_nonhydro`, `mo_velocity_advection`), shallow
atmosphere, flat ground. `grid` is the `TRiSKGrid` of a triangle mesh with
circumcentres. Lorenz staggering: `vn` at the edges × layers, `w` at the cells
× interfaces, `ρ`, `ρθ` and the prognostic Exner pressure `exner` at the cells
× layers (state `(vn, w, ρ, ρθ, exner)`).

**Orography**: `surface_height(lon, lat)` (or the heights of the cells) gives terrain-following
levels over the reference levels of `vgrid` (`terrain_following` with `coordinate`,
`flat_height`, `decay_scale`, `decay_exponent`), or `z_half` the interfaces of every cell
(cells × nlev+1, such as ICON's `z_ifc`). As in Zängl et al. (2015) the equations are along the
levels: the horizontal fluxes times the layer thickness `Δz_e` at the edges, divided by `Δz_i`;
the contravariant correction `w_c = vn ∂z/∂n + vt ∂z/∂t` (edges, full levels; bilinear to the
cells, interpolated to the interfaces, quadratically extrapolated to the ground, ICON
`w_concorr_c`) in the vertical fluxes and the vertical advection, `w = w_c` at the ground; the
horizontal pressure gradient `c_p θ_e (∂π'/∂n - ∂z/∂n ∂π'/∂z)` of the deviation `π'` from the
reference state at the heights of the levels (ICON `igradp_method = 1`). The Rayleigh damping
and the profiles of the damping and the diffusion are on the reference levels (`vct_a`). The defaults are those of ICON's
namelists (`nonhydrostatic_nml`, `diffusion_nml`, `dynamics_nml`) and ICON's
constants (`ICON_RD`, `ICON_CPD`, `ICON_GRAV`, earth radius and rotation).
`radius` and `omega` set the planet (a small planet: `small_planet`), `coriolis(x)` the Coriolis
parameter at the point `x` of the unit sphere (default `2Ω x₃`; `x -> 0.0` without rotation, a
constant for an f-plane).

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
mass flux `ρ_e v̄` and `ρθ` flux `ρ_e θ_e v̄` with `ρ_e`, `θ_e` from the
second-order upwind reconstruction of Miura (2007): the value of the upwind
cell plus its Green–Gauss gradient times the distance from the cell centre to
the backtrajectory point `x_e - (Δt/2)(vn n_e + vt t_e)` (ICON
`upwind_hflux_miura` with `iadv_rhotheta = 2`). `v̄` is ICON's averaged normal
velocity (`e_flx_avg`, see `flux_average`, centre weight `divavg_cntrwgt`):
its divergence is the cell average of the divergence of `vn`, which damps the
grid-scale divergence of the triangles. The vertical terms are implicit per
column (`solve_nh`): the Exner pressure is linearized around `π^n`,
`π = π^n + (R/c_v) π^n/(ρθ)^n Δ(ρθ)`, the vertical pressure gradient split
around ICON's reference atmosphere (`h_scal_bg`, `t0sl_bg`, `del_t_bg`),
`(1-β) θ_j ∂π'/∂z + θ'_j ∂π_ref/∂z` explicit and `β θ_j ∂π'/∂z` implicit with
the off-centring `beta` (`0.5 + vwind_offctr`), ρ and θ linearly interpolated
to the interfaces; `θ = (ρθ)^n((π/π^n - 1) c_v/R + 1)/ρ` follows from the new
`π`. Klemp's Rayleigh damping of `w` above `damping_height`
(`rayleigh_type = 2`, `rayleigh_coeff = damping_rate`) after the solve.

**Time stepping** (`step!`): the two-time-level predictor–corrector of ICON
(`solve_nh`, `istep = 1, 2`). Predictor: the velocity tendencies of `n` (with
the default `itime_scheme = 4` those of the last corrector, computed at the
predictor state of the previous step, all recomputed in the first step and
that of `vn` in the first of every `ndyn_substeps` steps, ICON `l_recompute`;
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
above `damping_height` (`lhdiff_w`); `θ += A ∇⋅(K ∇θ)` at constant `ρ`, with
`K` raised over cold pools in the lowest two layers, and `exner` updated with it (`lhdiff_temp`).
`hdiff_order = -1` switches it off; `hyperdiffusion = ν₄` adds `-ν₄ L(L(vn))`
to every dynamics step.

Differences to ICON: the pressure gradient over orography is that of
`igradp_method = 1` along the levels, without the truncation over steep slopes, or with
`pressure_gradient = :height` at constant height everywhere (`π'` of both cells interpolated
linearly in height to the edge, extrapolated below the ground of a cell; ICON's `igradp_method = 3`
only where the slope is steep, with a hydrostatic extrapolation), the divergence damping and the θ
diffusion without the metric correction; the vn diffusion is `itype_vn_diffu = 1`;
no extra CFL diffusion (`lextra_diffu`); no tracer or physics coupling
(`ndyn_substeps` only sets when the diffusion is called); no nest-boundary
terms. Started from ICON's state, one step agrees with ICON to 0.1–1 % of
ICON's own change of `w`, `ρ`, `θ_v` and `exner`. Serial.
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
    flux_edges::I        # 5 × ne: the edge, the other edges of its first and of its second cell
    flux_weights::M      # e_flx_avg: weights of the averaged vn of the mass flux
    vertex_edges::I
    curl::M              # t_{e,v} d_e / (a A_v)
    vertex_cells::I
    vertex_weight::M     # K_{i,v} / A_v (cells_aw_verts)
    edge_coriolis::V
    green_gauss::NTuple{3, M}   # n_{e,i} l_e n_e / A_i on the unit sphere: Green–Gauss gradient at the cells
    edge_vectors::M      # 12 × ne: x_e - x_l, x_e - x_r, n_e, t_e (unit sphere)
    cell_areas::V        # a² A_i
    dz::M                # layer thickness, cells × layers
    dzh::M               # distance of the layer centres at the interfaces, cells × interfaces
    wlow::M              # weight of the layer below at the interfaces, cells × interfaces
    orography::Bool      # terrain-following levels (the metric terms)
    metric::NamedTuple   # edges: Δz_e, wlow_e, wgtfacq_e, ∂z/∂n, ∂z/∂t; cells: z_full; host: cells, edges
    rayleigh::V          # Klemp's coefficient of w at the interfaces (rayleigh_w)
    vertical::NamedTuple # reference state and vertical metric coefficients
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

function ICONNonhydrostatic(grid::TRiSKGrid, vgrid::VerticalGrid; radius::Real=6.371229e6, omega::Real=7.29212e-5,
                            coriolis=x -> 2omega * x[3],
                            beta::Real=0.65, damping_height::Real=45000.0, damping_rate::Real=0.05,
                            itime_scheme::Integer=4, divdamp_order::Integer=4, divdamp_type::Integer=3,
                            divdamp_fac::Real=0.0025, divdamp_fac2::Real=0.004, divdamp_fac3::Real=0.004,
                            divdamp_fac4::Real=0.004, divdamp_z::Real=32500.0, divdamp_z2::Real=40000.0,
                            divdamp_z3::Real=60000.0, divdamp_z4::Real=80000.0, divdamp_fac_o2::Real=0.0,
                            divdamp_trans_start::Real=12500.0, divdamp_trans_end::Real=17500.0,
                            veladv_offctr::Real=0.25, rhotheta_offctr::Real=-0.1, exner_expol::Real=1 / 3,
                            rbf_scale::Real=0.5, hdiff_order::Integer=5, ndyn_substeps::Integer=5,
                            hdiff_efdt_ratio::Real=36.0, hdiff_w_efdt_ratio::Real=15.0, hdiff_smag_fac::Real=0.015,
                            hdiff_smag_fac2::Real=2e-6 * (1600 + 25000 + sqrt(1600 * (1600 + 50000))),
                            hdiff_smag_fac3::Real=0.0, hdiff_smag_fac4::Real=1.0, hdiff_smag_z::Real=32500.0,
                            hdiff_smag_z2::Real=1600 + 50000 + sqrt(1600 * (1600 + 50000)),
                            hdiff_smag_z3::Real=50000.0, hdiff_smag_z4::Real=90000.0, lhdiff_vn::Bool=true,
                            lhdiff_w::Bool=true, lhdiff_temp::Bool=true, rbf_vec_scale_v::Real=0.5,
                            divavg_cntrwgt::Real=0.5, hyperdiffusion::Real=0.0, surface_height=nothing,
                            z_half=nothing, coordinate::Symbol=:sleve, flat_height::Real=16000.0,
                            decay_scale::Real=4000.0, decay_exponent::Real=1.35, pressure_gradient::Symbol=:levels,
                            miura_reference::Bool=true, backend=KernelAbstractions.CPU())
    pressure_gradient in (:levels, :height) || throw(ArgumentError("pressure_gradient must be :levels or :height"))
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
    flux_edges, flux_weights = flux_average(grid, divergence; cntrwgt=divavg_cntrwgt)
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
    # terrain-following levels: the interfaces of the cells, at the edges by c_lin_e
    zi = model_levels(vgrid, grid.centers, surface_height, z_half; coordinate, flat_height, decay_scale, decay_exponent)
    orography = any(i -> any(j -> zi[i, j] != vgrid.z_half[j], 1:(nlev + 1)), 1:nc)
    cm = column_metrics(zi)
    em = column_metrics([edge_height[1, e] * zi[grid.edge_cells[1, e], j] + edge_height[2, e] * zi[grid.edge_cells[2, e], j]
                         for e in 1:ne, j in 1:(nlev + 1)])
    z_top, zh, nl = vgrid.z_half[end], vgrid.z_half, nlev
    # Klemp et al. (2008) at the interior interfaces from damping_height up (rayleigh_type = 2), reference levels
    rayleigh = [2 <= j <= nl && zh[j] >= damping_height ?
                damping_rate * (1 - tanh(3.8 * (z_top - zh[j]) / max(1e-6, z_top - damping_height))) : 0.0
                for j in 1:(nl + 1)]
    # reference atmosphere of ICON at the layers and interfaces
    ref_π(z) = reference_atmosphere(z)[2]
    θref = map(z -> reference_atmosphere(z)[1] / ref_π(z), cm.z_full)
    dπref = map(z -> -ICON_GRAV / (ICON_CPD * reference_atmosphere(z)[1] / ref_π(z)), cm.z_half)
    # ρ_ref, θ_ref of the Miura reconstruction (ICON rho_ref_mc, theta_ref_mc at the cells,
    # rho_ref_me, theta_ref_me at the edges at their own heights); zero: the full fields are reconstructed
    ref_ρ(z) = P0 * ref_π(z)^(ICON_CPD / ICON_RD) / (ICON_RD * reference_atmosphere(z)[1])
    ref_θ(z) = reference_atmosphere(z)[1] / ref_π(z)
    miura(f, x) = miura_reference ? map(f, x) : zero(x)
    ρref_c, θref_c, ρref_e, θref_e = miura(ref_ρ, cm.z_full), miura(ref_θ, cm.z_full), miura(ref_ρ, em.z_full),
                                     miura(ref_θ, em.z_full)
    # second-order ∂w/∂z at the interfaces (coeff1_dwdz, coeff2_dwdz) and the quadratic extrapolation
    # of a layer field to the ground (wgtfacq) at the cells and the edges
    c1dwdz, c2dwdz = zeros(nc, nl + 1), zeros(nc, nl + 1)
    for i in 1:nc, j in 2:nl
        c1dwdz[i, j] = cm.dz[i, j - 1] / cm.dz[i, j] / (cm.z_half[i, j + 1] - cm.z_half[i, j - 1])
        c2dwdz[i, j] = cm.dz[i, j] / cm.dz[i, j - 1] / (cm.z_half[i, j + 1] - cm.z_half[i, j - 1])
    end
    qc = [ground_extrapolation(cm, i)[n] for i in 1:nc, n in 1:3]
    qe = [ground_extrapolation(em, e)[n] for e in 1:ne, n in 1:3]
    # slopes of the levels at the edges: normal from the cells, tangential from the vertices (cells_aw_verts)
    vw = grid.vertex_kites ./ grid.vertex_areas'
    zv = zeros(nv, nl)
    for v in 1:nv, n in axes(grid.vertex_cells, 1)
        c = grid.vertex_cells[n, v]
        c > 0 && (zv[v, :] .+= vw[n, v] .* cm.z_full[c, :])
    end
    ddxn = [(cm.z_full[grid.edge_cells[2, e], k] - cm.z_full[grid.edge_cells[1, e], k]) / (a * d[e]) for e in 1:ne, k in 1:nl]
    zlev, zwt = height_interpolation(grid.edge_cells, cm.z_full, em.z_full)
    ddxt = [tangent_sign[e] * (zv[grid.edge_vertices[2, e], k] - zv[grid.edge_vertices[1, e], k]) / (a * l[e])
            for e in 1:ne, k in 1:nl]
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
            pg=z(ne, nlev), va=z(ne, nlev), F=z(ne, nlev), Fθ=z(ne, nlev), hρθ=z(nc, nlev), ρexpl=z(nc, nlev), ρθexpl=z(nc, nlev),
            θi=z(nc, nlev + 1), zβ=z(nc, nlev), ρx=z(nc, nlev), πx=z(nc, nlev), lower=z(nc, nlev + 1),
            diag=z(nc, nlev + 1), cprime=z(nc, nlev + 1), dprime=z(nc, nlev + 1), wmass=dev(vertical_mass(vgrid, :lumped)),
            ρi=z(nc, nlev + 1), θp=z(nc, nlev + 1), π0=z(nc, nlev), δ=z(nc, nlev), δ2=z(nc, nlev), G=z(ne, nlev),
            ζ=z(nv, nlev), Lvn=z(ne, nlev), uvert=ntuple(_ -> z(nv, nlev), 3), uvert2=ntuple(_ -> z(nv, nlev), 3),
            kh=z(ne, nlev), n2e=z(ne, nlev), Lw=z(nc, nlev + 1), substep=Ref(0),
            hd=z(ne, nlev), wcf=z(nc, nlev), wch=z(nc, nlev + 1), πp=z(nc, nlev), dπdz=z(nc, nlev),
            ρpr=z(nc, nlev), θpr=z(nc, nlev), y0=(vn=z(ne, nlev), w=z(nc, nlev + 1), ρ=z(nc, nlev), ρθ=z(nc, nlev), exner=z(nc, nlev)))
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
    k4w = hdiff_w_efdt_ratio > 0 ? min(1 / 48, ndyn_substeps / (36 * hdiff_w_efdt_ratio)) : 0.0
    n2w = [jb !== nothing && jb < j <= nlev ? ((zh[j] - zh[jb]) / (zh[nlev] - zh[jb]))^4 / 12 : 0.0
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
                              dev(flux_edges), dev(flux_weights), dev(grid.vertex_edges), dev(curl), dev(grid.vertex_cells),
                              dev(grid.vertex_kites ./ grid.vertex_areas'), dev([Float64(coriolis(x)) for x in grid.edge_points]),
                              map(dev, green_gauss), dev(edge_vectors), dev(a^2 .* grid.cell_areas),
                              dev(cm.dz), dev(cm.dzh), dev(cm.wlow), orography,
                              (dze=dev(em.dz), rdz=dev(1 ./ cm.dz), wlowe=dev(em.wlow), qe=dev(qe), ddxn=dev(ddxn), ddxt=dev(ddxt),
                               z_full=dev(cm.z_full), zlev=dev(zlev), zwt=dev(zwt),
                               pgrad=pressure_gradient == :height ? 2 : 1, cells=cm, edges=em),
                              dev(rayleigh),
                              (θref=dev(θref), πref=dev(map(ref_π, cm.z_full)), dπref=dev(dπref), c1dwdz=dev(c1dwdz),
                               c2dwdz=dev(c2dwdz), wgtfacq=dev(qc), ρref_c=dev(ρref_c), θref_c=dev(θref_c),
                               ρref_e=dev(ρref_e), θref_e=dev(θref_e)), work, a, Float64(beta),
                              dev(divdamp), dev(divdamp_o2), dev(divdamp_3d),
                              (0.5 - veladv_offctr, 0.5 + veladv_offctr),
                              (0.5 - rhotheta_offctr, 0.5 + rhotheta_offctr), Int(itime_scheme),
                              Float64(exner_expol), Float64(hyperdiffusion), Int(hdiff_order), Int(ndyn_substeps),
                              diffusion, 0, backend)
end

"State `(vn, w, ρ, ρθ, exner)` of the model, zero; the Exner pressure is prognostic as in ICON."
function zero_state(m::ICONNonhydrostatic)
    z = KernelAbstractions.zeros(m.backend, Float64, num_cells(m.grid), num_layers(m.vgrid))
    return (; invoke(zero_state, Tuple{CGridNonhydrostatic}, m)..., exner=z)
end

"""
    initial_state(m::ICONNonhydrostatic, profile)

As for the other C-grid models, in ICON's hydrostatic balance (`balanced_column`),
with the Exner pressure of the equation of state.
"""
function initial_state(m::ICONNonhydrostatic, profile)
    y = invoke(initial_state, Tuple{CGridNonhydrostatic, Any}, m, profile)
    return (; y..., exner=to_backend(m.backend, (ICON_RD .* Array(y.ρθ) ./ P0) .^ (ICON_RD / ICON_CVD)))
end

"""
Column of `profile(z) -> (T, p, ...)` in ICON's discrete hydrostatic balance
(`hydro_adjust`): `T` at the layer centres and `π` of the lowest layer kept,
above `θ_j ∂π'/∂z + θ'_j ∂π_ref/∂z = 0` at every interface with the reference
state of the vertical pressure gradient of the column solve (`icon_column_matrix_kernel!`).
"""
function balanced_column(m::ICONNonhydrostatic, profile, i)
    v, nlev, κ = column_grid(m, i), num_layers(m.vgrid), ICON_RD / ICON_CPD
    θref, πref, dπref = (Array(view(x, i, :)) for x in (m.vertical.θref, m.vertical.πref, m.vertical.dπref))
    T = [profile(z)[1] for z in v.z_full]
    πk = [(profile(z)[2] / P0)^κ for z in v.z_full]
    for j in 2:nlev
        πj, wl = πk[j], v.wlow[j]
        for _ in 1:50
            θl, θu = T[j - 1] / πk[j - 1], T[j] / πj
            θi = wl * θl + (1 - wl) * θu
            θp = wl * (θl - θref[j - 1]) + (1 - wl) * (θu - θref[j])
            πj = πk[j - 1] + πref[j] - πref[j - 1] - v.dzh[j] * θp * dπref[j] / θi
        end
        πk[j] = πj
    end
    ρθ = @. P0 / ICON_RD * πk^(ICON_CVD / ICON_RD)
    return ρθ ./ (T ./ πk), ρθ
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

"""
    flux_average(grid, divergence; cntrwgt=0.5, niter=1000) -> (edges, weights)

ICON's averaged normal velocity of the mass flux (`e_flx_avg`, `z_vn_avg`):
`v̄_e = Σ_n w_{n,e} vn_{edges[n,e]}` over the edge, the two other edges of its
first cell and the two of its second, such that the divergence of `v̄` is the
cell average `C` of the divergence of `vn`, `D v̄ = C D vn`, with ICON's
averaging `C` (`c_bln_avg`): the weight `cntrwgt` of the cell and those of
its three neighbours that reproduce linear fields in the tangent plane,
corrected iteratively (relaxation 0.46, centre within `cntrwgt ± 0.003`) for
the conservation of the area integral, `Σ_j A_j C_{j→i} = A_i`. `D v̄ = C D vn`
fixes the weights of the outer edges; the central weight follows from it in
each of the two cells and is their mean; the weights are scaled to reproduce
a constant vector. `cntrwgt = 1` is the identity.
"""
function flux_average(grid::TRiSKGrid, divergence::AbstractMatrix; cntrwgt::Real=0.5, niter::Integer=1000)
    nc, ne = num_cells(grid), num_edges(grid)
    ce, ec = grid.cell_edges, grid.edge_cells
    across(i, e) = ec[1, e] == i ? ec[2, e] : ec[1, e]
    slot(i, e) = findfirst(==(e), view(ce, :, i))
    nb = [across(i, ce[k, i]) for k in 1:3, i in 1:nc]
    back = [slot(nb[k, i], ce[k, i]) for k in 1:3, i in 1:nc]   # position of i in the list of its neighbour k
    # cell average: cntrwgt at the cell, the neighbours with zero first moment in the tangent plane
    C = zeros(4, nc)
    for i in 1:nc
        x = grid.centers[i]
        t1 = normalize(cross(x, abs(x[3]) < 0.9 ? SVector(0.0, 0.0, 1.0) : SVector(1.0, 0.0, 0.0)))
        t2 = cross(x, t1)
        p = [grid.centers[nb[k, i]] - x for k in 1:3]
        M = SMatrix{3, 3}(1.0, dot(t1, p[1]), dot(t2, p[1]), 1.0, dot(t1, p[2]), dot(t2, p[2]),
                          1.0, dot(t1, p[3]), dot(t2, p[3]))
        C[1, i] = cntrwgt
        C[2:4, i] .= M \ SVector(1 - cntrwgt, 0.0, 0.0)
    end
    A = grid.cell_areas
    resid = zeros(nc)
    for it in 1:niter
        for i in 1:nc
            resid[i] = (A[i] * C[1, i] + sum(A[nb[k, i]] * C[1 + back[k, i], nb[k, i]] for k in 1:3)) / A[i] - 1
        end
        if it < niter
            for i in 1:nc
                C[1, i] -= 0.46 * resid[i]
                for k in 1:3
                    C[1 + k, i] -= 0.46 * resid[nb[k, i]]
                end
                C[:, i] .-= (sum(view(C, :, i)) - 1) / 4
                C[1, i] = clamp(C[1, i], cntrwgt - 0.003, cntrwgt + 0.003)
            end
        else
            C[1, :] .-= resid
        end
    end
    D(i, e) = divergence[slot(i, e), i]
    Cn(i, j) = C[1 + findfirst(==(j), view(nb, :, i)), i]     # weight of the neighbour j in the average of i
    others(i, e) = [f for f in view(ce, :, i) if f != e]
    # weight of the edge g of cell i for the edge f of i (f ≠ g), from the condition at the cell across g
    outer(g, i, f) = (j = across(i, g); Cn(j, i) * D(i, f) / D(j, g))
    edges, weights = zeros(Int, 5, ne), zeros(5, ne)
    for e in 1:ne
        c1, c2 = ec[1, e], ec[2, e]
        a, b = others(c1, e), others(c2, e)
        edges[:, e] .= (e, a..., b...)
        central(i, j) = (C[1, i] * D(i, e) + Cn(i, j) * D(j, e) - sum(D(i, g) * outer(g, i, e) for g in others(i, e))) /
                        D(i, e)
        weights[1, e] = (central(c1, c2) + central(c2, c1)) / 2
        weights[2:3, e] .= [outer(e, c1, f) for f in a]
        weights[4:5, e] .= [outer(e, c2, f) for f in b]
        n = grid.edge_normals[e]
        weights[:, e] ./= sum(weights[k, e] * dot(n, grid.edge_normals[edges[k, e]]) for k in 1:5)
    end
    return edges, weights
end

ICONNonhydrostatic(dm::Plex, vgrid::VerticalGrid; kwargs...) = ICONNonhydrostatic(TRiSKGrid(dm), vgrid; kwargs...)
