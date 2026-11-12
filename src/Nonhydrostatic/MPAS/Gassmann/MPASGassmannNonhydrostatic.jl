"""
    MPASGassmannNonhydrostatic(grid::TRiSKGrid, vgrid::VerticalGrid; radius, omega,
                               coriolis=x -> 2omega * x[3], beta=0.65,
                               damping_height=Inf, damping_rate=0, kinetic_energy=:cell, hyperdiffusion=0,
                               pv_flux=:iap, vertical_advection=:iap, trisk_weights=:iap, smagorinsky=0.15,
                               vertical_theta=:upwind, time_stepping=:iap, vertical_solver=:five_band,
                               surface_height=nothing, z_half=nothing, coordinate=:sleve, flat_height=16000,
                               decay_scale=4000, decay_exponent=1.35, pressure_gradient=:levels, backend=CPU())

MPAS-Gassmann: the ICON-IAP form (Gassmann 2013) of the dry compressible Euler
equations on the extruded C grid of a Voronoi mesh, shallow atmosphere, with
orography (see below); `pv_flux=:trisk`, `trisk_weights=:exact`, `kinetic_energy=:mixed` give
the horizontal operators of MPAS instead. Lorenz staggering as in ICON and ICON-IAP
(Gassmann 2013):

| variable | location | array |
|---|---|---|
| `vn` normal velocity | lateral faces (edge × layer) | `(ne, nlev)` |
| `w` vertical velocity | horizontal faces (cell × level) | `(nc, nlev+1)` |
| `ρ`, `ρθ` | cells (cell × layer) | `(nc, nlev)` |

with the Exner pressure `π = (R ρθ / p₀)^(R/c_v)`. Per layer the horizontal
operators are those of `TRiSKShallowWater` with `ρ` in place of `h`: the PV
flux `Σ w_{e,e'} l_{e'} ρ_{e'} vn_{e'} q_{e,e'} / d_e`, the cell kinetic
energy, divergence and gradient. With `pv_flux = :iap` (default) the PV is
used as in ICON-IAP (`i_cori_method = 4`, `trisk_pv_edges`): the rhombus PV
`q_e = (ζ_e + f_e)/ρ_e` (dual-area averages of the vertex values, `f` at the
edge point), `q_{e,e'}` that of the edge between `e` and `e'`, the mean
`(q_e + q_{e'})/2` only for the opposite edge of a hexagon. `pv_flux = :trisk`
takes `q = (ζ + f)/ρ` at the vertices, `q_e` their mean and always
`q_{e,e'} = (q_e + q_{e'})/2`. Vertically the differences are centred on the staggered grid,
`ρ` linearly interpolated to the interfaces. `θ` at the interfaces in the
vertical `ρθ` flux is with `vertical_theta = :upwind` (default) that of
ICON-IAP (`ltheta_up_vert`, `impl_vert_adv_scalar_5diag`): `θ` of the old
state advected vertically with the mass flux of the stage state over half the
stage, implicitly (Crank–Nicolson, a pentadiagonal system per column), and its
third-order upwind interface values (`theta_stencil`; on stretched grids the
weight of the upwind cell is chosen so that constants are kept). The vertical
pressure gradient keeps `θ` linearly interpolated: with the upwind values also
there (`:iap`, as ICON-IAP, which conserves energy in the exchange with the
vertical kinetic energy) the stencil switches with the sign of round-off `w`
and a column at rest does not stay at rest. `:centred` takes the linear
interpolation in the flux as well.

    ∂vn/∂t = PV flux - ∂_n K - c_p θ_e ∂_n π - η w̃_e
    ∂w/∂t  = η ṽn - ∂K/∂z - c_p θ ∂π/∂z - g - r(z) w
    ∂ρ/∂t  = - ∇ ⋅ (ρ_e vn) - ∂(ρ w)/∂z
    ∂ρθ/∂t = - ∇ ⋅ (ρ_e θ_e vn) - ∂(ρ θ w)/∂z

Time stepping (`step!`): with `time_stepping = :iap` (default) the two-stage
predictor–corrector of ICON-IAP (`iap_step!`: both stages from the old state
over the whole step, the slow tendencies of the second the mean of both, the
horizontal pressure gradient with the Exner pressure off-centred in time), with
`:wicker_skamarock` the three stages of Wicker and Skamarock (2002). Each stage is
horizontally explicit (forward–backward: the mass fluxes use the new `vn`) and
vertically implicit for `w`, `ρ`, `ρθ` with the off-centring `beta` and the
linearized equation of state. With `vertical_solver = :five_band` (default) as
ICON-IAP (`divergent_modes_5band`, Gassmann and Herzog 2008, `five_band_solve_kernel!`): the
unknown is the vertical mass flux, time-centred in the products of density and `w`, the
vertical kinetic energy `w⁰ w/2` implicit, one pentadiagonal system per column; ICON-IAP takes
`beta = c_v/c_p`. `:tridiagonal` solves for `w` with `K_w` explicit, one tridiagonal system per
column (as in ICON, Zängl et al. 2015). `r(z)` is a Rayleigh damping of `w` above
`damping_height` (rate `damping_rate sin²`).

The momentum advection is in the 3D vector-invariant form (`ω × u` and
the gradient of `K = K_h + K_w`) as in ICON-IAP (`vorticity_tendencies`,
`kinetic_energy`) and CGSEM (CGDycore.jl), with the horizontal vorticity
`η = ∂vn/∂z - ∂w/∂n` at the edges of the interior interfaces and
`K_w = (dzh_k w_k² + dzh_{k+1} w_{k+1}²) / 4dz_k`. With
`vertical_advection = :iap` (default) the vorticity term is that of the mass
fluxes, `w̃_e = M_e/ρ_e` with `M_e = Σ_i (d_{e,i}/d_e) ρ_i w_i` at the
interfaces (the mass flux of the vertical solver) in the `vn` equation, and
`ṽn = (ρ_{e,k-1} vn_{k-1} + ρ_{e,k} vn_k) / 2ρ_e` with the cell inner product
`Σ_e (l_e d_{e,i}/A_i)` (ICON `e_inn_c`) in the `w` equation.
`:vector_invariant` is the same with `ρ = 1` in these averages. In both the
exchange between horizontal and vertical kinetic energy cancels exactly in
the energy `Σ_e l_e d_e ρ_e vn²/2 + Σ_i A_i ρ_i w²/2`. `:advective` uses
`-w ∂vn/∂z` and `-u ⋅ ∇w - w ∂w/∂z` instead.

`trisk_weights = :iap` (default) takes the TRiSK weights of ICON-IAP
(`iap_trisk_weights`, kite fractions of planar triangles), `:exact` those of
`TRiSKGrid` (exact spherical kites, exactly antisymmetric).

`smagorinsky` is the factor `hdiff_smag_fac` of the Smagorinsky diffusion of
ICON-IAP (`diff_diss_hori`, default 0.15, 0 switches it off): the coefficient
`K = hdiff_smag_fac A_e |D|` at the edges with the mean edge area
`A_e = 8πa²/n_edges` and the horizontal deformation
`|D|² = (∂_n vn - ∂_t vt)² + (∂_t vn + ∂_n vt)²` from the velocity vectors at
the cells (Perot reconstruction) and vertices, tapered to zero above
`damping_height` as ICON-IAP's `kmh_bg` (with `k2_updamp_coeff = 0`). The
diffusion of `vn` is `(1/ρ_e)[∇(ρKδ) - k × ∇(ρKζ)]⋅n` with `ρK` at cells and
vertices, which for constant `K` is the divergence of the deformation stress
`ρK D` (ICON-IAP discretizes that stress directly); its loss of kinetic energy
`Σ A_i ρK_i δ_i² + Σ A_v ρK_v ζ_v²` heats the cells (`ldiss_heat`,
`∂ρθ/∂t = Q/(c_p π)`). Like the hyperdiffusion it is evaluated once per step in
the old state.

`hyperdiffusion` is the coefficient `ν₄` (m⁴/s) of `-ν₄ L(L(vn))` with the
C-grid vector Laplacian `L(vn)_e = (δ_r - δ_l)/d_e - (ζ_{v2} - ζ_{v1})/l_e`
(`∇δ + k × ∇ζ`), evaluated once per step in the old state.

`kinetic_energy` mixes the cell kinetic energy with the vertex kinetic energy
as in `TRiSKShallowWater` (`:cell`, `:mixed` = 3/8 cell + 5/8 vertex as in
MPAS, Skamarock et al. 2012, `:vertex` or a weight); the pure cell kinetic
energy lets a strong jet become unstable on fine meshes (Hollingsworth et al.
1983).

**Orography**: `surface_height(lon, lat)` (or the heights of the cells) or `z_half` (cells ×
nlev+1) give terrain-following levels as in `ICONNonhydrostatic` (`terrain_following`). The
equations are along the levels: the horizontal fluxes times the layer thickness at the edges,
divided by that of the cell; the contravariant correction `w_c = u ⋅ ∇z` (cell inner product of
`vn ∂z/∂n`, interpolated to the interfaces, quadratically extrapolated to the ground) in the mass
flux of the column solves (`w = w_c` at the ground), in `θ` at the interfaces and in the vertical
coupling of `vn` (`w̃` of `w - w_c`), `w_c ∂w/∂z` added to the `w` tendency; the horizontal
pressure gradient `c_p θ_e (∂π'/∂n - ∂z/∂n ∂π'/∂z)` of the deviation from ICON's reference
atmosphere (`reference_atmosphere`), or with `pressure_gradient = :height` that of `π'` of both
cells interpolated linearly in height to the edge (`height_interpolation`, extrapolated below the
ground of a cell). `∇K` and `η` stay along the levels, so the exchange of
horizontal and vertical kinetic energy is exact only for flat ground. The Rayleigh damping and
the taper of the Smagorinsky diffusion are on the reference levels.

Arrays are `(n, nlev)` with the horizontal index fastest; every kernel runs
over `(n, nlev)` or, for the vertical solver, one work item per column.
"""
struct MPASGassmannNonhydrostatic{M, V, I, J, VK, BE} <: CGridNonhydrostatic
    grid::TRiSKGrid
    vgrid::VerticalGrid
    cell_edges::I
    divergence::M        # n_{e,i} l_e / (a A_i)
    kinetic::M           # α l_e d_{e,i} / (2 A_i)
    inner::M             # l_e d_{e,i} / A_i: cell inner product of edge fields (ICON e_inn_c)
    vertex_kinetic::VK   # nothing, or the vertex kinetic energy tables of TRiSKShallowWater
    edge_cells::I
    edge_vertices::I
    edge_weight::M       # d_{e,i} / d_e
    inverse_dual::V      # 1 / (a d_e)
    inverse_primal::V    # 1 / (a l_e)
    vertex_edges::I
    curl::M              # t_{e,v} d_e / (a A_v)
    vertex_cells::I
    vertex_weight::M     # K_{i,v} / A_v
    coriolis::V          # f at the vertices
    trisk_index::J       # (ne, nt, 3) Int32: TRiSK edge e' and the edges a, b of its PV (q_a + q_b)/2
    trisk::M             # (ne, nt): w_{e,e'} l_{e'} / 2d_e
    rhombus::M           # A_v / (A_v1 + A_v2) at the edge vertices
    edge_coriolis::V     # f at the edge points
    perot::NTuple{3, M}  # n_{e,i} l_e (x_e - x_i) / A_i, x_e the edge midpoint: velocity vector at the cells
    barycentric::M       # barycentric coordinates of the vertices in the triangles of the cell centres
    edge_frame::M        # (n_e, t_e) as 6 × ne: normal from left to right, tangent from vertex 1 to 2
    cell_vertices::I
    cell_kites::M        # K_{i,v} / A_i
    cell_edge_area::M    # l_e d_{e,i} / 2A_i: cell average of edge quantities
    smagorinsky::V       # hdiff_smag_fac A_e (1 - taper) per layer (m²)
    cell_areas::V        # a² A_i
    dz::M                # layer thickness, cells × layers
    dzh::M               # distance of the layer centres at the interfaces, cells × interfaces
    wlow::M              # weight of the layer below at the interfaces, cells × interfaces
    orography::Bool      # terrain-following levels (the metric terms)
    metric::NamedTuple   # edges: dz, dzh, ∂z/∂n; cells: 1/dz, z_full, π_ref, wgtfacq; host: cells, edges
    rayleigh::V          # damping rate at the interfaces
    work::NamedTuple
    radius::Float64
    beta::Float64
    hyperdiffusion::Float64
    iap_pv::Bool
    vertical::Int        # 0 advective, 1 vector invariant, 2 ICON-IAP mass fluxes
    vertical_theta::Int  # θ at the interfaces: 0 centred, 1 upwind in the flux, 2 upwind also in ∂π/∂z (ICON-IAP)
    iap_stepping::Bool   # time_stepping = :iap
    five_band::Bool      # vertical_solver = :five_band
    backend::BE
end

function MPASGassmannNonhydrostatic(grid::TRiSKGrid, vgrid::VerticalGrid; radius::Real=6.37122e6,
                                    omega::Real=7.292e-5, coriolis=x -> 2omega * x[3], beta::Real=0.65,
                                    damping_height::Real=Inf,
                                    damping_rate::Real=0.0, kinetic_energy::Union{Symbol, Real}=:cell,
                                    hyperdiffusion::Real=0.0, pv_flux::Symbol=:iap, vertical_advection::Symbol=:iap,
                                    trisk_weights::Symbol=:iap, smagorinsky::Real=0.15, vertical_theta::Symbol=:upwind,
                                    time_stepping::Symbol=:iap, vertical_solver::Symbol=:five_band,
                                    surface_height=nothing, z_half=nothing, coordinate::Symbol=:sleve,
                                    flat_height::Real=16000.0, decay_scale::Real=4000.0, decay_exponent::Real=1.35,
                                    pressure_gradient::Symbol=:levels, backend=KernelAbstractions.CPU())
    pressure_gradient in (:levels, :height) || throw(ArgumentError("pressure_gradient must be :levels or :height"))
    time_stepping in (:iap, :wicker_skamarock) || throw(ArgumentError("time_stepping must be :iap or :wicker_skamarock"))
    vertical_solver in (:five_band, :tridiagonal) ||
        throw(ArgumentError("vertical_solver must be :five_band or :tridiagonal"))
    vertical_solver == :five_band && vertical_advection == :advective &&
        throw(ArgumentError("vertical_solver = :five_band needs the vector-invariant vertical advection"))
    pv_flux in (:iap, :trisk) || throw(ArgumentError("pv_flux must be :iap or :trisk"))
    vertical = get(Dict(:advective => 0, :vector_invariant => 1, :iap => 2), vertical_advection) do
        throw(ArgumentError("vertical_advection must be :iap, :vector_invariant or :advective"))
    end
    vtheta = get(Dict(:centred => 0, :upwind => 1, :iap => 2), vertical_theta) do
        throw(ArgumentError("vertical_theta must be :upwind, :iap or :centred"))
    end
    a = Float64(radius)
    sw = TRiSKShallowWater(grid; coriolis, kinetic_energy, backend)
    nc, ne, nv, nlev = num_cells(grid), num_edges(grid), num_vertices(grid), num_layers(vgrid)
    z_top = vgrid.z_half[end]
    rayleigh = map(vgrid.z_half) do z
        z > damping_height ? damping_rate * sin(π / 2 * (z - damping_height) / (z_top - damping_height))^2 : 0.0
    end
    dev(x) = to_backend(backend, x)
    zeros2(n, m) = KernelAbstractions.zeros(backend, Float64, n, m)
    work = (θ=zeros2(nc, nlev), π=zeros2(nc, nlev), K=zeros2(nc, nlev), qv=zeros2(nv, nlev), Kv=zeros2(nv, nlev),
            ζv=zeros2(nv, nlev), ρv=zeros2(nv, nlev), qe=zeros2(ne, nlev),
            η=zeros2(ne, nlev + 1),
            F=zeros2(ne, nlev), Fθ=zeros2(ne, nlev), tvn=zeros2(ne, nlev), tw=zeros2(nc, nlev + 1),
            ρexpl=zeros2(nc, nlev), ρθexpl=zeros2(nc, nlev), cprime=zeros2(nc, nlev + 1),
            dprime=zeros2(nc, nlev + 1), wmass=to_backend(backend, vertical_mass(vgrid, :lumped)), δ=zeros2(nc, nlev), ζ=zeros2(nv, nlev), Lvn=zeros2(ne, nlev),
            hd=zeros2(ne, nlev), hρθ=zeros2(nc, nlev), uc=ntuple(_ -> zeros2(nc, nlev), 3),
            uv=ntuple(_ -> zeros2(nv, nlev), 3), Ke=zeros2(ne, nlev), ρKc=zeros2(nc, nlev), ρKv=zeros2(nv, nlev),
            θi=zeros2(nc, nlev + 1), ρe=zeros2(ne, nlev), Fs=zeros2(ne, nlev), θm=zeros2(nc, nlev),
            π0=zeros2(nc, nlev), γ0=zeros2(nc, nlev),
            band=KernelAbstractions.zeros(backend, Float64, nc, nlev, 2),
            band5=KernelAbstractions.zeros(backend, Float64, nc, nlev + 1, 4),
            tvn1=zeros2(ne, nlev), tw1=zeros2(nc, nlev + 1), θs=zeros2(nc, nlev), πm=zeros2(nc, nlev),
            πold=zeros2(nc, nlev), started=Ref(false), wcf=zeros2(nc, nlev), wch=zeros2(nc, nlev + 1),
            ŵ=zeros2(nc, nlev + 1), πp=zeros2(nc, nlev), dπdz=zeros2(nc, nlev),
            y0=(vn=zeros2(ne, nlev), w=zeros2(nc, nlev + 1), ρ=zeros2(nc, nlev), ρθ=zeros2(nc, nlev)))
    ew = grid.cell_distances ./ grid.dual_lengths'
    edge_weight = dev(ew)
    # terrain-following levels, at the edges with the edge weights; slopes of the levels at the edges
    zi = model_levels(vgrid, grid.centers, surface_height, z_half; coordinate, flat_height, decay_scale, decay_exponent)
    orography = any(i -> any(j -> zi[i, j] != vgrid.z_half[j], 1:(nlev + 1)), 1:nc)
    cm = column_metrics(zi)
    em = column_metrics([ew[1, e] * zi[grid.edge_cells[1, e], j] + ew[2, e] * zi[grid.edge_cells[2, e], j]
                         for e in 1:ne, j in 1:(nlev + 1)])
    ddxn = [(cm.z_full[grid.edge_cells[2, e], k] - cm.z_full[grid.edge_cells[1, e], k]) / (a * grid.dual_lengths[e])
            for e in 1:ne, k in 1:nlev]
    zlev, zwt = height_interpolation(grid.edge_cells, cm.z_full, em.z_full)
    metric = (dz=dev(em.dz), dzh=dev(em.dzh), ddxn=dev(ddxn), rdz=dev(1 ./ cm.dz), z_full=dev(cm.z_full),
              zlev=dev(zlev), zwt=dev(zwt), pgrad=pressure_gradient == :height ? 2 : 1,
              πref=dev(map(z -> reference_atmosphere(z)[2], cm.z_full)),
              q=dev([ground_extrapolation(cm, i)[n] for i in 1:nc, n in 1:3]), cells=cm, edges=em)
    inner = zeros(size(grid.cell_edges))
    for i in 1:nc, j in axes(grid.cell_edges, 1)
        e = grid.cell_edges[j, i]
        side = grid.edge_cells[1, e] == i ? 1 : 2
        grid.cell_signs[j, i] != 0 &&
            (inner[j, i] = grid.primal_lengths[e] * grid.cell_distances[side, e] / grid.cell_areas[i])
    end
    rhombus = grid.vertex_areas[grid.edge_vertices]
    rhombus ./= sum(rhombus; dims=1)
    pv_edges = pv_flux == :iap ? trisk_pv_edges(grid) : zeros(Int, size(grid.trisk_edges))
    trisk_weights in (:iap, :exact) || throw(ArgumentError("trisk_weights must be :iap or :exact"))
    W = trisk_weights == :iap ? iap_trisk_weights(grid) : grid.trisk_weights
    # TRiSK tables as in ICON with the edge index first (coalesced on GPUs), the PV of each neighbour
    # by two edges, (q_e + q_e')/2 or q_p twice, the 1/2 in the weight; padding points to e itself
    nt = size(grid.trisk_edges, 1)
    trisk_index = zeros(Int32, ne, nt, 3)
    trisk = zeros(ne, nt)
    for e in 1:ne, k in 1:nt
        e′, p = grid.trisk_edges[k, e], pv_edges[k, e]
        trisk[e, k] = W[k, e] * grid.primal_lengths[e′] / (2 * grid.dual_lengths[e])
        trisk_index[e, k, :] .= trisk[e, k] == 0 ? (e, e, e) : p == 0 ? (e′, e, e′) : (e′, p, p)
    end
    # Smagorinsky diffusion: Perot reconstruction, edge frames, coefficient per layer
    perot = ntuple(_ -> zeros(size(grid.cell_edges)), 3)
    cell_edge_area = zeros(size(grid.cell_edges))
    for i in 1:nc, j in axes(grid.cell_edges, 1)
        grid.cell_signs[j, i] == 0 && continue
        e, x = grid.cell_edges[j, i], grid.centers[i]
        r = normalize(grid.vertices[grid.edge_vertices[1, e]] + grid.vertices[grid.edge_vertices[2, e]]) - x
        r -= dot(r, x) * x
        p = grid.cell_signs[j, i] * grid.primal_lengths[e] / grid.cell_areas[i] .* r
        foreach(c -> perot[c][j, i] = p[c], 1:3)
        cell_edge_area[j, i] = inner[j, i] / 2
    end
    # barycentric coordinates of the vertices in the triangles of their cell centres
    barycentric = zeros(size(grid.vertex_cells))
    for v in 1:nv
        cs = [grid.vertex_cells[j, v] for j in axes(grid.vertex_cells, 1) if grid.vertex_kites[j, v] != 0]
        length(cs) == 3 || error("Smagorinsky diffusion needs three cells per vertex")
        X = hcat((grid.centers[c] for c in cs)...)
        λ = Vector(Matrix(X) \ Vector(grid.vertices[v]))
        λ ./= sum(λ)
        k = 0
        for j in axes(grid.vertex_cells, 1)
            grid.vertex_kites[j, v] != 0 && (barycentric[j, v] = λ[k += 1])
        end
    end
    frame = zeros(6, ne)
    for e in 1:ne
        x, n = grid.edge_points[e], grid.edge_normals[e]
        t = grid.vertices[grid.edge_vertices[2, e]] - grid.vertices[grid.edge_vertices[1, e]]
        t = normalize(t - dot(t, x) * x)
        frame[1:3, e] .= n
        frame[4:6, e] .= t
    end
    taper = map(vgrid.z_full) do z
        z > damping_height ? 1 - cos(π / 2 * (z - damping_height) / (vgrid.z_full[end] - damping_height)) : 0.0
    end
    smag = Float64(smagorinsky) * 8π * a^2 / ne .* (1 .- taper)
    Array(sw.trisk_edges) == grid.trisk_edges || error("serial TRiSK tables expected in global numbering")
    return MPASGassmannNonhydrostatic(grid, vgrid, sw.cell_edges, dev(Array(sw.divergence) ./ a), sw.kinetic,
                                      dev(inner),
                                      sw.vertex_kinetic, sw.edge_cells, sw.edge_vertices, edge_weight, dev(Array(sw.inverse_dual) ./ a),
                                      dev(1 ./ (a .* grid.primal_lengths)), sw.vertex_edges,
                                      dev(Array(sw.curl) ./ a), sw.vertex_cells, sw.vertex_height, sw.coriolis,
                                      dev(trisk_index), dev(trisk), dev(rhombus),
                                      dev([Float64(coriolis(x)) for x in grid.edge_points]), map(dev, perot), dev(barycentric),
                                      dev(frame),
                                      dev(grid.cell_vertices), dev(grid.cell_kites ./ grid.cell_areas'), dev(cell_edge_area),
                                      dev(smag), dev(a^2 .* grid.cell_areas),
                                      dev(cm.dz), dev(cm.dzh), dev(cm.wlow), orography, metric, dev(rayleigh), work, a,
                                      Float64(beta),
                                      Float64(hyperdiffusion), pv_flux == :iap, vertical, vtheta, time_stepping == :iap,
                                      vertical_solver == :five_band,
                                      backend)
end

"Former name of `MPASGassmannNonhydrostatic`."
const TRiSKNonhydrostatic = MPASGassmannNonhydrostatic
