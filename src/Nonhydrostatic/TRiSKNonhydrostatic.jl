"""
    TRiSKNonhydrostatic(grid::TRiSKGrid, vgrid::VerticalGrid; radius, omega, beta=0.65,
                        damping_height=Inf, damping_rate=0, kinetic_energy=:cell, hyperdiffusion=0,
                        pv_flux=:iap, vertical_advection=:iap, trisk_weights=:iap, smagorinsky=0.15,
                        vertical_theta=:upwind, backend=CPU())

Dry compressible Euler equations on the extruded C grid of a Voronoi mesh,
shallow atmosphere, flat ground. Lorenz staggering as in ICON and ICON-IAP
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

Time stepping (`step!`): the three stages of Wicker and Skamarock (2002), each
horizontally explicit (forward–backward: the mass fluxes use the new `vn`) and
vertically implicit for `w`, `ρ`, `ρθ` with the off-centring `beta` and the
linearized equation of state, one tridiagonal system for `w` per column (as
in ICON, Zängl et al. 2015). `r(z)` is a Rayleigh damping of `w` above
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

Arrays are `(n, nlev)` with the horizontal index fastest; every kernel runs
over `(n, nlev)` or, for the vertical solver, one work item per column.
"""
struct TRiSKNonhydrostatic{M, V, I, VK, BE}
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
    trisk_edges::I
    trisk::M             # w_{e,e'} l_{e'} / d_e
    pv_edges::I          # ICON-IAP PV usage of the TRiSK edges, 0 for the mean
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
    dz::V
    dzh::V
    wlow::V
    rayleigh::V          # damping rate at the interfaces
    work::NamedTuple
    radius::Float64
    beta::Float64
    hyperdiffusion::Float64
    iap_pv::Bool
    vertical::Int        # 0 advective, 1 vector invariant, 2 ICON-IAP mass fluxes
    vertical_theta::Int  # θ at the interfaces: 0 centred, 1 upwind in the flux, 2 upwind also in ∂π/∂z (ICON-IAP)
    backend::BE
end

const Rd = 287.0
const Cpd = 1004.0
const Cvd = Cpd - Rd
const P0 = 1.0e5
const GRAV = 9.80616

function TRiSKNonhydrostatic(grid::TRiSKGrid, vgrid::VerticalGrid; radius::Real=6.37122e6,
                             omega::Real=7.292e-5, beta::Real=0.65, damping_height::Real=Inf,
                             damping_rate::Real=0.0, kinetic_energy::Union{Symbol, Real}=:cell,
                             hyperdiffusion::Real=0.0, pv_flux::Symbol=:iap, vertical_advection::Symbol=:iap,
                             trisk_weights::Symbol=:iap, smagorinsky::Real=0.15, vertical_theta::Symbol=:upwind,
                             backend=KernelAbstractions.CPU())
    pv_flux in (:iap, :trisk) || throw(ArgumentError("pv_flux must be :iap or :trisk"))
    vertical = get(Dict(:advective => 0, :vector_invariant => 1, :iap => 2), vertical_advection) do
        throw(ArgumentError("vertical_advection must be :iap, :vector_invariant or :advective"))
    end
    vtheta = get(Dict(:centred => 0, :upwind => 1, :iap => 2), vertical_theta) do
        throw(ArgumentError("vertical_theta must be :upwind, :iap or :centred"))
    end
    a = Float64(radius)
    sw = TRiSKShallowWater(grid; coriolis=x -> 2omega * x[3], kinetic_energy, backend)
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
            dprime=zeros2(nc, nlev + 1), δ=zeros2(nc, nlev), ζ=zeros2(nv, nlev), Lvn=zeros2(ne, nlev),
            hd=zeros2(ne, nlev), hρθ=zeros2(nc, nlev), uc=ntuple(_ -> zeros2(nc, nlev), 3),
            uv=ntuple(_ -> zeros2(nv, nlev), 3), Ke=zeros2(ne, nlev), ρKc=zeros2(nc, nlev), ρKv=zeros2(nv, nlev),
            θi=zeros2(nc, nlev + 1), ρe=zeros2(ne, nlev), Fs=zeros2(ne, nlev), θm=zeros2(nc, nlev),
            band=KernelAbstractions.zeros(backend, Float64, nc, nlev, 5),
            y0=(vn=zeros2(ne, nlev), w=zeros2(nc, nlev + 1), ρ=zeros2(nc, nlev), ρθ=zeros2(nc, nlev)))
    edge_weight = dev(grid.cell_distances ./ grid.dual_lengths')
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
    trisk = [W[k, e] * grid.primal_lengths[grid.trisk_edges[k, e]] / grid.dual_lengths[e] for k in axes(W, 1), e in 1:ne]
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
    return TRiSKNonhydrostatic(grid, vgrid, sw.cell_edges, dev(Array(sw.divergence) ./ a), sw.kinetic,
                               dev(inner),
                               sw.vertex_kinetic, sw.edge_cells, sw.edge_vertices, edge_weight, dev(Array(sw.inverse_dual) ./ a),
                               dev(1 ./ (a .* grid.primal_lengths)), sw.vertex_edges,
                               dev(Array(sw.curl) ./ a), sw.vertex_cells, sw.vertex_height, sw.coriolis,
                               sw.trisk_edges, dev(trisk), dev(pv_edges), dev(rhombus),
                               dev([2omega * x[3] for x in grid.edge_points]), map(dev, perot), dev(barycentric),
                               dev(frame),
                               dev(grid.cell_vertices), dev(grid.cell_kites ./ grid.cell_areas'), dev(cell_edge_area),
                               dev(smag), dev(a^2 .* grid.cell_areas),
                               dev(vgrid.dz), dev(vgrid.dzh), dev(vgrid.wlow), dev(rayleigh), work, a, Float64(beta),
                               Float64(hyperdiffusion), pv_flux == :iap, vertical, vtheta, backend)
end

"State `(vn, w, ρ, ρθ)` of the model, zero."
function zero_state(m::TRiSKNonhydrostatic)
    nc, ne, nlev = num_cells(m.grid), num_edges(m.grid), num_layers(m.vgrid)
    z(n, k) = KernelAbstractions.zeros(m.backend, Float64, n, k)
    return (vn=z(ne, nlev), w=z(nc, nlev + 1), ρ=z(nc, nlev), ρθ=z(nc, nlev))
end

function launch!(m::TRiSKNonhydrostatic, kernel, ndrange, args...)
    kernel(m.backend)(args...; ndrange)
    KernelAbstractions.synchronize(m.backend)
    return nothing
end

# --- kernels -------------------------------------------------------------------

"Potential temperature, Exner pressure and horizontal kinetic energy at the cells."
@kernel function diagnose_kernel!(θ, πe, K, @Const(vn), @Const(ρ), @Const(ρθ), @Const(cell_edges), @Const(kinetic))
    i, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(cell_edges, 1)
        s += kinetic[j, i] * vn[cell_edges[j, i], k]^2
    end
    @inbounds K[i, k] = s
    @inbounds θ[i, k] = ρθ[i, k] / ρ[i, k]
    @inbounds πe[i, k] = (Rd * ρθ[i, k] / P0)^(Rd / Cvd)
end

"Vertex kinetic energy `Σ_e (A_{e,v}/A_v) vn_e²` of every layer."
@kernel function vertex_kinetic_kernel!(Kv, @Const(vn), @Const(vertex_edges), @Const(weights))
    v, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(vertex_edges, 1)
        s += weights[j, v] * vn[vertex_edges[j, v], k]^2
    end
    @inbounds Kv[v, k] = s
end

"Add the vertex part `Σ_v (1 - α) K_{i,v}/A_i K_v` to the cell kinetic energy."
@kernel function cell_from_vertex_kernel!(K, @Const(Kv), @Const(cell_vertices), @Const(kites))
    i, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(cell_vertices, 1)
        s += kites[j, i] * Kv[cell_vertices[j, i], k]
    end
    @inbounds K[i, k] += s
end

"Add the vertical kinetic energy `(dzh_k w_k² + dzh_{k+1} w_{k+1}²) / 4dz_k` of the layer to `K`."
@kernel function w_kinetic_kernel!(K, @Const(w), @Const(dz), @Const(dzh))
    i, k = @index(Global, NTuple)
    @inbounds K[i, k] += (dzh[k] * w[i, k]^2 + dzh[k + 1] * w[i, k + 1]^2) / (4 * dz[k])
end

"Horizontal vorticity `η = ∂vn/∂z - ∂w/∂n` at the edges of the interior interfaces (zero at ground and top)."
@kernel function eta_kernel!(η, @Const(vn), @Const(w), @Const(edge_cells), @Const(inverse_dual), @Const(dzh))
    e, j = @index(Global, NTuple)
    nlev = size(vn, 2)
    @inbounds if 2 <= j <= nlev
        η[e, j] = (vn[e, j] - vn[e, j - 1]) / dzh[j] - (w[edge_cells[2, e], j] - w[edge_cells[1, e], j]) * inverse_dual[e]
    else
        η[e, j] = 0.0
    end
end

"Density `ρ_e` at the edges of the stage state and the mass flux `ρ_e vn`, once per stage for the gathers."
@kernel function edge_density_kernel!(ρe, Fs, @Const(vn), @Const(ρ), @Const(edge_cells), @Const(edge_weight))
    e, k = @index(Global, NTuple)
    @inbounds begin
        r = edge_weight[1, e] * ρ[edge_cells[1, e], k] + edge_weight[2, e] * ρ[edge_cells[2, e], k]
        ρe[e, k] = r
        Fs[e, k] = r * vn[e, k]
    end
end

"""
Explicit tendency of `w` in vector-invariant form, `η ṽn - ∂K/∂z` at the interior
interfaces, `ṽn = (ρ_{e,j-1} vn_{j-1} + ρ_{e,j} vn_j) / 2ρ_e` (`ρ = 1` without
`massweighted`) with the cell inner product.
"""
@kernel function w_tendency_vi_kernel!(tw, @Const(η), @Const(vn), @Const(ρe), @Const(K), @Const(cell_edges),
                                       @Const(inner), @Const(dzh), @Const(wlow), massweighted)
    i, j = @index(Global, NTuple)
    nlev = size(vn, 2)
    @inbounds if 2 <= j <= nlev
        s = 0.0
        for n in axes(cell_edges, 1)
            e = cell_edges[n, i]
            ρl, ρu = massweighted ? (ρe[e, j - 1], ρe[e, j]) : (1.0, 1.0)
            s += inner[n, i] * η[e, j] * (ρl * vn[e, j - 1] + ρu * vn[e, j]) / (2 * (wlow[j] * ρl + (1 - wlow[j]) * ρu))
        end
        tw[i, j] = s - (K[i, j] - K[i, j - 1]) / dzh[j]
    else
        tw[i, j] = 0.0
    end
end

"Vorticity `ζ`, density and potential vorticity `(ζ + f)/ρ` at the vertices of every layer."
@kernel function vertex_pv_kernel!(qv, ζv, ρv, @Const(vn), @Const(ρ), @Const(vertex_edges), @Const(curl),
                                   @Const(vertex_cells), @Const(vertex_weight), @Const(coriolis))
    v, k = @index(Global, NTuple)
    ζ = η = 0.0
    @inbounds for j in axes(vertex_edges, 1)
        ζ += curl[j, v] * vn[vertex_edges[j, v], k]
    end
    @inbounds for j in axes(vertex_cells, 1)
        η += vertex_weight[j, v] * ρ[vertex_cells[j, v], k]
    end
    @inbounds ζv[v, k] = ζ
    @inbounds ρv[v, k] = η
    @inbounds qv[v, k] = (ζ + coriolis[v]) / η
end

"PV at the edges: the rhombus PV `(ζ_e + f_e)/ρ_e` (`iap`) or the mean of the vertex PV."
@kernel function edge_pv_kernel!(qe, @Const(qv), @Const(ζv), @Const(ρv), @Const(edge_vertices), @Const(rhombus),
                                 @Const(edge_coriolis), iap)
    e, k = @index(Global, NTuple)
    @inbounds v1, v2 = edge_vertices[1, e], edge_vertices[2, e]
    @inbounds if iap
        ζ = rhombus[1, e] * ζv[v1, k] + rhombus[2, e] * ζv[v2, k]
        η = rhombus[1, e] * ρv[v1, k] + rhombus[2, e] * ρv[v2, k]
        qe[e, k] = (ζ + edge_coriolis[e]) / η
    else
        qe[e, k] = (qv[v1, k] + qv[v2, k]) / 2
    end
end

"`w̃_e = M_e/ρ_e` at interface `j`, `M_e = Σ_i (d_{e,i}/d_e) ρ_i w_i` (`ρ = 1` without `massweighted`)."
@inline function edge_w(w, ρ, e, l, r, j, edge_weight, wlow, massweighted)
    @inbounds begin
        ρhl = massweighted ? wlow[j] * ρ[l, j - 1] + (1 - wlow[j]) * ρ[l, j] : 1.0
        ρhr = massweighted ? wlow[j] * ρ[r, j - 1] + (1 - wlow[j]) * ρ[r, j] : 1.0
        M = edge_weight[1, e] * ρhl * w[l, j] + edge_weight[2, e] * ρhr * w[r, j]
        return M / (edge_weight[1, e] * ρhl + edge_weight[2, e] * ρhr)
    end
end

"Explicit tendency of `vn` (PV flux, kinetic energy and pressure gradient, vertical advection)."
@kernel function vn_tendency_kernel!(tvn, @Const(vn), @Const(Fs), @Const(w), @Const(ρ), @Const(θ), @Const(πe), @Const(K),
                                     @Const(qe), @Const(η), @Const(edge_cells), @Const(edge_weight),
                                     @Const(inverse_dual), @Const(trisk_edges), @Const(trisk), @Const(pv_edges),
                                     @Const(dz), @Const(dzh), @Const(wlow), nlev, vertical)
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        s = 0.0
        for j in axes(trisk_edges, 1)
            e′ = trisk_edges[j, e]
            p = pv_edges[j, e]
            s += trisk[j, e] * Fs[e′, k] * (p == 0 ? (qe[e, k] + qe[e′, k]) / 2 : qe[p, k])
        end
        θe = edge_weight[1, e] * θ[l, k] + edge_weight[2, e] * θ[r, k]
        t = s - inverse_dual[e] * (K[r, k] - K[l, k] + Cpd * θe * (πe[r, k] - πe[l, k]))
        if vertical > 0
            # -η w̃ at the two interfaces of the layer, w̃ = M_e/ρ_e (ρ = 1 for the plain vector-invariant form),
            # adjoint to ṽn in the w equation
            mw = vertical == 2
            Wl = k > 1 ? edge_w(w, ρ, e, l, r, k, edge_weight, wlow, mw) : 0.0
            Wu = k < nlev ? edge_w(w, ρ, e, l, r, k + 1, edge_weight, wlow, mw) : 0.0
            tvn[e, k] = t - (dzh[k] * η[e, k] * Wl + dzh[k + 1] * η[e, k + 1] * Wu) / (2 * dz[k])
        else
            # vertical advection -w ∂vn/∂z with vn at the interfaces by the mean (one-sided at the boundaries)
            we = (w[l, k] + w[r, k] + w[l, k + 1] + w[r, k + 1]) / 4
            vlow = k > 1 ? (vn[e, k - 1] + vn[e, k]) / 2 : vn[e, k]
            vup = k < nlev ? (vn[e, k] + vn[e, k + 1]) / 2 : vn[e, k]
            tvn[e, k] = t - we * (vup - vlow) / dz[k]
        end
    end
end

"`vn_new = vn_base + τ (tvn + hd)`."
@kernel function vn_update_kernel!(vn, @Const(vn0), @Const(tvn), @Const(hd), τ)
    e, k = @index(Global, NTuple)
    @inbounds vn[e, k] = vn0[e, k] + τ * (tvn[e, k] + hd[e, k])
end

"Horizontal divergence `δ` at the cells and relative vorticity `ζ` at the vertices of every layer."
@kernel function divergence_only_kernel!(δ, @Const(vn), @Const(cell_edges), @Const(divergence))
    i, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(cell_edges, 1)
        s += divergence[j, i] * vn[cell_edges[j, i], k]
    end
    @inbounds δ[i, k] = s
end

@kernel function vorticity_kernel!(ζ, @Const(vn), @Const(vertex_edges), @Const(curl))
    v, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(vertex_edges, 1)
        s += curl[j, v] * vn[vertex_edges[j, v], k]
    end
    @inbounds ζ[v, k] = s
end

"Vector Laplacian `(δ_r - δ_l)/d_e - (ζ_{v2} - ζ_{v1})/l_e`, times `scale`."
@kernel function vector_laplacian_kernel!(L, @Const(δ), @Const(ζ), @Const(edge_cells), @Const(edge_vertices),
                                          @Const(inverse_dual), @Const(inverse_primal), scale)
    e, k = @index(Global, NTuple)
    @inbounds L[e, k] = scale * ((δ[edge_cells[2, e], k] - δ[edge_cells[1, e], k]) * inverse_dual[e] -
                                 (ζ[edge_vertices[2, e], k] - ζ[edge_vertices[1, e], k]) * inverse_primal[e])
end

"Divergence and velocity vector (Perot) at the cells."
@kernel function smagorinsky_cell_kernel!(δ, ux, uy, uz, @Const(vn), @Const(cell_edges), @Const(divergence),
                                          @Const(px), @Const(py), @Const(pz))
    i, k = @index(Global, NTuple)
    s = x = y = z = 0.0
    @inbounds for j in axes(cell_edges, 1)
        v = vn[cell_edges[j, i], k]
        s += divergence[j, i] * v
        x += px[j, i] * v
        y += py[j, i] * v
        z += pz[j, i] * v
    end
    @inbounds δ[i, k], ux[i, k], uy[i, k], uz[i, k] = s, x, y, z
end

"Vorticity, density (kite average) and velocity vector (barycentric interpolation of the cells) at the vertices."
@kernel function smagorinsky_vertex_kernel!(ζ, ρv, vx, vy, vz, @Const(vn), @Const(ρ), @Const(ux), @Const(uy),
                                            @Const(uz), @Const(vertex_edges), @Const(curl), @Const(vertex_cells),
                                            @Const(vertex_weight), @Const(barycentric))
    v, k = @index(Global, NTuple)
    s = r = x = y = z = 0.0
    @inbounds for j in axes(vertex_edges, 1)
        s += curl[j, v] * vn[vertex_edges[j, v], k]
    end
    @inbounds for j in axes(vertex_cells, 1)
        c, a, b = vertex_cells[j, v], vertex_weight[j, v], barycentric[j, v]
        r += a * ρ[c, k]
        x += b * ux[c, k]
        y += b * uy[c, k]
        z += b * uz[c, k]
    end
    @inbounds ζ[v, k], ρv[v, k], vx[v, k], vy[v, k], vz[v, k] = s, r, x, y, z
end

"Smagorinsky coefficient `K = c A_e |D|` at the edges from the velocity gradient across and along the edge."
@kernel function smagorinsky_coefficient_kernel!(Ke, @Const(ux), @Const(uy), @Const(uz), @Const(vx), @Const(vy),
                                                 @Const(vz), @Const(edge_cells), @Const(edge_vertices), @Const(frame),
                                                 @Const(inverse_dual), @Const(inverse_primal), @Const(coefficient))
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        v1, v2 = edge_vertices[1, e], edge_vertices[2, e]
        an = SVector(ux[r, k] - ux[l, k], uy[r, k] - uy[l, k], uz[r, k] - uz[l, k]) * inverse_dual[e]
        at = SVector(vx[v2, k] - vx[v1, k], vy[v2, k] - vy[v1, k], vz[v2, k] - vz[v1, k]) * inverse_primal[e]
        n = SVector(frame[1, e], frame[2, e], frame[3, e])
        t = SVector(frame[4, e], frame[5, e], frame[6, e])
        tension = dot(n, an) - dot(t, at)
        shear = dot(n, at) + dot(t, an)
        Ke[e, k] = coefficient[k] * sqrt(tension^2 + shear^2)
    end
end

"`ρK` at the cells (area mean of the edges) and vertices (mean of the edges)."
@kernel function smagorinsky_cell_coefficient_kernel!(ρKc, @Const(Ke), @Const(ρ), @Const(cell_edges),
                                                      @Const(cell_edge_area))
    i, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(cell_edges, 1)
        s += cell_edge_area[j, i] * Ke[cell_edges[j, i], k]
    end
    @inbounds ρKc[i, k] = ρ[i, k] * s
end

@kernel function smagorinsky_vertex_coefficient_kernel!(ρKv, @Const(Ke), @Const(ρv), @Const(vertex_edges))
    v, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(vertex_edges, 1)
        s += Ke[vertex_edges[j, v], k]
    end
    @inbounds ρKv[v, k] = ρv[v, k] * s / size(vertex_edges, 1)
end

"Add `(1/ρ_e)[∇(ρKδ) - k × ∇(ρKζ)]⋅n` to `hd`."
@kernel function smagorinsky_tendency_kernel!(hd, @Const(ρKc), @Const(ρKv), @Const(δ), @Const(ζ), @Const(ρ),
                                              @Const(edge_cells), @Const(edge_vertices), @Const(edge_weight),
                                              @Const(inverse_dual), @Const(inverse_primal))
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        v1, v2 = edge_vertices[1, e], edge_vertices[2, e]
        ρe = edge_weight[1, e] * ρ[l, k] + edge_weight[2, e] * ρ[r, k]
        hd[e, k] += ((ρKc[r, k] * δ[r, k] - ρKc[l, k] * δ[l, k]) * inverse_dual[e] -
                     (ρKv[v2, k] * ζ[v2, k] - ρKv[v1, k] * ζ[v1, k]) * inverse_primal[e]) / ρe
    end
end

"Dissipative heating `∂ρθ/∂t = (ρK_i δ_i² + Σ_v (K_{i,v}/A_i) ρK_v ζ_v²) / (c_p π_i)`."
@kernel function smagorinsky_heating_kernel!(hρθ, @Const(ρKc), @Const(ρKv), @Const(δ), @Const(ζ), @Const(ρθ),
                                             @Const(cell_vertices), @Const(cell_kites))
    i, k = @index(Global, NTuple)
    Q = 0.0
    @inbounds for j in axes(cell_vertices, 1)
        v = cell_vertices[j, i]
        Q += cell_kites[j, i] * ρKv[v, k] * ζ[v, k]^2
    end
    @inbounds Q += ρKc[i, k] * δ[i, k]^2
    @inbounds hρθ[i, k] = Q / (Cpd * (Rd * ρθ[i, k] / P0)^(Rd / Cvd))
end

"Horizontal mass and `ρθ` fluxes with the new `vn`."
@kernel function flux_kernel!(F, Fθ, @Const(vn), @Const(ρ), @Const(θ), @Const(edge_cells), @Const(edge_weight))
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        f = (edge_weight[1, e] * ρ[l, k] + edge_weight[2, e] * ρ[r, k]) * vn[e, k]
        F[e, k] = f
        Fθ[e, k] = f * (edge_weight[1, e] * θ[l, k] + edge_weight[2, e] * θ[r, k])
    end
end

"Explicit parts of `ρ` and `ρθ`: base state minus `τ` times the horizontal flux divergence (plus the heating)."
@kernel function divergence_kernel!(ρexpl, ρθexpl, @Const(ρ0), @Const(ρθ0), @Const(F), @Const(Fθ), @Const(hρθ),
                                    @Const(cell_edges), @Const(divergence), τ)
    i, k = @index(Global, NTuple)
    s = sθ = 0.0
    @inbounds for j in axes(cell_edges, 1)
        e = cell_edges[j, i]
        s += divergence[j, i] * F[e, k]
        sθ += divergence[j, i] * Fθ[e, k]
    end
    @inbounds ρexpl[i, k] = ρ0[i, k] - τ * s
    @inbounds ρθexpl[i, k] = ρθ0[i, k] - τ * (sθ - hρθ[i, k])
end

"Explicit advection of `w` at the interior interfaces: `-u ⋅ ∇w - w ∂w/∂z`."
@kernel function w_tendency_kernel!(tw, @Const(w), @Const(vn), @Const(cell_edges), @Const(kinetic),
                                    @Const(edge_cells), @Const(inverse_dual), @Const(dz))
    i, j = @index(Global, NTuple)
    nlev = size(vn, 2)
    @inbounds if 2 <= j <= nlev
        s = 0.0
        for n in axes(cell_edges, 1)
            e = cell_edges[n, i]
            v = (vn[e, j - 1] + vn[e, j]) / 2
            s += 2 * kinetic[n, i] * v * (w[edge_cells[2, e], j] - w[edge_cells[1, e], j]) * inverse_dual[e]
        end
        tw[i, j] = -s - w[i, j] * (w[i, j + 1] - w[i, j - 1]) / (dz[j - 1] + dz[j])
    else
        tw[i, j] = 0.0
    end
end

"""
Weights of the cells `j-2, j-1, j, j+1` in `θ` at interface `j` for the mass
flux `M`: linear interpolation plus `1/6` of the curvature at the upwind cell,
third-order upwind (ICON-IAP `impl_vert_adv_scalar_5diag`), linear next to the
boundaries where the upwind stencil would leave the column.
"""
@inline function theta_stencil(j, M, dz, dzh, wlow, nlev)
    σ = 1 / 6
    a, b, c, d = 0.0, wlow[j], 1 - wlow[j], 0.0
    if M >= 0
        if j >= 3
            a -= σ * dzh[j] / dz[j - 1]
            b += σ * (dzh[j] + dzh[j - 1]) / dz[j - 1]
            c -= σ * dzh[j - 1] / dz[j - 1]
        end
    elseif j <= nlev - 1
        b -= σ * dzh[j + 1] / dz[j]
        c += σ * (dzh[j] + dzh[j + 1]) / dz[j]
        d -= σ * dzh[j] / dz[j]
    end
    return (a, b, c, d)
end

"""
`θ` at the interfaces for the vertical solver. `upwind`: implicit vertical
advection of `θ` of the base state with the mass flux of the stage state over
`τ/2` (Crank–Nicolson, one pentadiagonal system per column) and the
third-order upwind interface values of the result, as ICON-IAP
(`ltheta_up_vert`); otherwise `θ` of the stage state linearly interpolated.
"""
@kernel function theta_interface_kernel!(θi, A, θm, @Const(θs), @Const(ρ0), @Const(ρθ0), @Const(ρs), @Const(ws),
                                         @Const(dz), @Const(dzh), @Const(wlow), τ, nlev, upwind)
    i = @index(Global)
    @inbounds begin
        θi[i, 1] = θi[i, nlev + 1] = 0.0
        if !upwind
            for j in 2:nlev
                θi[i, j] = wlow[j] * θs[i, j - 1] + (1 - wlow[j]) * θs[i, j]
            end
        else
            M(j) = (wlow[j] * ρs[i, j - 1] + (1 - wlow[j]) * ρs[i, j]) * ws[i, j]
            # band A[i, k, 3 + c - k] of row k, rhs θ of the base state
            for k in 1:nlev
                for d in 1:5
                    A[i, k, d] = 0.0
                end
                A[i, k, 3] = 1.0
                θm[i, k] = ρθ0[i, k] / ρ0[i, k]
            end
            # θm + τ/(2ρ dz) [M_{k+1} (θ_{k+1} - θm_k) - M_k (θ_k - θm_k)] = θ0
            for k in 1:nlev
                f = τ / (2 * ρs[i, k] * dz[k])
                for (j, s) in ((k + 1, 1), (k, -1))
                    2 <= j <= nlev || continue
                    Mj = M(j)
                    c = theta_stencil(j, Mj, dz, dzh, wlow, nlev)
                    for n in 1:4
                        cell = j - 3 + n
                        1 <= cell <= nlev && (A[i, k, cell - k + 3] += s * f * Mj * c[n])
                    end
                    A[i, k, 3] -= s * f * Mj
                end
            end
            # banded elimination without pivoting, back substitution
            for k in 1:nlev - 1, r in k + 1:min(k + 2, nlev)
                l = A[i, r, k - r + 3] / A[i, k, 3]
                for c in k + 1:min(k + 2, nlev)
                    A[i, r, c - r + 3] -= l * A[i, k, c - k + 3]
                end
                θm[i, r] -= l * θm[i, k]
            end
            for k in nlev:-1:1
                s = θm[i, k]
                for c in k + 1:min(k + 2, nlev)
                    s -= A[i, k, c - k + 3] * θm[i, c]
                end
                θm[i, k] = s / A[i, k, 3]
            end
            for j in 2:nlev
                c = theta_stencil(j, M(j), dz, dzh, wlow, nlev)
                s = 0.0
                for n in 1:4
                    cell = j - 3 + n
                    1 <= cell <= nlev && (s += c[n] * θm[i, cell])
                end
                θi[i, j] = s
            end
        end
    end
end

"""
Vertically implicit step of one column: `w`, `ρ`, `ρθ` at `t + τ` from the
explicit parts, with `θ` at the interfaces from `theta_interface_kernel!` in
the `ρθ` flux (and with `iap_theta` in the pressure gradient, else `θ` of the
stage state linearly interpolated), `ρ` there from the stage state and the
linearization of `π` around the base state.
"""
@kernel function vertical_solve_kernel!(w, ρ, ρθ, cp_, dp_, @Const(w0), @Const(ρ0), @Const(ρθ0), @Const(θs),
                                        @Const(θi), @Const(ρs), @Const(tw), @Const(ρexpl), @Const(ρθexpl), @Const(dz),
                                        @Const(dzh), @Const(wlow), @Const(rayleigh), τ, β, nlev, iap_theta)
    i = @index(Global)
    @inbounds begin
        κ = Rd / Cvd
        exner(x) = (Rd * x / P0)^κ
        θh(j) = iap_theta ? θi[i, j] : wlow[j] * θs[i, j - 1] + (1 - wlow[j]) * θs[i, j]
        ρh(j) = wlow[j] * ρs[i, j - 1] + (1 - wlow[j]) * ρs[i, j]
        γ(k) = κ * exner(ρθ0[i, k]) / ρθ0[i, k] * τ / dz[k]
        πexpl(k) = exner(ρθ0[i, k]) * (1 + κ * (ρθexpl[i, k] - ρθ0[i, k]) / ρθ0[i, k])
        # interface j couples to j - 1 and j + 1 through M = b w + c (zero at the boundaries)
        b(j) = 2 <= j <= nlev ? ρh(j) * β : 0.0
        c(j) = 2 <= j <= nlev ? ρh(j) * (1 - β) * w0[i, j] : 0.0
        θb(j) = 2 <= j <= nlev ? θi[i, j] : 0.0
        # Thomas algorithm, forward sweep over j = 2:nlev
        cp_[i, 1] = 0.0
        dp_[i, 1] = 0.0
        for j in 2:nlev
            aj = τ * Cpd * θh(j) * β / dzh[j]
            γu, γl = γ(j), γ(j - 1)
            lower = -aj * γl * θb(j - 1) * b(j - 1)
            diag = 1 + τ * rayleigh[j] + aj * (γu + γl) * θb(j) * b(j)
            upper = -aj * γu * θb(j + 1) * b(j + 1)
            rhs = w0[i, j] + τ * (tw[i, j] - GRAV) -
                  τ * Cpd * θh(j) * (1 - β) * (exner(ρθ0[i, j]) - exner(ρθ0[i, j - 1])) / dzh[j] -
                  aj * (πexpl(j) - πexpl(j - 1)) -
                  aj * (-γu * θb(j + 1) * c(j + 1) + (γu + γl) * θb(j) * c(j) - γl * θb(j - 1) * c(j - 1))
            denom = diag - lower * cp_[i, j - 1]
            cp_[i, j] = upper / denom
            dp_[i, j] = (rhs - lower * dp_[i, j - 1]) / denom
        end
        w[i, nlev + 1] = 0.0
        for j in nlev:-1:2
            w[i, j] = dp_[i, j] - cp_[i, j] * w[i, j + 1]
        end
        w[i, 1] = 0.0
        # mass and ρθ fluxes M_j, θ_j M_j and the new ρ, ρθ
        Mlow = θMlow = 0.0
        for k in 1:nlev
            if k < nlev
                M = b(k + 1) * w[i, k + 1] + c(k + 1)
                θM = θb(k + 1) * M
            else
                M = θM = 0.0
            end
            ρ[i, k] = ρexpl[i, k] - τ / dz[k] * (M - Mlow)
            ρθ[i, k] = ρθexpl[i, k] - τ / dz[k] * (θM - θMlow)
            Mlow, θMlow = M, θM
        end
    end
end

# --- time stepping ---------------------------------------------------------------

"""
    stage!(m, y, y0, ys, τ)

One HEVI stage: `y = y0 + τ T(ys)` with the horizontal terms explicit in the
stage state `ys` and the vertical acoustic and buoyancy terms implicit.
"""
function stage!(m::TRiSKNonhydrostatic, y, y0, ys, τ)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne, nv = num_cells(m.grid), num_edges(m.grid), num_vertices(m.grid)
    launch!(m, diagnose_kernel!, (nc, nlev), w.θ, w.π, w.K, ys.vn, ys.ρ, ys.ρθ, m.cell_edges, m.kinetic)
    vk = m.vertex_kinetic
    if vk !== nothing
        launch!(m, vertex_kinetic_kernel!, (nv, nlev), w.Kv, ys.vn, m.vertex_edges, vk.weights)
        launch!(m, cell_from_vertex_kernel!, (nc, nlev), w.K, w.Kv, vk.cell_vertices, vk.kites)
    end
    launch!(m, vertex_pv_kernel!, (nv, nlev), w.qv, w.ζv, w.ρv, ys.vn, ys.ρ, m.vertex_edges, m.curl, m.vertex_cells,
            m.vertex_weight, m.coriolis)
    launch!(m, edge_pv_kernel!, (ne, nlev), w.qe, w.qv, w.ζv, w.ρv, m.edge_vertices, m.rhombus, m.edge_coriolis,
            m.iap_pv)
    launch!(m, edge_density_kernel!, (ne, nlev), w.ρe, w.Fs, ys.vn, ys.ρ, m.edge_cells, m.edge_weight)
    if m.vertical > 0
        launch!(m, w_kinetic_kernel!, (nc, nlev), w.K, ys.w, m.dz, m.dzh)
        launch!(m, eta_kernel!, (ne, nlev + 1), w.η, ys.vn, ys.w, m.edge_cells, m.inverse_dual, m.dzh)
        launch!(m, w_tendency_vi_kernel!, (nc, nlev + 1), w.tw, w.η, ys.vn, w.ρe, w.K, m.cell_edges, m.inner, m.dzh,
                m.wlow, m.vertical == 2)
    else
        launch!(m, w_tendency_kernel!, (nc, nlev + 1), w.tw, ys.w, ys.vn, m.cell_edges, m.kinetic, m.edge_cells,
                m.inverse_dual, m.dz)
    end
    launch!(m, vn_tendency_kernel!, (ne, nlev), w.tvn, ys.vn, w.Fs, ys.w, ys.ρ, w.θ, w.π, w.K, w.qe, w.η, m.edge_cells,
            m.edge_weight, m.inverse_dual, m.trisk_edges, m.trisk, m.pv_edges, m.dz, m.dzh, m.wlow, nlev, m.vertical)
    launch!(m, vn_update_kernel!, (ne, nlev), y.vn, y0.vn, w.tvn, w.hd, τ)
    launch!(m, flux_kernel!, (ne, nlev), w.F, w.Fθ, y.vn, ys.ρ, w.θ, m.edge_cells, m.edge_weight)
    launch!(m, divergence_kernel!, (nc, nlev), w.ρexpl, w.ρθexpl, y0.ρ, y0.ρθ, w.F, w.Fθ, w.hρθ, m.cell_edges,
            m.divergence, τ)
    launch!(m, theta_interface_kernel!, nc, w.θi, w.band, w.θm, w.θ, y0.ρ, y0.ρθ, ys.ρ, ys.w, m.dz, m.dzh, m.wlow,
            τ, nlev, m.vertical_theta > 0)
    launch!(m, vertical_solve_kernel!, nc, y.w, y.ρ, y.ρθ, w.cprime, w.dprime, y0.w, y0.ρ, y0.ρθ, w.θ, w.θi, ys.ρ,
            w.tw, w.ρexpl, w.ρθexpl, m.dz, m.dzh, m.wlow, m.rayleigh, τ, m.beta, nlev, m.vertical_theta == 2)
    return y
end

"""
    step!(m, y, dt, ys=zero_state(m))

One Wicker–Skamarock RK3 step of length `dt` (stages `dt/3`, `dt/2`, `dt`),
in place; `ys` is work space for the stage state, the base state is kept in
the work arrays (no allocation on the device).
"""
function step!(m::TRiSKNonhydrostatic, y, dt, ys=zero_state(m))
    y0 = m.work.y0
    foreach(copyto!, y0, y)
    hyperdiffusion!(m.work.hd, m, y0.vn)
    smagorinsky!(m.work.hd, m.work.hρθ, m, y0)
    stage!(m, ys, y0, y0, dt / 3)
    stage!(m, y, y0, ys, dt / 2)
    foreach(copyto!, ys, y)
    stage!(m, y, y0, ys, dt)
    return y
end

"Vector Laplacian of `vn` times `scale` into `L`."
function vector_laplacian!(L, m::TRiSKNonhydrostatic, vn, scale=1.0)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne, nv = num_cells(m.grid), num_edges(m.grid), num_vertices(m.grid)
    launch!(m, divergence_only_kernel!, (nc, nlev), w.δ, vn, m.cell_edges, m.divergence)
    launch!(m, vorticity_kernel!, (nv, nlev), w.ζ, vn, m.vertex_edges, m.curl)
    launch!(m, vector_laplacian_kernel!, (ne, nlev), L, w.δ, w.ζ, m.edge_cells, m.edge_vertices, m.inverse_dual,
            m.inverse_primal, scale)
    return L
end

"Hyperdiffusion `-ν₄ L(L(vn))` into `hd` (zero without hyperdiffusion)."
function hyperdiffusion!(hd, m::TRiSKNonhydrostatic, vn)
    if m.hyperdiffusion == 0
        fill!(hd, 0.0)
    else
        vector_laplacian!(m.work.Lvn, m, vn)
        vector_laplacian!(hd, m, m.work.Lvn, -m.hyperdiffusion)
    end
    return hd
end

"""
    smagorinsky!(hd, hρθ, m, y)

Add the Smagorinsky diffusion of `vn` in the state `y` to `hd` and put its
heating into `hρθ` (zero without diffusion).
"""
function smagorinsky!(hd, hρθ, m::TRiSKNonhydrostatic, y)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne, nv = num_cells(m.grid), num_edges(m.grid), num_vertices(m.grid)
    fill!(hρθ, 0.0)
    all(iszero, m.smagorinsky) && return hd
    launch!(m, smagorinsky_cell_kernel!, (nc, nlev), w.δ, w.uc..., y.vn, m.cell_edges, m.divergence, m.perot...)
    launch!(m, smagorinsky_vertex_kernel!, (nv, nlev), w.ζ, w.ρv, w.uv..., y.vn, y.ρ, w.uc..., m.vertex_edges, m.curl,
            m.vertex_cells, m.vertex_weight, m.barycentric)
    launch!(m, smagorinsky_coefficient_kernel!, (ne, nlev), w.Ke, w.uc..., w.uv..., m.edge_cells, m.edge_vertices,
            m.edge_frame, m.inverse_dual, m.inverse_primal, m.smagorinsky)
    launch!(m, smagorinsky_cell_coefficient_kernel!, (nc, nlev), w.ρKc, w.Ke, y.ρ, m.cell_edges, m.cell_edge_area)
    launch!(m, smagorinsky_vertex_coefficient_kernel!, (nv, nlev), w.ρKv, w.Ke, w.ρv, m.vertex_edges)
    launch!(m, smagorinsky_tendency_kernel!, (ne, nlev), hd, w.ρKc, w.ρKv, w.δ, w.ζ, y.ρ, m.edge_cells,
            m.edge_vertices, m.edge_weight, m.inverse_dual, m.inverse_primal)
    launch!(m, smagorinsky_heating_kernel!, (nc, nlev), hρθ, w.ρKc, w.ρKv, w.δ, w.ζ, y.ρθ, m.cell_vertices,
            m.cell_kites)
    return hd
end

# --- diagnostics -------------------------------------------------------------------

"Pressure `p₀ π^(c_p/R)` of the lowest layer at every cell (on the host)."
lowest_pressure(m::TRiSKNonhydrostatic, y) = P0 .* (Rd .* Array(y.ρθ)[:, 1] ./ P0) .^ (Cpd / Cvd)

"Total mass `Σ a² A_i dz_k ρ_ik`."
total_mass(m::TRiSKNonhydrostatic, y) = sum(Array(m.cell_areas) .* (Array(y.ρ) * Array(m.dz)))
