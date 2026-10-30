"""
    TRiSKNonhydrostatic(grid::TRiSKGrid, vgrid::VerticalGrid; radius, omega, beta=0.65,
                        damping_height=Inf, damping_rate=0, kinetic_energy=:cell, hyperdiffusion=0,
                        backend=CPU())

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
flux `Σ w_{e,e'} l_{e'} ρ_{e'} vn_{e'} (q_e + q_{e'}) / 2d_e` with
`q = (ζ + f)/ρ` at the vertices, the cell kinetic energy, divergence and
gradient. Vertically the differences are centred on the staggered grid,
`θ` and `ρ` linearly interpolated to the interfaces.

    ∂vn/∂t = PV flux - ∂_n K - c_p θ_e ∂_n π - w_e ∂vn/∂z
    ∂w/∂t  = - u ⋅ ∇w - w ∂w/∂z - c_p θ ∂π/∂z - g - r(z) w
    ∂ρ/∂t  = - ∇ ⋅ (ρ_e vn) - ∂(ρ w)/∂z
    ∂ρθ/∂t = - ∇ ⋅ (ρ_e θ_e vn) - ∂(ρ θ w)/∂z

Time stepping (`step!`): the three stages of Wicker and Skamarock (2002), each
horizontally explicit (forward–backward: the mass fluxes use the new `vn`) and
vertically implicit for `w`, `ρ`, `ρθ` with the off-centring `beta` and the
linearized equation of state, one tridiagonal system for `w` per column (as
in ICON, Zängl et al. 2015). `r(z)` is a Rayleigh damping of `w` above
`damping_height` (rate `damping_rate sin²`).

`vertical_advection = :vector_invariant` (default) writes the momentum
advection in the 3D vector-invariant form as in CGSEM (CGDycore.jl):
`(ω × u)` with the horizontal vorticity `η = ∂vn/∂z - ∂w/∂n` at the edges
of the interfaces, `-η w̄` in the `vn` equation, `+η v̄n` in the `w` equation,
and the gradient of the full kinetic energy `K = K_h + w²/2`. The averages
are adjoint to each other (`v̄n` the mean of the two layers, `w̄` of the two
cells, the cell weights `l_e d_e / 4A_i` of the edge areas `l_e d_e / 2`), so
the exchange between horizontal and vertical kinetic energy cancels exactly.
`:advective` uses `-w ∂vn/∂z` and `-u ⋅ ∇w - w ∂w/∂z` instead.

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
    edge_area_weight::M  # l_e d_e / (4 A_i): cell average of edge quantities, adjoint to the edge mean
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
    cell_areas::V        # a² A_i
    dz::V
    dzh::V
    wlow::V
    rayleigh::V          # damping rate at the interfaces
    work::NamedTuple
    radius::Float64
    beta::Float64
    hyperdiffusion::Float64
    vector_invariant::Bool
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
                             hyperdiffusion::Real=0.0, vertical_advection::Symbol=:vector_invariant,
                             backend=KernelAbstractions.CPU())
    vertical_advection in (:vector_invariant, :advective) ||
        throw(ArgumentError("vertical_advection must be :vector_invariant or :advective"))
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
            η=zeros2(ne, nlev + 1),
            F=zeros2(ne, nlev), Fθ=zeros2(ne, nlev), tvn=zeros2(ne, nlev), tw=zeros2(nc, nlev + 1),
            ρexpl=zeros2(nc, nlev), ρθexpl=zeros2(nc, nlev), cprime=zeros2(nc, nlev + 1),
            dprime=zeros2(nc, nlev + 1), δ=zeros2(nc, nlev), ζ=zeros2(nv, nlev), Lvn=zeros2(ne, nlev),
            hd=zeros2(ne, nlev))
    edge_weight = dev(grid.cell_distances ./ grid.dual_lengths')
    edge_area_weight = zeros(size(grid.cell_edges))
    for i in 1:nc, j in axes(grid.cell_edges, 1)
        e = grid.cell_edges[j, i]
        grid.cell_signs[j, i] != 0 &&
            (edge_area_weight[j, i] = grid.primal_lengths[e] * grid.dual_lengths[e] / (4 * grid.cell_areas[i]))
    end
    return TRiSKNonhydrostatic(grid, vgrid, sw.cell_edges, dev(Array(sw.divergence) ./ a), sw.kinetic,
                               dev(edge_area_weight),
                               sw.vertex_kinetic, sw.edge_cells, sw.edge_vertices, edge_weight, dev(Array(sw.inverse_dual) ./ a),
                               dev(1 ./ (a .* grid.primal_lengths)), sw.vertex_edges,
                               dev(Array(sw.curl) ./ a), sw.vertex_cells, sw.vertex_height, sw.coriolis,
                               sw.trisk_edges, sw.trisk, dev(a^2 .* grid.cell_areas), dev(vgrid.dz), dev(vgrid.dzh),
                               dev(vgrid.wlow), dev(rayleigh), work, a, Float64(beta), Float64(hyperdiffusion),
                               vertical_advection == :vector_invariant, backend)
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

"Add the vertical kinetic energy `(w_k² + w_{k+1}²)/4` of the layer to `K`."
@kernel function w_kinetic_kernel!(K, @Const(w))
    i, k = @index(Global, NTuple)
    @inbounds K[i, k] += (w[i, k]^2 + w[i, k + 1]^2) / 4
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

"Explicit tendency of `w` in vector-invariant form: `η v̄n - ∂K/∂z` at the interior interfaces."
@kernel function w_tendency_vi_kernel!(tw, @Const(η), @Const(vn), @Const(K), @Const(cell_edges),
                                       @Const(edge_area_weight), @Const(dzh))
    i, j = @index(Global, NTuple)
    nlev = size(vn, 2)
    @inbounds if 2 <= j <= nlev
        s = 0.0
        for n in axes(cell_edges, 1)
            e = cell_edges[n, i]
            s += edge_area_weight[n, i] * η[e, j] * (vn[e, j - 1] + vn[e, j]) / 2
        end
        tw[i, j] = s - (K[i, j] - K[i, j - 1]) / dzh[j]
    else
        tw[i, j] = 0.0
    end
end

"Potential vorticity `(ζ + f)/ρ` at the vertices of every layer."
@kernel function vertex_pv_kernel!(qv, @Const(vn), @Const(ρ), @Const(vertex_edges), @Const(curl),
                                   @Const(vertex_cells), @Const(vertex_weight), @Const(coriolis))
    v, k = @index(Global, NTuple)
    ζ = η = 0.0
    @inbounds for j in axes(vertex_edges, 1)
        ζ += curl[j, v] * vn[vertex_edges[j, v], k]
    end
    @inbounds for j in axes(vertex_cells, 1)
        η += vertex_weight[j, v] * ρ[vertex_cells[j, v], k]
    end
    @inbounds qv[v, k] = (ζ + coriolis[v]) / η
end

"Explicit tendency of `vn` (PV flux, kinetic energy and pressure gradient, vertical advection)."
@kernel function vn_tendency_kernel!(tvn, @Const(vn), @Const(w), @Const(ρ), @Const(θ), @Const(πe), @Const(K),
                                     @Const(qv), @Const(η), @Const(edge_cells), @Const(edge_vertices),
                                     @Const(edge_weight), @Const(inverse_dual), @Const(trisk_edges), @Const(trisk),
                                     @Const(dz), @Const(dzh), nlev, vector_invariant)
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        qe = (qv[edge_vertices[1, e], k] + qv[edge_vertices[2, e], k]) / 2
        s = 0.0
        for j in axes(trisk_edges, 1)
            e′ = trisk_edges[j, e]
            F′ = (edge_weight[1, e′] * ρ[edge_cells[1, e′], k] + edge_weight[2, e′] * ρ[edge_cells[2, e′], k]) * vn[e′, k]
            q′ = (qv[edge_vertices[1, e′], k] + qv[edge_vertices[2, e′], k]) / 2
            s += trisk[j, e] * F′ * (qe + q′)
        end
        θe = edge_weight[1, e] * θ[l, k] + edge_weight[2, e] * θ[r, k]
        t = s / 2 - inverse_dual[e] * (K[r, k] - K[l, k] + Cpd * θe * (πe[r, k] - πe[l, k]))
        if vector_invariant
            # -η w̄ at the two interfaces of the layer, adjoint to the layer mean of vn in the w equation
            Wl, Wu = (w[l, k] + w[r, k]) / 2, (w[l, k + 1] + w[r, k + 1]) / 2
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

"Explicit parts of `ρ` and `ρθ`: base state minus `τ` times the horizontal flux divergence."
@kernel function divergence_kernel!(ρexpl, ρθexpl, @Const(ρ0), @Const(ρθ0), @Const(F), @Const(Fθ),
                                    @Const(cell_edges), @Const(divergence), τ)
    i, k = @index(Global, NTuple)
    s = sθ = 0.0
    @inbounds for j in axes(cell_edges, 1)
        e = cell_edges[j, i]
        s += divergence[j, i] * F[e, k]
        sθ += divergence[j, i] * Fθ[e, k]
    end
    @inbounds ρexpl[i, k] = ρ0[i, k] - τ * s
    @inbounds ρθexpl[i, k] = ρθ0[i, k] - τ * sθ
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
Vertically implicit step of one column: `w`, `ρ`, `ρθ` at `t + τ` from the
explicit parts, with `θ`, `ρ` at the interfaces from the stage state and the
linearization of `π` around the base state.
"""
@kernel function vertical_solve_kernel!(w, ρ, ρθ, cp_, dp_, @Const(w0), @Const(ρ0), @Const(ρθ0), @Const(θs),
                                        @Const(ρs), @Const(tw), @Const(ρexpl), @Const(ρθexpl), @Const(dz), @Const(dzh),
                                        @Const(wlow), @Const(rayleigh), τ, β, nlev)
    i = @index(Global)
    @inbounds begin
        κ = Rd / Cvd
        exner(x) = (Rd * x / P0)^κ
        θh(j) = wlow[j] * θs[i, j - 1] + (1 - wlow[j]) * θs[i, j]
        ρh(j) = wlow[j] * ρs[i, j - 1] + (1 - wlow[j]) * ρs[i, j]
        γ(k) = κ * exner(ρθ0[i, k]) / ρθ0[i, k] * τ / dz[k]
        πexpl(k) = exner(ρθ0[i, k]) * (1 + κ * (ρθexpl[i, k] - ρθ0[i, k]) / ρθ0[i, k])
        # interface j couples to j - 1 and j + 1 through M = b w + c (zero at the boundaries)
        b(j) = 2 <= j <= nlev ? ρh(j) * β : 0.0
        c(j) = 2 <= j <= nlev ? ρh(j) * (1 - β) * w0[i, j] : 0.0
        θb(j) = 2 <= j <= nlev ? θh(j) : 0.0
        # Thomas algorithm, forward sweep over j = 2:nlev
        cp_[i, 1] = 0.0
        dp_[i, 1] = 0.0
        for j in 2:nlev
            aj = τ * Cpd * θh(j) * β / dzh[j]
            γu, γl = γ(j), γ(j - 1)
            lower = -aj * γl * θb(j - 1) * b(j - 1)
            diag = 1 + τ * rayleigh[j] + aj * (γu + γl) * θh(j) * b(j)
            upper = -aj * γu * θb(j + 1) * b(j + 1)
            rhs = w0[i, j] + τ * (tw[i, j] - GRAV) -
                  τ * Cpd * θh(j) * (1 - β) * (exner(ρθ0[i, j]) - exner(ρθ0[i, j - 1])) / dzh[j] -
                  aj * (πexpl(j) - πexpl(j - 1)) -
                  aj * (-γu * θb(j + 1) * c(j + 1) + (γu + γl) * θh(j) * c(j) - γl * θb(j - 1) * c(j - 1))
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
                θM = θh(k + 1) * M
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
    launch!(m, vertex_pv_kernel!, (nv, nlev), w.qv, ys.vn, ys.ρ, m.vertex_edges, m.curl, m.vertex_cells,
            m.vertex_weight, m.coriolis)
    if m.vector_invariant
        launch!(m, w_kinetic_kernel!, (nc, nlev), w.K, ys.w)
        launch!(m, eta_kernel!, (ne, nlev + 1), w.η, ys.vn, ys.w, m.edge_cells, m.inverse_dual, m.dzh)
        launch!(m, w_tendency_vi_kernel!, (nc, nlev + 1), w.tw, w.η, ys.vn, w.K, m.cell_edges, m.edge_area_weight,
                m.dzh)
    else
        launch!(m, w_tendency_kernel!, (nc, nlev + 1), w.tw, ys.w, ys.vn, m.cell_edges, m.kinetic, m.edge_cells,
                m.inverse_dual, m.dz)
    end
    launch!(m, vn_tendency_kernel!, (ne, nlev), w.tvn, ys.vn, ys.w, ys.ρ, w.θ, w.π, w.K, w.qv, w.η, m.edge_cells,
            m.edge_vertices, m.edge_weight, m.inverse_dual, m.trisk_edges, m.trisk, m.dz, m.dzh, nlev,
            m.vector_invariant)
    launch!(m, vn_update_kernel!, (ne, nlev), y.vn, y0.vn, w.tvn, w.hd, τ)
    launch!(m, flux_kernel!, (ne, nlev), w.F, w.Fθ, y.vn, ys.ρ, w.θ, m.edge_cells, m.edge_weight)
    launch!(m, divergence_kernel!, (nc, nlev), w.ρexpl, w.ρθexpl, y0.ρ, y0.ρθ, w.F, w.Fθ, m.cell_edges,
            m.divergence, τ)
    launch!(m, vertical_solve_kernel!, nc, y.w, y.ρ, y.ρθ, w.cprime, w.dprime, y0.w, y0.ρ, y0.ρθ, w.θ, ys.ρ,
            w.tw, w.ρexpl, w.ρθexpl, m.dz, m.dzh, m.wlow, m.rayleigh, τ, m.beta, nlev)
    return y
end

"""
    step!(m, y, dt, ys=zero_state(m))

One Wicker–Skamarock RK3 step of length `dt` (stages `dt/3`, `dt/2`, `dt`),
in place; `ys` is work space for the stage state.
"""
function step!(m::TRiSKNonhydrostatic, y, dt, ys=zero_state(m))
    y0 = map(copy, y)
    hyperdiffusion!(m.work.hd, m, y0.vn)
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

# --- diagnostics -------------------------------------------------------------------

"Pressure `p₀ π^(c_p/R)` of the lowest layer at every cell (on the host)."
lowest_pressure(m::TRiSKNonhydrostatic, y) = P0 .* (Rd .* Array(y.ρθ)[:, 1] ./ P0) .^ (Cpd / Cvd)

"Total mass `Σ a² A_i dz_k ρ_ik`."
total_mass(m::TRiSKNonhydrostatic, y) = sum(Array(m.cell_areas) .* (Array(y.ρ) * Array(m.dz)))
