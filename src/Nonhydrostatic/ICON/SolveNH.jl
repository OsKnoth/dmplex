# Fast waves of ICON (`solve_nh`): Exner pressure, Miura reconstruction of ρ and θ at the edges, pressure gradient, fluxes, vertical solve.

"θ and the Exner pressure at the cells."
@kernel function icon_thermo_kernel!(θ, πe, @Const(ρ), @Const(ρθ))
    i, k = @index(Global, NTuple)
    @inbounds θ[i, k] = ρθ[i, k] / ρ[i, k]
    @inbounds πe[i, k] = (ICON_RD * ρθ[i, k] / P0)^(ICON_RD / ICON_CVD)
end

"Green–Gauss gradients of `ρ` and `θ` at the cells (edge values by `c_lin_e`), unit sphere."
@kernel function icon_cell_gradient_kernel!(g1, g2, g3, h1, h2, h3, @Const(ρ), @Const(θ), @Const(cell_edges),
                                            @Const(edge_cells), @Const(edge_height), @Const(gx), @Const(gy),
                                            @Const(gz))
    i, k = @index(Global, NTuple)
    a1 = a2 = a3 = b1 = b2 = b3 = 0.0
    @inbounds for n in axes(cell_edges, 1)
        e = cell_edges[n, i]
        l, r = edge_cells[1, e], edge_cells[2, e]
        ρe = edge_height[1, e] * ρ[l, k] + edge_height[2, e] * ρ[r, k]
        θe = edge_height[1, e] * θ[l, k] + edge_height[2, e] * θ[r, k]
        a1 += gx[n, i] * ρe; a2 += gy[n, i] * ρe; a3 += gz[n, i] * ρe
        b1 += gx[n, i] * θe; b2 += gy[n, i] * θe; b3 += gz[n, i] * θe
    end
    @inbounds g1[i, k], g2[i, k], g3[i, k] = a1, a2, a3
    @inbounds h1[i, k], h2[i, k], h3[i, k] = b1, b2, b3
end

"""
`ρ_e`, `θ_e` of Miura's second-order upwind reconstruction at the
backtrajectory point of the edge over `Δt/2`, and the horizontal pressure
gradient `c_p θ_e ∂π/∂n` of the extrapolated Exner pressure `πex`; over orography
(`oro = 1`) `c_p θ_e (∂π'/∂n - ∂z/∂n ∂π'/∂z)` with `πex = π'` and `dπdz = ∂π'/∂z`; at constant
height (`oro = 2`, `height_difference`); or (`oro = 3`) in the lowest `nt` layers from the Taylor
expansion `π' + δ (∂π'/∂z + δ c₂)` of the layer `zlev` of either cell at the distance `zwt` from
its centre (`taylor_gradient`, `c₂ = ½ ∂²π'/∂z²` of `icon_exner_curvature_kernel!`), above as `oro = 1`.
As in ICON, `ρ`, `θ` (and their gradients) are the deviations from the reference state, which is
added at the height of the edge (`ρref_e`, `θref_e`; zero for the reconstruction of the full fields).
"""
@kernel function icon_miura_kernel!(ρe, θe, pg, @Const(vn), @Const(vt), @Const(ρ), @Const(θ), @Const(πex),
                                    @Const(g1), @Const(g2), @Const(g3), @Const(h1), @Const(h2), @Const(h3),
                                    @Const(edge_cells), @Const(edge_vectors), @Const(inverse_dual), half_dt_a,
                                    @Const(dπdz), @Const(ddxn), @Const(edge_height), @Const(zlev), @Const(zwt), oro,
                                    @Const(ρref_e), @Const(θref_e), @Const(c2), nt)
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        u, v = vn[e, k], vt[e, k]
        up, o = u >= 0 ? (l, 0) : (r, 3)
        d1 = edge_vectors[o + 1, e] - half_dt_a * (u * edge_vectors[7, e] + v * edge_vectors[10, e])
        d2 = edge_vectors[o + 2, e] - half_dt_a * (u * edge_vectors[8, e] + v * edge_vectors[11, e])
        d3 = edge_vectors[o + 3, e] - half_dt_a * (u * edge_vectors[9, e] + v * edge_vectors[12, e])
        ρe[e, k] = ρref_e[e, k] + ρ[up, k] + g1[up, k] * d1 + g2[up, k] * d2 + g3[up, k] * d3
        t = θref_e[e, k] + θ[up, k] + h1[up, k] * d1 + h2[up, k] * d2 + h3[up, k] * d3
        θe[e, k] = t
        if oro == 2
            dπ = height_difference(πex, edge_cells, zlev, zwt, e, k) * inverse_dual[e]
        elseif oro == 3 && k <= nt
            taylor(s, c) = (j = zlev[s, e, k]; δ = zwt[s, e, k]; πex[c, j] + δ * (dπdz[c, j] + δ * c2[c, j]))
            dπ = (taylor(2, r) - taylor(1, l)) * inverse_dual[e]
        else
            dπ = (πex[r, k] - πex[l, k]) * inverse_dual[e]
            oro != 0 && (dπ -= ddxn[e, k] * (edge_height[1, e] * dπdz[l, k] + edge_height[2, e] * dπdz[r, k]))
        end
        pg[e, k] = ICON_CPD * t * dπ
    end
end

"""
For `taylor_gradient`: `c₂ = ½ ∂²π'/∂z²` of the layers from the hydrostatic balance of the deviation
`θ'` of `θ` from the reference state, `½ (A ∂θ'/∂z - B θ')` with `A = g/(c_p θ_ref²)` and
`B = 2g/(c_p θ_ref³) ∂θ_ref/∂z` (ICON `z_dexner_dz_c(2)`), and `∂θ/∂z` of the layers, both across the
layer from the interfaces (`interface_values`).
"""
@kernel function icon_exner_curvature_kernel!(c2, dθdz, @Const(θ), @Const(θref), @Const(A), @Const(B), @Const(wlow),
                                              @Const(q), @Const(dz), nlev)
    i, k = @index(Global, NTuple)
    @inbounds begin
        t(j) = θ[i, j]
        tp(j) = θ[i, j] - θref[i, j]
        lo, hi = interface_values(t, i, k, wlow, q, nlev)
        lp, hp = interface_values(tp, i, k, wlow, q, nlev)
        dθdz[i, k] = (hi - lo) / dz[i, k]
        c2[i, k] = (A[i, k] * (hp - lp) / dz[i, k] - B[i, k] * tp(k)) / 2
    end
end

"""
Hydrostatic correction of the Taylor pressure gradient (`taylor_gradient`) at the edges `pe`, layers `pk`
below the higher ground of their cells: `pg += c_p θ_e dist g/c_p (θ₂ - θ₁) 4/(θ₁ + θ₂)² / d`, `θ_s`
of cell `s` at the height of the expansion of the lowest layer (ICON `z_hydro_corr`).
"""
@kernel function icon_gradp_extrapolation_kernel!(pg, @Const(θe), @Const(θ), @Const(dθdz), @Const(pe), @Const(pk),
                                                  @Const(dist), @Const(zlev), @Const(zwt), @Const(edge_cells),
                                                  @Const(inverse_dual))
    p = @index(Global)
    @inbounds begin
        e, k = pe[p], pk[p]
        θs(s) = (c = edge_cells[s, e]; j = zlev[s, e, 1]; θ[c, j] + zwt[s, e, 1] * dθdz[c, j])
        θ1, θ2 = θs(1), θs(2)
        pg[e, k] += θe[e, k] * dist[p] * ICON_GRAV * inverse_dual[e] * (θ2 - θ1) * 4 / (θ1 + θ2)^2
    end
end

"`vn = vn0 + Δt (a T_now + b T_new - pg + hd)`."
@kernel function icon_vn_update_kernel!(vn, @Const(vn0), @Const(tnow), @Const(tnew), @Const(pg), @Const(hd), dt, a, b)
    e, k = @index(Global, NTuple)
    @inbounds vn[e, k] = vn0[e, k] + dt * (a * tnow[e, k] + b * tnew[e, k] - pg[e, k] + hd[e, k])
end

"Averaged normal velocity `v̄_e = Σ_n w_{n,e} vn_{edges[n,e]}` of the mass flux (ICON `z_vn_avg`)."
@kernel function icon_edge_average_kernel!(va, @Const(vn), @Const(edges), @Const(weights))
    e, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for n in axes(edges, 1)
        s += weights[n, e] * vn[edges[n, e], k]
    end
    @inbounds va[e, k] = s
end

"Mass and `ρθ` fluxes `ρ_e vn Δz_e`, `ρ_e θ_e vn Δz_e` (with the averaged `vn`; `dze` 1 for flat ground)."
@kernel function icon_flux_kernel!(F, Fθ, @Const(vn), @Const(ρe), @Const(θe), @Const(dze))
    e, k = @index(Global, NTuple)
    @inbounds f = ρe[e, k] * vn[e, k] * at_ik(dze, e, k)
    @inbounds F[e, k] = f
    @inbounds Fθ[e, k] = f * θe[e, k]
end

"`G = scale (δ_r - δ_l)/d_e`, the gradient of a cell field at the edges."
@kernel function icon_edge_gradient_kernel!(G, @Const(δ), @Const(edge_cells), @Const(inverse_dual), scale)
    e, k = @index(Global, NTuple)
    @inbounds G[e, k] = scale * (δ[edge_cells[2, e], k] - δ[edge_cells[1, e], k]) * inverse_dual[e]
end

"""
`ρ` and `θ` at the interior interfaces for the column solve: the time average
`a0 x^n + as x*` of the cells below and above, linearly interpolated, minus
`(Δt/2) w* ∂x/∂z` (backtrajectory, `half_dt = Δt/2`); with `as = half_dt = 0`
the linear interpolation of `n` (predictor), `w*` contravariant (`wch` the correction). `θp`
is the interpolated `θ - θ_ref` without the backtrajectory term (ICON `z_theta_v_pr_ic`).
"""
@kernel function icon_interface_kernel!(θi, ρi, θp, @Const(ρ0), @Const(ρθ0), @Const(ρs), @Const(ρθs), @Const(ws),
                                        @Const(wch), @Const(θref), @Const(dzh), @Const(wlow), a0, as, half_dt, nlev)
    i = @index(Global)
    @inbounds begin
        θi[i, 1] = θi[i, nlev + 1] = ρi[i, 1] = ρi[i, nlev + 1] = θp[i, 1] = θp[i, nlev + 1] = 0.0
        ρl = a0 * ρ0[i, 1] + as * ρs[i, 1]
        θl = a0 * ρθ0[i, 1] / ρ0[i, 1] + as * ρθs[i, 1] / ρs[i, 1]
        for j in 2:nlev
            ρu = a0 * ρ0[i, j] + as * ρs[i, j]
            θu = a0 * ρθ0[i, j] / ρ0[i, j] + as * ρθs[i, j] / ρs[i, j]
            c = half_dt * (ws[i, j] - wch[i, j]) / dzh[i, j]
            wl = wlow[i, j]
            ρi[i, j] = wl * ρl + (1 - wl) * ρu - c * (ρu - ρl)
            θi[i, j] = wl * θl + (1 - wl) * θu - c * (θu - θl)
            θp[i, j] = wl * (θl - θref[i, j - 1]) + (1 - wl) * (θu - θref[i, j])
            ρl, θl = ρu, θu
        end
    end
end

# ICON's vertically implicit step (`solve_nh`): `w`, `ρ`, the Exner pressure `πn` and `ρθ` at `t + τ`
# from the state of `n` (`w0`, `ρ0`, `ρθ0`, the prognostic Exner pressure `π0`), the explicit horizontal
# parts `ρexpl`, `ρθexpl`, the advection `tw` of `w` and `ρ`, `θ` at the interfaces (`ρi`, `θi`,
# `θp = θ - θ_ref`). The vertical pressure gradient is split around the reference state,
# `θ_j ∂π'/∂z + θ'_j ∂π_ref/∂z` with the analytic `∂π_ref/∂z = -g/(c_p θ_ref)` (`dπref`) that balances
# gravity, the first part off-centred with `β`; `π` is linearized around `π0`,
# `π = π0 + (R/c_v) π0/ρθ0 Δ(ρθ)`, and `ρθ` follows from it, so that `ρθ` is conserved in flux form.
# Over orography the explicit mass flux has the contravariant correction, `ρ_j ((1 - β) w0_j - w_c,j)`,
# and `w = w_c` at the ground. Klemp's damping of `w`, `w/(1 + τ r_j)`, after the solve. As in ICON all but the Thomas sweeps run
# over cells × levels: the explicit parts of the layers, the tridiagonal system of the interfaces, the
# sweeps of the columns, the update of the layers.

"Explicit mass flux `ρ_j ((1 - β) w0_j - w_c,j)` at interface `j` (zero at the ground and the lid)."
@inline icon_explicit_flux(ρi, w0, wch, β, i, j, nlev) = 2 <= j <= nlev ? ρi[i, j] * ((1 - β) * w0[i, j] - wch[i, j]) : 0.0

"`α = β θ_j ρ_j` of the implicit mass flux at interface `j` (zero at the ground and the lid)."
@inline icon_implicit_weight(θi, ρi, β, i, j, nlev) = 2 <= j <= nlev ? β * θi[i, j] * ρi[i, j] : 0.0

"ICON's `z_beta` of layer `k` (`zβ`), its explicit `ρ` (`ρx`) and `π'` (`πx`)."
@kernel function icon_column_explicit_kernel!(zβ, ρx, πx, @Const(w0), @Const(wch), @Const(ρθ0), @Const(π0),
                                              @Const(θi), @Const(ρi), @Const(ρexpl), @Const(ρθexpl), @Const(πref),
                                              @Const(dz), τ, β, nlev)
    i, k = @index(Global, NTuple)
    @inbounds begin
        Ml, Mu = icon_explicit_flux(ρi, w0, wch, β, i, k, nlev), icon_explicit_flux(ρi, w0, wch, β, i, k + 1, nlev)
        b = τ * ICON_RD * π0[i, k] / (ICON_CVD * ρθ0[i, k] * dz[i, k])
        zβ[i, k] = b
        ρx[i, k] = ρexpl[i, k] - τ / dz[i, k] * (Mu - Ml)
        πx[i, k] = π0[i, k] - πref[i, k] + ICON_RD * π0[i, k] / (ICON_CVD * ρθ0[i, k]) * (ρθexpl[i, k] - ρθ0[i, k]) -
                   b * (θi[i, k + 1] * Mu - θi[i, k] * Ml)
    end
end

"The tridiagonal system for `w` at interface `j`: `lower`, `diag`, the upper diagonal in `cp_`, the right side in `dp_`."
@kernel function icon_column_matrix_kernel!(lower, diag, cp_, dp_, @Const(zβ), @Const(πx), @Const(w0), @Const(π0),
                                            @Const(θi), @Const(θp), @Const(ρi), @Const(tw), @Const(πref),
                                            @Const(dπref), @Const(dzh), τ, β, nlev)
    i, j = @index(Global, NTuple)
    @inbounds if 2 <= j <= nlev
        bl, bu = zβ[i, j - 1], zβ[i, j]
        γ = τ * ICON_CPD * β * θi[i, j] / dzh[i, j]
        lower[i, j] = -γ * bl * icon_implicit_weight(θi, ρi, β, i, j - 1, nlev)
        cp_[i, j] = -γ * bu * icon_implicit_weight(θi, ρi, β, i, j + 1, nlev)
        diag[i, j] = 1 + γ * icon_implicit_weight(θi, ρi, β, i, j, nlev) * (bl + bu)
        pg = (1 - β) * θi[i, j] * ((π0[i, j] - πref[i, j]) - (π0[i, j - 1] - πref[i, j - 1])) / dzh[i, j] +
             θp[i, j] * dπref[i, j]
        dp_[i, j] = w0[i, j] + τ * (tw[i, j] - ICON_CPD * pg) - γ * (πx[i, j] - πx[i, j - 1])
    else
        lower[i, j] = cp_[i, j] = dp_[i, j] = 0.0
        diag[i, j] = 1.0
    end
end

"The Thomas sweeps of a column for `w` over the interior interfaces, then Klemp's damping; `w = w_c` at the ground."
@kernel function icon_column_sweep_kernel!(w, cp_, dp_, @Const(lower), @Const(diag), @Const(rayleigh), @Const(wch), τ,
                                           nlev)
    i = @index(Global)
    @inbounds begin
        for j in 2:nlev
            l = lower[i, j]
            denom = diag[i, j] - l * cp_[i, j - 1]
            cp_[i, j] = cp_[i, j] / denom
            dp_[i, j] = (dp_[i, j] - l * dp_[i, j - 1]) / denom
        end
        w[i, nlev + 1] = 0.0
        wu = 0.0
        for j in nlev:-1:2
            wj = dp_[i, j] - cp_[i, j] * wu
            w[i, j] = wj / (1 + τ * rayleigh[j])
            wu = wj
        end
        w[i, 1] = wch[i, 1]
    end
end

"`ρ`, `πn` and `ρθ` of layer `k` from `w`."
@kernel function icon_column_update_kernel!(ρ, ρθ, πn, @Const(w), @Const(zβ), @Const(ρx), @Const(πx), @Const(ρθ0),
                                            @Const(π0), @Const(θi), @Const(ρi), @Const(πref), @Const(dz), τ, β, nlev)
    i, k = @index(Global, NTuple)
    @inbounds begin
        wl, wu = k > 1 ? w[i, k] : 0.0, w[i, k + 1]
        ρ[i, k] = ρx[i, k] - β * τ / dz[i, k] * (ρi[i, k + 1] * wu - ρi[i, k] * wl)
        π = πx[i, k] + πref[i, k] - zβ[i, k] * (icon_implicit_weight(θi, ρi, β, i, k + 1, nlev) * wu -
                                             icon_implicit_weight(θi, ρi, β, i, k, nlev) * wl)
        πn[i, k] = π
        ρθ[i, k] = ρθ0[i, k] * ((π / π0[i, k] - 1) * ICON_CVD / ICON_RD + 1)
    end
end

"""
Divergence `δ += s_k ∂(w - w_c)/∂z` of the cells (the 3D divergence of the divergence damping, `s` =
scalfac_dd3d), `w_c` the contravariant correction at the interfaces (`wch`, zero for flat ground): over
orography the terrain-following flow `w = w_c` has no divergence.
"""
@kernel function icon_dwdz_kernel!(δ, @Const(w), @Const(wch), @Const(dz), @Const(s))
    i, k = @index(Global, NTuple)
    @inbounds δ[i, k] += s[k] * ((w[i, k + 1] - wch[i, k + 1]) - (w[i, k] - wch[i, k])) / dz[i, k]
end

"`vn += scale_k (δ_r - δ_l)/d_e`, the divergence damping with the coefficient `scale` per layer."
@kernel function icon_divdamp_kernel!(vn, @Const(δ), @Const(edge_cells), @Const(inverse_dual), @Const(scale))
    e, k = @index(Global, NTuple)
    @inbounds vn[e, k] += scale[k] * (δ[edge_cells[2, e], k] - δ[edge_cells[1, e], k]) * inverse_dual[e]
end

"""
Divergence damping of the predictor state `ys` added to `vn`: the gradient of
`D = ∇⋅vn* (+ ∂w*/∂z)`, second order with `divdamp_o2`, fourth order with
`divdamp` through `∇(∇⋅∇D)` (ICON `z_graddiv_vn`, `z_graddiv2_vn`).
"""
function divergence_damping!(vn, m::ICONNonhydrostatic, ys)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne = num_cells(m.grid), num_edges(m.grid)
    launch!(m, icon_cell_sum_kernel!, (nc, nlev), w.δ, ys.vn, m.cell_edges, m.divergence)
    launch!(m, icon_dwdz_kernel!, (nc, nlev), w.δ, ys.w, w.wch, m.dz, m.divdamp_3d)
    launch!(m, icon_edge_gradient_kernel!, (ne, nlev), w.G, w.δ, m.edge_cells, m.inverse_dual, 1.0)
    launch!(m, icon_cell_sum_kernel!, (nc, nlev), w.δ2, w.G, m.cell_edges, m.divergence)
    launch!(m, icon_divdamp_kernel!, (ne, nlev), vn, w.δ, m.edge_cells, m.inverse_dual, m.divdamp_o2)
    launch!(m, icon_divdamp_kernel!, (ne, nlev), vn, w.δ2, m.edge_cells, m.inverse_dual, m.divdamp)
    return vn
end

"""
Horizontal flux divergence of `vn` with `ρ_e`, `θ_e`, then the column solve from
the base state `y0` into `y` and the Exner pressure `πn`; `ρ`, `θ` at the
interfaces of `y0` (predictor) or of the time average of `y0` and the
predictor state `ys` (`corrector`).
"""
function fast_waves!(m::ICONNonhydrostatic, y, πn, y0, ys, tw, dt, corrector::Bool)
    w, v, nlev = m.work, m.vertical, num_layers(m.vgrid)
    nc, ne = num_cells(m.grid), num_edges(m.grid)
    section(m, "edgecomp") do
        launch!(m, icon_edge_average_kernel!, (ne, nlev), w.va, y.vn, m.flux_edges, m.flux_weights)
        launch!(m, icon_flux_kernel!, (ne, nlev), w.F, w.Fθ, w.va, w.ρe, w.θe, m.orography ? m.metric.dze : 1.0)
        contravariant_correction!(m, y.vn)
    end
    section(m, "vimpl") do
        launch!(m, divergence_kernel!, (nc, nlev), w.ρexpl, w.ρθexpl, y0.ρ, y0.ρθ, w.F, w.Fθ, w.hρθ, m.cell_edges,
                m.divergence, dt, m.orography ? m.metric.rdz : 1.0)
        a0, as = corrector ? m.rth_weights : (1.0, 0.0)
        launch!(m, icon_interface_kernel!, nc, w.θi, w.ρi, w.θp, y0.ρ, y0.ρθ, ys.ρ, ys.ρθ, ys.w, w.wch, v.θref, m.dzh,
                m.wlow, a0, as, corrector ? dt / 2 : 0.0, nlev)
        launch!(m, icon_column_explicit_kernel!, (nc, nlev), w.zβ, w.ρx, w.πx, y0.w, w.wch, y0.ρθ, w.π0, w.θi, w.ρi,
                w.ρexpl, w.ρθexpl, v.πref, m.dz, dt, m.beta, nlev)
        launch!(m, icon_column_matrix_kernel!, (nc, nlev + 1), w.lower, w.diag, w.cprime, w.dprime, w.zβ, w.πx,
                y0.w, w.π0, w.θi, w.θp, w.ρi, tw, v.πref, v.dπref, m.dzh, dt, m.beta, nlev)
        launch!(m, icon_column_sweep_kernel!, nc, y.w, w.cprime, w.dprime, w.lower, w.diag, m.rayleigh, w.wch, dt, nlev)
        launch!(m, icon_column_update_kernel!, (nc, nlev), y.ρ, y.ρθ, πn, y.w, w.zβ, w.ρx, w.πx, y0.ρθ, w.π0, w.θi,
                w.ρi, v.πref, m.dz, dt, m.beta, nlev)
    end
    return y
end
