# Fast waves of ICON (`solve_nh`): Exner pressure, Miura reconstruction of ρ and θ at the edges, pressure gradient, fluxes, vertical solve.

"θ and the Exner pressure at the cells."
@kernel function icon_thermo_kernel!(θ, πe, @Const(ρ), @Const(ρθ))
    i, k = @index(Global, NTuple)
    @inbounds θ[i, k] = ρθ[i, k] / ρ[i, k]
    @inbounds πe[i, k] = (Rd * ρθ[i, k] / P0)^(Rd / Cvd)
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
gradient `c_p θ_e ∂π/∂n` of the extrapolated Exner pressure.
"""
@kernel function icon_miura_kernel!(ρe, θe, pg, @Const(vn), @Const(vt), @Const(ρ), @Const(θ), @Const(πex),
                                    @Const(g1), @Const(g2), @Const(g3), @Const(h1), @Const(h2), @Const(h3),
                                    @Const(edge_cells), @Const(edge_vectors), @Const(inverse_dual), half_dt_a)
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        u, v = vn[e, k], vt[e, k]
        up, o = u >= 0 ? (l, 0) : (r, 3)
        d1 = edge_vectors[o + 1, e] - half_dt_a * (u * edge_vectors[7, e] + v * edge_vectors[10, e])
        d2 = edge_vectors[o + 2, e] - half_dt_a * (u * edge_vectors[8, e] + v * edge_vectors[11, e])
        d3 = edge_vectors[o + 3, e] - half_dt_a * (u * edge_vectors[9, e] + v * edge_vectors[12, e])
        ρe[e, k] = ρ[up, k] + g1[up, k] * d1 + g2[up, k] * d2 + g3[up, k] * d3
        t = θ[up, k] + h1[up, k] * d1 + h2[up, k] * d2 + h3[up, k] * d3
        θe[e, k] = t
        pg[e, k] = Cpd * t * (πex[r, k] - πex[l, k]) * inverse_dual[e]
    end
end

"`vn = vn0 + Δt (a T_now + b T_new - pg + hd)`."
@kernel function icon_vn_update_kernel!(vn, @Const(vn0), @Const(tnow), @Const(tnew), @Const(pg), @Const(hd), dt, a, b)
    e, k = @index(Global, NTuple)
    @inbounds vn[e, k] = vn0[e, k] + dt * (a * tnow[e, k] + b * tnew[e, k] - pg[e, k] + hd[e, k])
end

"Mass and `ρθ` fluxes `ρ_e vn`, `ρ_e θ_e vn`."
@kernel function icon_flux_kernel!(F, Fθ, @Const(vn), @Const(ρe), @Const(θe))
    e, k = @index(Global, NTuple)
    @inbounds f = ρe[e, k] * vn[e, k]
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
the linear interpolation of `n` (predictor).
"""
@kernel function icon_interface_kernel!(θi, ρi, @Const(ρ0), @Const(ρθ0), @Const(ρs), @Const(ρθs), @Const(ws),
                                        @Const(dzh), @Const(wlow), a0, as, half_dt, nlev)
    i = @index(Global)
    @inbounds begin
        θi[i, 1] = θi[i, nlev + 1] = ρi[i, 1] = ρi[i, nlev + 1] = 0.0
        ρl = a0 * ρ0[i, 1] + as * ρs[i, 1]
        θl = a0 * ρθ0[i, 1] / ρ0[i, 1] + as * ρθs[i, 1] / ρs[i, 1]
        for j in 2:nlev
            ρu = a0 * ρ0[i, j] + as * ρs[i, j]
            θu = a0 * ρθ0[i, j] / ρ0[i, j] + as * ρθs[i, j] / ρs[i, j]
            c = half_dt * ws[i, j] / dzh[j]
            ρi[i, j] = wlow[j] * ρl + (1 - wlow[j]) * ρu - c * (ρu - ρl)
            θi[i, j] = wlow[j] * θl + (1 - wlow[j]) * θu - c * (θu - θl)
            ρl, θl = ρu, θu
        end
    end
end

"Divergence `δ += s_k ∂w/∂z` of the cells (the 3D divergence of the divergence damping, `s` = scalfac_dd3d)."
@kernel function icon_dwdz_kernel!(δ, @Const(w), @Const(dz), @Const(s))
    i, k = @index(Global, NTuple)
    @inbounds δ[i, k] += s[k] * (w[i, k + 1] - w[i, k]) / dz[k]
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
    launch!(m, icon_dwdz_kernel!, (nc, nlev), w.δ, ys.w, m.dz, m.divdamp_3d)
    launch!(m, icon_edge_gradient_kernel!, (ne, nlev), w.G, w.δ, m.edge_cells, m.inverse_dual, 1.0)
    launch!(m, icon_cell_sum_kernel!, (nc, nlev), w.δ2, w.G, m.cell_edges, m.divergence)
    launch!(m, icon_divdamp_kernel!, (ne, nlev), vn, w.δ, m.edge_cells, m.inverse_dual, m.divdamp_o2)
    launch!(m, icon_divdamp_kernel!, (ne, nlev), vn, w.δ2, m.edge_cells, m.inverse_dual, m.divdamp)
    return vn
end

"""
Horizontal flux divergence of `vn` with `ρ_e`, `θ_e`, then the column solve from
the base state `y0` into `y`; `ρ`, `θ` at the interfaces of `y0` (predictor) or
of the time average of `y0` and the predictor state `ys` (`corrector`).
"""
function fast_waves!(m::ICONNonhydrostatic, y, y0, ys, tw, dt, corrector::Bool)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne = num_cells(m.grid), num_edges(m.grid)
    launch!(m, icon_flux_kernel!, (ne, nlev), w.F, w.Fθ, y.vn, w.ρe, w.θe)
    launch!(m, divergence_kernel!, (nc, nlev), w.ρexpl, w.ρθexpl, y0.ρ, y0.ρθ, w.F, w.Fθ, w.hρθ, m.cell_edges,
            m.divergence, dt)
    a0, as = corrector ? m.rth_weights : (1.0, 0.0)
    launch!(m, icon_interface_kernel!, nc, w.θi, w.ρi, y0.ρ, y0.ρθ, ys.ρ, ys.ρθ, ys.w, m.dzh, m.wlow, a0, as,
            corrector ? dt / 2 : 0.0, nlev)
    launch!(m, vertical_solve_kernel!, nc, y.w, y.ρ, y.ρθ, w.cprime, w.dprime, y0.w, y0.ρθ, w.π0, w.γ0, w.θ, w.θi,
            w.ρi, tw, w.ρexpl, w.ρθexpl, m.dz, m.dzh, m.wlow, m.rayleigh, w.wmass, dt, m.beta, nlev, true, true)
    return y
end
