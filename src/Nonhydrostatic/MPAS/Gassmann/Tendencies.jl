# Explicit vn tendency of MPAS-Gassmann (PV flux, ∇K, pressure gradient, vertical coupling) and its update.

"`w̃_e = M_e/ρ_e` at interface `j`, `M_e = Σ_i (d_{e,i}/d_e) ρ_i w_i` (`ρ = 1` without `massweighted`)."
@inline function edge_w(w, ρ, e, l, r, j, edge_weight, wlow, massweighted)
    @inbounds begin
        ρhl = massweighted ? wlow[j] * ρ[l, j - 1] + (1 - wlow[j]) * ρ[l, j] : 1.0
        ρhr = massweighted ? wlow[j] * ρ[r, j - 1] + (1 - wlow[j]) * ρ[r, j] : 1.0
        M = edge_weight[1, e] * ρhl * w[l, j] + edge_weight[2, e] * ρhr * w[r, j]
        return M / (edge_weight[1, e] * ρhl + edge_weight[2, e] * ρhr)
    end
end

"Explicit tendency of `vn` (PV flux, kinetic energy and, with `pressure = 1`, the pressure gradient, vertical advection)."
@kernel function vn_tendency_kernel!(tvn, @Const(vn), @Const(Fs), @Const(w), @Const(ρ), @Const(θ), @Const(πe), @Const(K),
                                     @Const(qe), @Const(η), @Const(edge_cells), @Const(edge_weight),
                                     @Const(inverse_dual), @Const(trisk_index), @Const(trisk), @Const(dz), @Const(dzh),
                                     @Const(wlow), nlev, vertical, pressure)
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        s = 0.0
        for j in axes(trisk, 2)
            s += trisk[e, j] * Fs[trisk_index[e, j, 1], k] * (qe[trisk_index[e, j, 2], k] + qe[trisk_index[e, j, 3], k])
        end
        θe = edge_weight[1, e] * θ[l, k] + edge_weight[2, e] * θ[r, k]
        t = s - inverse_dual[e] * (K[r, k] - K[l, k] + pressure * Cpd * θe * (πe[r, k] - πe[l, k]))
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

"`tvn -= c_p θ_e ∂_n π` with `θ_e` the edge mean of `θ` (the pressure gradient of the ICON-IAP step)."
@kernel function pressure_gradient_kernel!(tvn, @Const(θ), @Const(πe), @Const(edge_cells), @Const(edge_weight),
                                           @Const(inverse_dual))
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        θe = edge_weight[1, e] * θ[l, k] + edge_weight[2, e] * θ[r, k]
        tvn[e, k] -= inverse_dual[e] * Cpd * θe * (πe[r, k] - πe[l, k])
    end
end

"`x = (x + x1)/2`."
@kernel function mean_kernel!(x, @Const(x1))
    i, k = @index(Global, NTuple)
    @inbounds x[i, k] = (x[i, k] + x1[i, k]) / 2
end

"`π_m = (R/c_p - 1/2) π_old + (3/2 - R/c_p) π` (the time-off-centred Exner pressure of ICON-IAP), then `π_old = π`."
@kernel function exner_mean_kernel!(πm, πold, @Const(π0))
    i, k = @index(Global, NTuple)
    @inbounds begin
        πm[i, k] = (Rd / Cpd - 0.5) * πold[i, k] + (1.5 - Rd / Cpd) * π0[i, k]
        πold[i, k] = π0[i, k]
    end
end
