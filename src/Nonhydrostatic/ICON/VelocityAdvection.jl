# Velocity tendencies of ICON (`mo_velocity_advection`): RBF tangential velocity, kinetic energy, vorticity flux, advection of w.

"Tangential velocity (RBF) and kinetic energy at the edges."
@kernel function icon_edge_kinetic_kernel!(vt, Ke, @Const(vn), @Const(rbf_edges), @Const(rbf))
    e, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(rbf_edges, 1)
        s += rbf[j, e] * vn[rbf_edges[j, e], k]
    end
    @inbounds vt[e, k] = s
    @inbounds Ke[e, k] = (vn[e, k]^2 + s^2) / 2
end

"Cell sum `Σ_j c_{j,i} x_{e_j}` of an edge field (bilinear cell average, divergence)."
@kernel function icon_cell_sum_kernel!(y, @Const(x), @Const(cell_edges), @Const(c))
    i, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(cell_edges, 1)
        s += c[j, i] * x[cell_edges[j, i], k]
    end
    @inbounds y[i, k] = s
end

"""
Velocity tendency of `vn`: vorticity flux, kinetic-energy gradient (`coeff_gradekin`), vertical
advection with the contravariant `w - w_c` of the interfaces (`wch`: `w_c` at the interfaces), zero
at the ground, averaged to the full level as in ICON (`z_w_con_c_full`); `dz`, `wlow`, `q`
(wgtfacq_e) at the edges.
"""
@kernel function icon_vn_tendency_kernel!(tvn, @Const(vn), @Const(vt), @Const(Ke), @Const(K), @Const(ζv), @Const(w),
                                          @Const(wch), @Const(edge_cells), @Const(edge_vertices), @Const(edge_height),
                                          @Const(gradient), @Const(edge_coriolis), @Const(dz), @Const(wlow), @Const(q),
                                          nlev)
    e, k = @index(Global, NTuple)
    @inbounds begin
        l, r = edge_cells[1, e], edge_cells[2, e]
        ζ = (ζv[edge_vertices[1, e], k] + ζv[edge_vertices[2, e], k]) / 2
        g1, g2 = gradient[1, e], gradient[2, e]
        grad_K = Ke[e, k] * (g1 - g2) + g2 * K[r, k] - g1 * K[l, k]
        wcon(c) = ((k > 1 ? w[c, k] - wch[c, k] : 0.0) + (w[c, k + 1] - wch[c, k + 1])) / 2
        we = edge_height[1, e] * wcon(l) + edge_height[2, e] * wcon(r)
        # vn at the ground quadratically extrapolated (wgtfacq_e), at the lid that of the top layer
        vlow = k > 1 ? wlow[e, k] * vn[e, k - 1] + (1 - wlow[e, k]) * vn[e, k] :
               nlev >= 3 ? q[e, 1] * vn[e, 1] + q[e, 2] * vn[e, 2] + q[e, 3] * vn[e, 3] : vn[e, 1]
        vup = k < nlev ? wlow[e, k + 1] * vn[e, k] + (1 - wlow[e, k + 1]) * vn[e, k + 1] : vn[e, k]
        tvn[e, k] = (edge_coriolis[e] + ζ) * vt[e, k] - grad_K - we * (vup - vlow) / dz[e, k]
    end
end

"`w` at the vertices (area weights of the cells)."
@kernel function icon_vertex_w_kernel!(wv, @Const(w), @Const(vertex_cells), @Const(weight))
    v, j = @index(Global, NTuple)
    s = 0.0
    @inbounds for n in axes(vertex_cells, 1)
        c = vertex_cells[n, v]
        c > 0 && (s += weight[n, v] * w[c, j])
    end
    @inbounds wv[v, j] = s
end

"`vn_ie ∂w/∂n + vt_ie ∂w/∂t` at the edges of the interior interfaces."
@kernel function icon_edge_w_gradient_kernel!(gw, @Const(vn), @Const(vt), @Const(w), @Const(wv), @Const(edge_cells),
                                              @Const(edge_vertices), @Const(inverse_dual), @Const(inverse_primal),
                                              @Const(tangent_sign), @Const(wlow), nlev)
    e, j = @index(Global, NTuple)
    @inbounds if 2 <= j <= nlev
        vni = wlow[e, j] * vn[e, j - 1] + (1 - wlow[e, j]) * vn[e, j]
        vti = wlow[e, j] * vt[e, j - 1] + (1 - wlow[e, j]) * vt[e, j]
        dn = (w[edge_cells[2, e], j] - w[edge_cells[1, e], j]) * inverse_dual[e]
        dt = tangent_sign[e] * (wv[edge_vertices[2, e], j] - wv[edge_vertices[1, e], j]) * inverse_primal[e]
        gw[e, j] = vni * dn + vti * dt
    else
        gw[e, j] = 0.0
    end
end

"""
Advection of `w` at the interior interfaces: `-Σ_e b_{i,e} gw_e - (w - w_c) ∂w/∂z` with
the second-order `∂w/∂z` of the unequal layers (`c1`, `c2`: coeff1_dwdz, coeff2_dwdz), `wch` the
contravariant correction `w_c` at the interfaces.
"""
@kernel function icon_w_tendency_kernel!(tw, @Const(gw), @Const(w), @Const(wch), @Const(cell_edges), @Const(bilinear),
                                         @Const(c1), @Const(c2), nlev)
    i, j = @index(Global, NTuple)
    @inbounds if 2 <= j <= nlev
        s = 0.0
        for n in axes(cell_edges, 1)
            s += bilinear[n, i] * gw[cell_edges[n, i], j]
        end
        tw[i, j] = -s - (w[i, j] - wch[i, j]) *
                        (c1[i, j] * w[i, j + 1] - c2[i, j] * w[i, j - 1] + (c2[i, j] - c1[i, j]) * w[i, j])
    else
        tw[i, j] = 0.0
    end
end

"""
Contravariant correction `w_c = vn ∂z/∂n + vt ∂z/∂t` at the edges (full levels), `vt` by the RBF
(ICON `z_w_concorr_me`).
"""
@kernel function icon_concorr_edge_kernel!(wce, @Const(vn), @Const(rbf_edges), @Const(rbf), @Const(ddxn), @Const(ddxt))
    e, k = @index(Global, NTuple)
    s = 0.0
    @inbounds for j in axes(rbf_edges, 1)
        s += rbf[j, e] * vn[rbf_edges[j, e], k]
    end
    @inbounds wce[e, k] = vn[e, k] * ddxn[e, k] + s * ddxt[e, k]
end

"""
The contravariant correction of `vn` over orography at the full levels (`work.wcf`) and the
interfaces (`work.wch`) of the cells, the edge values averaged with the bilinear weights (ICON
`w_concorr_c`); zero for flat ground.
"""
function contravariant_correction!(m::ICONNonhydrostatic, vn)
    m.orography || return nothing
    w, x, nlev = m.work, m.metric, num_layers(m.vgrid)
    nc, ne = num_cells(m.grid), num_edges(m.grid)
    launch!(m, icon_concorr_edge_kernel!, (ne, nlev), w.G, vn, m.rbf_edges, m.rbf, x.ddxn, x.ddxt)
    launch!(m, icon_cell_sum_kernel!, (nc, nlev), w.wcf, w.G, m.cell_edges, m.bilinear)
    launch!(m, concorr_interface_kernel!, (nc, nlev + 1), w.wch, w.wcf, m.wlow, m.vertical.wgtfacq, nlev)
    return nothing
end

"""
Velocity tendencies of the state `y` into `tvn`, `tw` (with `vn_only` only
`tvn`, ICON `lvn_only`); θ, π, `vt` of `y` stay in the work arrays.
"""
function velocity_tendencies!(tvn, tw, m::ICONNonhydrostatic, y; vn_only::Bool=false)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne, nv = num_cells(m.grid), num_edges(m.grid), num_vertices(m.grid)
    launch!(m, icon_thermo_kernel!, (nc, nlev), w.θ, w.π, y.ρ, y.ρθ)
    launch!(m, icon_edge_kinetic_kernel!, (ne, nlev), w.vt, w.Ke, y.vn, m.rbf_edges, m.rbf)
    launch!(m, icon_cell_sum_kernel!, (nc, nlev), w.K, w.Ke, m.cell_edges, m.bilinear)
    launch!(m, vorticity_kernel!, (nv, nlev), w.ζv, y.vn, m.vertex_edges, m.curl)
    contravariant_correction!(m, y.vn)
    x = m.metric
    launch!(m, icon_vn_tendency_kernel!, (ne, nlev), tvn, y.vn, w.vt, w.Ke, w.K, w.ζv, y.w, w.wch, m.edge_cells,
            m.edge_vertices, m.edge_height, m.gradient, m.edge_coriolis, x.dze, x.wlowe, x.qe, nlev)
    vn_only && return nothing
    launch!(m, icon_vertex_w_kernel!, (nv, nlev + 1), w.wv, y.w, m.vertex_cells, m.vertex_weight)
    launch!(m, icon_edge_w_gradient_kernel!, (ne, nlev + 1), w.gw, y.vn, w.vt, y.w, w.wv, m.edge_cells,
            m.edge_vertices, m.inverse_dual, m.inverse_primal, m.tangent_sign, x.wlowe, nlev)
    launch!(m, icon_w_tendency_kernel!, (nc, nlev + 1), tw, w.gw, y.w, w.wch, m.cell_edges, m.bilinear, m.vertical.c1dwdz,
            m.vertical.c2dwdz, nlev)
    return nothing
end
